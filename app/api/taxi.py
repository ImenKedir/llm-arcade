from typing import Optional, Dict, Any, AsyncGenerator, Annotated
import logging
import gymnasium as gym
import json
import asyncio
import random
from pydantic import BaseModel, Field
from typing import Literal

from fastapi import APIRouter, Request, Header, HTTPException, Depends
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from app.services.openrouter_client import OpenRouterClient, get_openrouter_client
from app.services.llm_agent import LLMGameAgent

# Set up logger
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/taxi",
    tags=["taxi"],
)

# Set up templates
templates = Jinja2Templates(directory="templates")

# Define Taxi action schema
class TaxiAction(BaseModel):
    """Schema for Taxi game actions"""
    action: Literal["south", "north", "east", "west", "pickup", "dropoff"] = Field(
        ...,
        description="The action to take in the taxi game: 'south', 'north', 'east', 'west' for movement, 'pickup' to pick up passenger, 'dropoff' to drop off passenger."
    )
    reasoning: str = Field(
        ...,
        description="Explanation of why this action was chosen based on the current game state."
    )
    
    def to_env_action(self) -> int:
        """Convert the action string to the corresponding action ID for the Taxi environment"""
        action_map = {
            "south": 0,
            "north": 1,
            "east": 2,
            "west": 3,
            "pickup": 4,
            "dropoff": 5
        }
        return action_map[self.action]

# Game rules for the LLM
TAXI_RULES = """
In the Taxi environment, you are a taxi driver navigating a 5x5 grid world to pick up and drop off passengers.

The grid has four designated locations: Red (R), Green (G), Yellow (Y), and Blue (B).
- The taxi can move in four directions: north, south, east, and west.
- The taxi can pick up a passenger at their current location.
- The taxi can drop off a passenger at their destination.

The goal is to pick up the passenger at their current location and drop them off at their destination.

Your available actions are:
1. "south": Move the taxi one cell down
2. "north": Move the taxi one cell up
3. "east": Move the taxi one cell to the right
4. "west": Move the taxi one cell to the left
5. "pickup": Pick up the passenger (only works when the taxi is at the passenger's location)
6. "dropoff": Drop off the passenger (only works when the taxi is at the destination and the passenger is in the taxi)

The game ends when:
- The passenger is successfully delivered to their destination (success)
- The maximum number of steps is reached (failure)

Choose your actions wisely to minimize the number of steps needed to complete the task.
"""

# Initialize the taxi environment
try:
    env = gym.make("Taxi-v3")
    logger.info("Taxi environment initialized successfully")
except Exception as e:
    logger.error(f"Error initializing Taxi environment: {e}", exc_info=True)
    env = None

# Helper function to render a taxi game state
def render_taxi_state(state: int, reward: float = 0.0, 
                     terminated: bool = False, truncated: bool = False,
                     step_count: int = 0, action: int = None) -> Dict[str, Any]:
    """
    Render a taxi game state in a format suitable for display
    """
    if env is None:
        raise HTTPException(status_code=500, detail="Taxi environment not initialized")
    
    # Decode the state
    decoded = env.unwrapped.decode(state)
    taxi_row, taxi_col, passenger_idx, destination_idx = decoded
    
    # Create a map representation
    map_representation = [
        "+---------+",
        "|R: | : :G|",
        "| : | : : |",
        "| : : : : |",
        "| | : | : |",
        "|Y| : |B: |",
        "+---------+"
    ]
    
    # Convert to a list of characters for easier manipulation
    map_chars = [list(row) for row in map_representation]
    
    # Place the taxi
    if 0 <= taxi_row < 5 and 0 <= taxi_col < 5:
        # Calculate the position in the ASCII map
        # Each cell is 2 characters wide
        map_row = taxi_row + 1  # +1 for the top border
        map_col = 1 + taxi_col * 2  # +1 for the left border, *2 for the width of each cell
        
        # Place the taxi (replace the character with 'T')
        if map_chars[map_row][map_col] in [' ', ':', '|']:
            map_chars[map_row][map_col] = 'T'
    
    # Convert back to strings
    map_representation = [''.join(row) for row in map_chars]
    
    # Determine game status
    game_status = "Playing"
    if terminated:
        if reward > 0:
            game_status = "Success"
        elif reward < 0:
            game_status = "Failure"
        else:
            game_status = "Terminated"
    
    # Convert passenger location to a readable format
    passenger_locations = ["Red", "Green", "Yellow", "Blue", "In Taxi"]
    passenger_location = passenger_locations[passenger_idx] if 0 <= passenger_idx < len(passenger_locations) else "Unknown"
    
    # Convert destination to a readable format
    destinations = ["Red", "Green", "Yellow", "Blue"]
    destination = destinations[destination_idx] if 0 <= destination_idx < len(destinations) else "Unknown"
    
    # Convert action to a readable format if provided
    action_names = ["Move South", "Move North", "Move East", "Move West", "Pickup Passenger", "Drop Off Passenger"]
    action_name = action_names[action] if action is not None and 0 <= action < len(action_names) else "None"
    
    return {
        "state": state,
        "decoded_state": {
            "taxi_row": taxi_row,
            "taxi_col": taxi_col,
            "passenger_location": passenger_location,
            "destination": destination
        },
        "map": map_representation,
        "reward": reward,
        "status": game_status,
        "step_count": step_count,
        "action": action_name,
        "terminated": terminated,
        "truncated": truncated
    }

# Factory function for Taxi LLM agent
def get_taxi_agent(
    openrouter_client: OpenRouterClient = Depends(get_openrouter_client)
) -> LLMGameAgent[TaxiAction]:
    """
    Factory function to create a Taxi LLM agent.
    """
    return LLMGameAgent[TaxiAction](openrouter_client)

# Endpoint to get a random taxi game state
@router.get("", response_class=HTMLResponse)
async def get_taxi_game(
    request: Request,
    hx_request: Annotated[Optional[str], Header()] = None
):
    """
    Render the taxi game page
    """
    try:
        if env is None:
            raise HTTPException(status_code=500, detail="Taxi environment not initialized")
        
        # Reset the environment to get a new game
        observation, info = env.reset()
        
        # Render the initial state
        game_state = render_taxi_state(observation)
        
        # Return HTML response
        return templates.TemplateResponse(
            "taxi.html", {"request": request, "game": game_state}
        )
    except Exception as e:
        logger.error(f"Error in get_taxi_game: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred while generating taxi game")

# Stream taxi games using SSE
@router.get("/stream")
async def stream_taxi_game(
    num_games: int = 10,
    test: bool = False,
    agent: LLMGameAgent[TaxiAction] = Depends(get_taxi_agent),
):
    """
    Stream a series of taxi games with Server-Sent Events (SSE)
    
    Args:
        num_games: Number of games to play (default: 10)
        test: If True, use random actions instead of LLM (default: False)
    """
    return StreamingResponse(
        taxi_stream_generator(num_games, test, agent),
        media_type="text/event-stream"
    )

# Generator function for streaming taxi game updates
async def taxi_stream_generator(
    num_games: int = 10, 
    test: bool = False,
    agent: LLMGameAgent[TaxiAction] = None
) -> AsyncGenerator[str, None]:
    """
    Generate a stream of taxi game states
    
    Args:
        num_games: Number of games to play
        test: If True, use random actions instead of LLM
        agent: The LLM agent to use (if test is False)
    """
    try:
        if env is None:
            yield f"data: {json.dumps({'error': 'Taxi environment not initialized'})}\n\n"
            return
        
        # Track game statistics
        stats = {
            "games_played": 0,
            "successes": 0,
            "failures": 0,
            "total_reward": 0.0
        }
        
        # Limit the number of games to a reasonable range
        num_games = max(1, min(100, num_games))
        
        # Play the specified number of games
        for game_num in range(num_games):
            # Reset the environment to get a new game
            observation, info = env.reset()
            
            # Render the initial state
            game_state = render_taxi_state(observation)
            game_state["game_number"] = game_num + 1
            game_state["total_games"] = num_games
            
            # Send the initial state
            yield f"data: {json.dumps(game_state)}\n\n"
            await asyncio.sleep(0.5)  # Short delay for visualization
            
            # Play until the game is over
            terminated = False
            truncated = False
            step_count = 0
            total_reward = 0
            
            while not terminated and not truncated and step_count < 100:
                # Choose an action
                if test:
                    # Choose a random action
                    action = env.action_space.sample()
                    game_state["llm_reasoning"] = "Using random action (test mode)"
                else:
                    try:
                        # Get action from LLM
                        llm_action = await agent.get_action(
                            action_model=TaxiAction,
                            game_state=game_state,
                            game_name="Taxi",
                            game_rules=TAXI_RULES
                        )
                        
                        # Convert LLM action to environment action
                        action = llm_action.to_env_action()
                        
                        # Include reasoning in the game state
                        game_state["llm_reasoning"] = llm_action.reasoning
                    except Exception as e:
                        logger.error(f"Error getting LLM action: {e}", exc_info=True)
                        # Fallback to random action if LLM fails
                        action = env.action_space.sample()
                        game_state["llm_reasoning"] = f"Error getting LLM action: {str(e)}"
                
                # Execute the action
                observation, reward, terminated, truncated, info = env.step(action)
                step_count += 1
                total_reward += reward
                
                # Render the new state
                game_state = render_taxi_state(
                    observation, reward, terminated, truncated, step_count, action
                )
                game_state["game_number"] = game_num + 1
                game_state["total_games"] = num_games
                
                # Send the updated state
                yield f"data: {json.dumps(game_state)}\n\n"
                await asyncio.sleep(0.5)  # Short delay for visualization
            
            # Update statistics
            stats["games_played"] += 1
            if terminated and reward > 0:
                stats["successes"] += 1
            else:
                stats["failures"] += 1
            stats["total_reward"] += total_reward
            
            # Send the statistics
            yield f"data: {json.dumps({'stats': stats})}\n\n"
            
            # Short delay between games
            await asyncio.sleep(1.0)
        
        # Send a completion message
        yield f"data: {json.dumps({'completed': True, 'stats': stats})}\n\n"
        
    except Exception as e:
        logger.error(f"Error in taxi_stream_generator: {str(e)}", exc_info=True)
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
