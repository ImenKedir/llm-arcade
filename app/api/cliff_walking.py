import gymnasium as gym
import numpy as np
import asyncio
import json
import logging
from fastapi import APIRouter, Request, Depends, Header, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Annotated, AsyncGenerator
from pydantic import BaseModel, Field
from typing import Literal

from app.services.openrouter_client import OpenRouterClient, get_openrouter_client
from app.services.llm_agent import LLMGameAgent

# Setup logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/cliff_walking",
    tags=["cliff_walking"],
)

# Setup templates
templates = Jinja2Templates(directory="templates")

# Define CliffWalking action schema
class CliffWalkingAction(BaseModel):
    """Schema for CliffWalking game actions"""
    action: Literal["left", "down", "right", "up"] = Field(
        ...,
        description="The action to take in the CliffWalking game: 'left', 'down', 'right', or 'up'."
    )
    reasoning: str = Field(
        ...,
        description="Explanation of why this action was chosen based on the current game state."
    )
    
    def to_env_action(self) -> int:
        """Convert the action string to the corresponding action ID for the CliffWalking environment"""
        action_map = {
            "left": 0,
            "down": 1,
            "right": 2,
            "up": 3
        }
        return action_map[self.action]

# Game rules for the LLM
CLIFF_WALKING_RULES = """
In the CliffWalking environment, you are an agent navigating from a start position to a goal position while avoiding falling off a cliff.

The environment is a grid world:
- 'S' is the starting point (bottom-left corner)
- 'G' is the goal (bottom-right corner)
- 'C' represents the cliff (all cells along the bottom row between start and goal)
- Empty cells are safe to walk on

The agent can move in four directions: left, down, right, and up.
Unlike FrozenLake, the movement is deterministic (you always move in the direction you choose).

The goal is to reach the goal position ('G') without falling off the cliff ('C').

Your available actions are:
1. "left": Move one cell to the left
2. "down": Move one cell down
3. "right": Move one cell to the right
4. "up": Move one cell up

The game ends when:
- The agent reaches the goal (success)
- The agent falls off the cliff (failure)
- The maximum number of steps is reached (failure)

Each step gives a reward of -1, and falling off the cliff gives a reward of -100.
Choose your actions wisely to reach the goal with the minimum number of steps.
"""

# Initialize the CliffWalking environment
try:
    env = gym.make('CliffWalking-v0', render_mode=None)
    logger.info("CliffWalking environment initialized successfully")
except Exception as e:
    logger.error(f"Error initializing CliffWalking environment: {e}", exc_info=True)
    env = None

# Helper function to render the CliffWalking state
def render_cliff_walking_state(state, action=None, reward=None, terminated=False, truncated=False, step_count=0):
    """
    Render the CliffWalking state as a grid.
    
    Args:
        state: Current state (position) in the CliffWalking environment
        action: Last action taken
        reward: Last reward received
        terminated: Whether the episode has terminated
        truncated: Whether the episode has been truncated
        step_count: Current step count in the episode
        
    Returns:
        Dict containing the rendered state
    """
    try:
        if env is None:
            raise ValueError("CliffWalking environment not initialized")
            
        # CliffWalking is a 4x12 grid
        height, width = 4, 12
        
        # Convert state to row, col
        row = state // width
        col = state % width
        
        # Create grid
        grid = []
        
        for i in range(height):
            row_str = ""
            for j in range(width):
                # Start position (bottom-left)
                if i == height - 1 and j == 0:
                    cell = "S"
                # Goal position (bottom-right)
                elif i == height - 1 and j == width - 1:
                    cell = "G"
                # Cliff (bottom row except for S and G)
                elif i == height - 1 and j > 0 and j < width - 1:
                    cell = "C"
                # Current position
                elif i == row and j == col:
                    cell = "X"
                # Empty space
                else:
                    cell = "."
                row_str += cell
            grid.append(row_str)
        
        # Action mapping
        action_map = {
            0: "Left",
            1: "Down",
            2: "Right",
            3: "Up",
            None: "None"
        }
        
        # Status
        status = "Playing"
        if terminated:
            if reward == -100:  # Fell off cliff
                status = "Failure (Cliff)"
            elif row == height - 1 and col == width - 1:  # Reached goal
                status = "Success"
            else:
                status = "Terminated"
        elif truncated:
            status = "Truncated"
        
        return {
            "grid": grid,
            "state": int(state),
            "position": {"row": int(row), "col": int(col)},
            "action": action_map.get(action, "None"),
            "reward": float(reward) if reward is not None else 0.0,
            "status": status,
            "terminated": terminated,
            "truncated": truncated,
            "step_count": step_count
        }
    except Exception as e:
        logger.error(f"Error rendering CliffWalking state: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error rendering state: {str(e)}")

# Factory function for CliffWalking LLM agent
def get_cliff_walking_agent(
    openrouter_client: OpenRouterClient = Depends(get_openrouter_client)
) -> LLMGameAgent[CliffWalkingAction]:
    """
    Factory function to create a CliffWalking LLM agent.
    """
    return LLMGameAgent[CliffWalkingAction](openrouter_client)

# Endpoint to get the CliffWalking game page
@router.get("", response_class=HTMLResponse)
async def get_cliff_walking_page(
    request: Request,
    hx_request: Annotated[Optional[str], Header()] = None
):
    """
    Render the CliffWalking game interface.
    """
    try:
        if env is None:
            raise HTTPException(status_code=500, detail="CliffWalking environment not initialized")
        
        # Reset the environment to get a new game
        state, _ = env.reset()
        
        # Render the initial state
        game_state = render_cliff_walking_state(state)
        
        # Return HTML response
        return templates.TemplateResponse(
            "cliff_walking.html", {"request": request, "game": game_state}
        )
    except Exception as e:
        logger.error(f"Error in get_cliff_walking_page: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred while generating CliffWalking game")

@router.get("/stream")
async def stream_cliff_walking_game(
    num_games: int = 10,
    test: bool = False,
    agent: LLMGameAgent[CliffWalkingAction] = Depends(get_cliff_walking_agent),
):
    """
    Stream a series of CliffWalking games with Server-Sent Events (SSE)
    
    Args:
        num_games: Number of games to play (default: 10)
        test: If True, use random actions instead of LLM (default: False)
    """
    return StreamingResponse(
        cliff_walking_stream_generator(num_games, test, agent),
        media_type="text/event-stream"
    )

async def cliff_walking_stream_generator(
    num_games: int = 10, 
    test: bool = False,
    agent: LLMGameAgent[CliffWalkingAction] = None
) -> AsyncGenerator[str, None]:
    """
    Generate a stream of CliffWalking game states
    
    Args:
        num_games: Number of games to play
        test: If True, use random actions instead of LLM
        agent: The LLM agent to use (if test is False)
    """
    try:
        if env is None:
            yield f"event: error\ndata: {json.dumps({'error': 'CliffWalking environment not initialized'})}\n\n"
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
            state, _ = env.reset()
            
            # Render the initial state
            game_state = render_cliff_walking_state(state)
            game_state["game_number"] = game_num + 1
            game_state["total_games"] = num_games
            
            # Send the initial state
            yield f"event: game_state\ndata: {json.dumps(game_state)}\n\n"
            await asyncio.sleep(0.5)  # Short delay for visualization
            
            # Play until the game is over
            terminated = False
            truncated = False
            step_count = 0
            total_reward = 0.0
            
            while not terminated and not truncated and step_count < 100:  # Limit steps to avoid infinite loops
                # Choose an action
                if test:
                    # Choose a random action
                    action = env.action_space.sample()
                    game_state["llm_reasoning"] = "Using random action (test mode)"
                else:
                    try:
                        # Get action from LLM
                        llm_action = await agent.get_action(
                            action_model=CliffWalkingAction,
                            game_state=game_state,
                            game_name="CliffWalking",
                            game_rules=CLIFF_WALKING_RULES
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
                next_state, reward, terminated, truncated, info = env.step(action)
                step_count += 1
                total_reward += reward
                
                # Render the new state
                game_state = render_cliff_walking_state(
                    next_state, action, reward, terminated, truncated, step_count
                )
                game_state["game_number"] = game_num + 1
                game_state["total_games"] = num_games
                
                # Send the updated state
                yield f"event: game_state\ndata: {json.dumps(game_state)}\n\n"
                await asyncio.sleep(0.5)  # Short delay for visualization
                
                # Update state for next iteration
                state = next_state
            
            # Update statistics
            stats["games_played"] += 1
            if terminated and state == (env.unwrapped.nrow - 1) * env.unwrapped.ncol + env.unwrapped.ncol - 1:  # Reached goal
                stats["successes"] += 1
            else:
                stats["failures"] += 1
            stats["total_reward"] += total_reward
            
            # Send the statistics
            yield f"event: game_over\ndata: {json.dumps({'stats': stats})}\n\n"
            
            # Short delay between games
            await asyncio.sleep(1.0)
        
        # Send a completion message
        yield f"event: stream_end\ndata: {json.dumps({'completed': True, 'stats': stats})}\n\n"
        
    except Exception as e:
        logger.error(f"Error in cliff_walking_stream_generator: {str(e)}", exc_info=True)
        yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
    finally:
        try:
            # No need to close the environment as it's reused
            pass
        except Exception as e:
            logger.error(f"Error closing CliffWalking environment: {e}", exc_info=True)

@router.post("/play_with_llm")
async def play_with_llm(
    agent: Annotated[LLMGameAgent[CliffWalkingAction], Depends(get_cliff_walking_agent)]
):
    """
    Play a CliffWalking game using the LLM agent
    """
    try:
        if env is None:
            raise HTTPException(status_code=500, detail="CliffWalking environment not initialized")
        
        # Reset the environment
        state, _ = env.reset()
        
        # Render the initial state
        game_state = render_cliff_walking_state(state)
        
        # Play until the game is over
        terminated = False
        truncated = False
        step_count = 0
        total_reward = 0.0
        actions_taken = []
        
        while not terminated and not truncated and step_count < 100:  # Limit steps to avoid infinite loops
            # Get action from LLM
            llm_action = await agent.get_action(
                action_model=CliffWalkingAction,
                game_state=game_state,
                game_name="CliffWalking",
                game_rules=CLIFF_WALKING_RULES
            )
            
            # Convert LLM action to environment action
            action = llm_action.to_env_action()
            
            # Execute the action
            next_state, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            total_reward += reward
            
            # Record the action and reasoning
            actions_taken.append({
                "step": step_count,
                "action": ["Left", "Down", "Right", "Up"][action],
                "reasoning": llm_action.reasoning,
                "reward": reward
            })
            
            # Update state for next iteration
            state = next_state
            game_state = render_cliff_walking_state(state, action, reward, terminated, truncated, step_count)
        
        # Determine the outcome
        goal_state = (env.unwrapped.nrow - 1) * env.unwrapped.ncol + env.unwrapped.ncol - 1
        outcome = "Success" if terminated and state == goal_state else "Failure"
        
        return {
            "outcome": outcome,
            "total_reward": total_reward,
            "steps": step_count,
            "actions": actions_taken,
            "final_state": game_state
        }
        
    except Exception as e:
        logger.error(f"Error in play_with_llm: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error playing CliffWalking with LLM: {str(e)}")
