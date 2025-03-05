import gymnasium as gym
import numpy as np
import asyncio
import logging
import json
from fastapi import APIRouter, Request, HTTPException, Depends, Header
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
    prefix="/frozen_lake",
    tags=["frozen_lake"],
)

# Setup templates
templates = Jinja2Templates(directory="templates")

# Define action mapping
action_map = {
    0: "Left",
    1: "Down",
    2: "Right",
    3: "Up"
}

# Define cell type mapping
cell_map = {
    "S": "Start",
    "F": "Frozen",
    "H": "Hole",
    "G": "Goal"
}

# Define FrozenLake action schema
class FrozenLakeAction(BaseModel):
    """Schema for FrozenLake game actions"""
    action: Literal["left", "down", "right", "up"] = Field(
        ...,
        description="The action to take in the FrozenLake game: 'left', 'down', 'right', or 'up'."
    )
    reasoning: str = Field(
        ...,
        description="Explanation of why this action was chosen based on the current game state."
    )
    
    def to_env_action(self) -> int:
        """Convert the action string to the corresponding action ID for the FrozenLake environment"""
        action_map = {
            "left": 0,
            "down": 1,
            "right": 2,
            "up": 3
        }
        return action_map[self.action]

# Game rules for the LLM
FROZEN_LAKE_RULES = """
In the FrozenLake environment, you are an agent navigating a frozen lake to reach a goal.

The lake is represented as a grid:
- 'S' is the starting point
- 'G' is the goal
- 'F' is frozen surface (safe to walk on)
- 'H' is a hole (if you fall in, you lose)

The agent can move in four directions: left, down, right, and up.
The ice is slippery, so the agent won't always move in the intended direction.

The goal is to reach the goal tile ('G') without falling into any holes ('H').

Your available actions are:
1. "left": Try to move one cell to the left
2. "down": Try to move one cell down
3. "right": Try to move one cell to the right
4. "up": Try to move one cell up

The game ends when:
- The agent reaches the goal (success)
- The agent falls into a hole (failure)
- The maximum number of steps is reached (failure)

Choose your actions wisely to navigate safely to the goal.
"""

# Initialize the FrozenLake environment
try:
    env = gym.make("FrozenLake-v1")
    logger.info("FrozenLake environment initialized successfully")
except Exception as e:
    logger.error(f"Error initializing FrozenLake environment: {e}", exc_info=True)
    env = None

def render_frozen_lake_state(state, action=None, reward=None, terminated=False, truncated=False, step_count=0):
    """
    Render the FrozenLake state as a grid.
    
    Args:
        state: Current state (position) in the FrozenLake environment
        action: Last action taken
        reward: Last reward received
        terminated: Whether the episode has terminated
        truncated: Whether the episode has been truncated
        step_count: Current step count in the episode
        
    Returns:
        Dict containing the rendered state
    """
    try:
        # Create environment to get the map
        if env is None:
            raise ValueError("FrozenLake environment not initialized")
            
        desc = env.unwrapped.desc.astype(str)
        
        # Convert state to row, col
        row = state // env.unwrapped.ncol
        col = state % env.unwrapped.ncol
        
        # Create a copy of the map to modify
        grid = []
        for r in range(len(desc)):
            row_chars = []
            for c in range(len(desc[r])):
                cell = desc[r][c]
                # Convert bytes to string if needed
                if isinstance(cell, bytes):
                    cell = cell.decode('utf-8')
                
                # Mark the agent's position
                if r == row and c == col:
                    row_chars.append("X")
                else:
                    if cell == b'S' or cell == 'S':
                        row_chars.append("S")
                    elif cell == b'G' or cell == 'G':
                        row_chars.append("G")
                    elif cell == b'H' or cell == 'H':
                        row_chars.append("H")
                    else:  # Frozen
                        row_chars.append("F")
            grid.append(row_chars)
        
        # Determine status
        status = "In Progress"
        if terminated:
            if reward > 0:
                status = "Success"
            else:
                status = "Failure"
        elif truncated:
            status = "Truncated"
        
        return {
            "grid": grid,
            "state": int(state),
            "position": {"row": int(row), "col": int(col)},
            "action": action_map.get(action, "None"),
            "reward": float(reward) if reward is not None else 0.0,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "status": status,
            "step_count": step_count
        }
    except Exception as e:
        logger.error(f"Error rendering FrozenLake state: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error rendering state: {str(e)}")

# Factory function for FrozenLake LLM agent
def get_frozen_lake_agent(
    openrouter_client: OpenRouterClient = Depends(get_openrouter_client)
) -> LLMGameAgent[FrozenLakeAction]:
    """
    Factory function to create a FrozenLake LLM agent.
    """
    return LLMGameAgent[FrozenLakeAction](openrouter_client)

@router.get("", response_class=HTMLResponse)
async def get_frozen_lake(
    request: Request,
    hx_request: Annotated[Optional[str], Header()] = None
):
    """
    Render the FrozenLake game interface.
    """
    try:
        if env is None:
            raise HTTPException(status_code=500, detail="FrozenLake environment not initialized")
        
        # Reset the environment to get a new game
        state, _ = env.reset()
        
        # Render the initial state
        game_state = render_frozen_lake_state(state)
        
        # Return HTML response
        return templates.TemplateResponse(
            "frozen_lake.html", {"request": request, "game": game_state}
        )
    except Exception as e:
        logger.error(f"Error in get_frozen_lake: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred while generating FrozenLake game")

@router.get("/stream")
async def stream_frozen_lake_game(
    num_games: int = 10,
    test: bool = False,
    agent: LLMGameAgent[FrozenLakeAction] = Depends(get_frozen_lake_agent),
):
    """
    Stream a series of FrozenLake games with Server-Sent Events (SSE)
    
    Args:
        num_games: Number of games to play (default: 10)
        test: If True, use random actions instead of LLM (default: False)
    """
    return StreamingResponse(
        generate_frozen_lake_stream(num_games, test, agent),
        media_type="text/event-stream"
    )

async def generate_frozen_lake_stream(
    num_games: int = 10, 
    test: bool = False,
    agent: LLMGameAgent[FrozenLakeAction] = None
) -> AsyncGenerator[str, None]:
    """
    Generate a stream of FrozenLake game states
    
    Args:
        num_games: Number of games to play
        test: If True, use random actions instead of LLM
        agent: The LLM agent to use (if test is False)
    """
    try:
        if env is None:
            yield f"event: error\ndata: {json.dumps({'error': 'FrozenLake environment not initialized'})}\n\n"
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
            game_state = render_frozen_lake_state(state)
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
                            action_model=FrozenLakeAction,
                            game_state=game_state,
                            game_name="FrozenLake",
                            game_rules=FROZEN_LAKE_RULES
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
                game_state = render_frozen_lake_state(
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
            if terminated and reward > 0:
                stats["successes"] += 1
            else:
                stats["failures"] += 1
            stats["total_reward"] += total_reward
            
            # Send the statistics
            yield f"event: game_state\ndata: {json.dumps({'stats': stats})}\n\n"
            
            # Short delay between games
            await asyncio.sleep(1.0)
        
        # Send a completion message
        yield f"event: game_over\ndata: {json.dumps(stats)}\n\n"
        yield f"event: stream_end\ndata: {json.dumps({'completed': True})}\n\n"
        
    except Exception as e:
        logger.error(f"Error in generate_frozen_lake_stream: {str(e)}", exc_info=True)
        yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

@router.post("/play_with_llm")
async def play_with_llm(
    agent: Annotated[LLMGameAgent[FrozenLakeAction], Depends(get_frozen_lake_agent)]
):
    """
    Play a FrozenLake game using the LLM agent
    """
    try:
        if env is None:
            raise HTTPException(status_code=500, detail="FrozenLake environment not initialized")
        
        # Reset the environment
        state, _ = env.reset()
        
        # Render the initial state
        game_state = render_frozen_lake_state(state)
        
        # Play until the game is over
        terminated = False
        truncated = False
        step_count = 0
        total_reward = 0.0
        actions_taken = []
        
        while not terminated and not truncated and step_count < 100:  # Limit steps to avoid infinite loops
            # Get action from LLM
            llm_action = await agent.get_action(
                action_model=FrozenLakeAction,
                game_state=game_state,
                game_name="FrozenLake",
                game_rules=FROZEN_LAKE_RULES
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
                "action": action_map[action],
                "reasoning": llm_action.reasoning,
                "reward": reward
            })
            
            # Update state for next iteration
            state = next_state
            game_state = render_frozen_lake_state(state, action, reward, terminated, truncated, step_count)
        
        # Determine the outcome
        outcome = "Success" if terminated and reward > 0 else "Failure"
        
        return {
            "outcome": outcome,
            "total_reward": total_reward,
            "steps": step_count,
            "actions": actions_taken,
            "final_state": game_state
        }
        
    except Exception as e:
        logger.error(f"Error in play_with_llm: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error playing FrozenLake with LLM: {str(e)}")
