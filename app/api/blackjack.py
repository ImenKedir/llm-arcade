from typing import Optional, Annotated, Dict, Any, AsyncGenerator
import logging
import gymnasium as gym
import json
import asyncio
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
    prefix="/blackjack",
    tags=["blackjack"],
)

# Set up templates
templates = Jinja2Templates(directory="templates")

# Define Blackjack action schema
class BlackjackAction(BaseModel):
    """Schema for Blackjack game actions"""
    action: Literal["hit", "stick"] = Field(
        ...,
        description="The action to take in the blackjack game. 'hit' to request another card, 'stick' to keep current hand."
    )
    reasoning: str = Field(
        ...,
        description="Explanation of why this action was chosen based on the current game state."
    )
    
    def to_env_action(self) -> int:
        """Convert the action string to the corresponding action ID for the Blackjack environment"""
        return 1 if self.action == "hit" else 0

# Factory function for Blackjack LLM agent
def get_blackjack_agent(
    openrouter_client: OpenRouterClient = Depends(get_openrouter_client)
) -> LLMGameAgent[BlackjackAction]:
    """
    Factory function to create a Blackjack LLM agent.
    """
    return LLMGameAgent[BlackjackAction](openrouter_client)

# Game rules for the LLM
BLACKJACK_RULES = """
Blackjack is a card game where the goal is to get a hand value as close to 21 as possible without going over.
- You are playing against the dealer
- Number cards (2-10) are worth their face value
- Face cards (Jack, Queen, King) are worth 10
- Aces can be worth 1 or 11 (called a 'usable ace' when counted as 11)
- If your hand exceeds 21, you 'bust' and lose
- The dealer must hit until their hand is at least 17
- If you get closer to 21 than the dealer without busting, you win
- If the dealer busts, you win
- If you tie with the dealer, it's a draw

Your available actions are:
1. "hit": Take another card to increase your hand value
2. "stick": Keep your current hand and end your turn

Choose "hit" if your hand is likely to benefit from another card (e.g., low total, usable ace).
Choose "stick" if your hand is strong or if hitting would risk busting.
"""

# Initialize the blackjack environment
try:
    env = gym.make("Blackjack-v1", sab=True)
    logger.info("Blackjack environment initialized successfully")
except Exception as e:
    logger.error(f"Error initializing Blackjack environment: {e}", exc_info=True)
    env = None

# Helper function to render a blackjack game state
def render_blackjack_state(player_score: int, dealer_card: int, usable_ace: bool, 
                          terminated: bool = False, reward: float = 0.0, 
                          step_count: int = 0, action: int = None) -> Dict[str, Any]:
    """
    Render a blackjack game state in a format suitable for display with retro 8-bit styling
    """
    # Determine player status
    player_status = "Playing"
    if terminated:
        if reward > 0:
            player_status = "Win"
        elif reward < 0:
            player_status = "Lose"
        else:
            player_status = "Draw"
    
    # Convert dealer card to a readable format with retro styling
    dealer_card_display = {
        1: "A", 10: "10", 11: "J", 12: "Q", 13: "K"
    }.get(dealer_card, str(dealer_card))
    
    # Convert action to readable format with retro styling
    action_display = None
    if action is not None:
        action_display = "Hit" if action == 1 else "Stick"
    
    return {
        "player_score": player_score,
        "dealer_card": dealer_card_display,
        "usable_ace": "Yes" if usable_ace else "No",
        "status": player_status,
        "terminated": terminated,
        "reward": reward,
        "step_count": step_count,
        "action": action_display
    }

# Blackjack game endpoint
@router.get("", response_class=HTMLResponse)
async def get_blackjack(
    request: Request,
    hx_request: Annotated[Optional[str], Header()] = None
):
    """
    Get a random blackjack game state
    """
    try:
        if env is None:
            raise HTTPException(status_code=500, detail="Blackjack environment not initialized")
        
        # Reset the environment to get a new game
        observation, info = env.reset()
        player_score, dealer_card, usable_ace = observation
        
        # Render the initial state
        game_state = render_blackjack_state(player_score, dealer_card, usable_ace)
        
        # Return HTML response
        return templates.TemplateResponse(
            "blackjack.html", {"request": request, "game": game_state}
        )
    except Exception as e:
        logger.error(f"Error in get_blackjack: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred while generating blackjack game")

# Stream blackjack games using SSE
@router.get("/stream")
async def stream_blackjack_game(
    num_games: int = 10,
    test: bool = False,
    agent: LLMGameAgent[BlackjackAction] = Depends(get_blackjack_agent),
):
    """
    Stream a series of blackjack games with Server-Sent Events (SSE)
    
    Args:
        num_games: Number of games to play (default: 10)
        test: If True, use random actions instead of LLM (default: False)
    """
    return StreamingResponse(
        blackjack_stream_generator(num_games, test, agent),
        media_type="text/event-stream"
    )

# Generator function for streaming blackjack game updates
async def blackjack_stream_generator(
    num_games: int = 10, 
    test: bool = False,
    agent: LLMGameAgent[BlackjackAction] = None
) -> AsyncGenerator[str, None]:
    """
    Generate a stream of blackjack game states
    
    Args:
        num_games: Number of games to play
        test: If True, use random actions instead of LLM
        agent: The LLM agent to use (if test is False)
    """
    try:
        if env is None:
            yield f"data: {json.dumps({'error': 'Blackjack environment not initialized'})}\n\n"
            return
        
        # Track game statistics
        stats = {
            "games_played": 0,
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "total_reward": 0.0
        }
        
        # Limit the number of games to a reasonable range
        num_games = max(1, min(100, num_games))
        
        # Play the specified number of games
        for game_num in range(num_games):
            # Reset the environment to get a new game
            observation, info = env.reset()
            player_score, dealer_card, usable_ace = observation
            
            # Render the initial state
            game_state = render_blackjack_state(player_score, dealer_card, usable_ace)
            game_state["game_number"] = game_num + 1
            game_state["total_games"] = num_games
            
            # Send the initial state
            yield f"data: {json.dumps(game_state)}\n\n"
            await asyncio.sleep(0.5)  # Short delay for visualization
            
            # Play until the game is over
            terminated = False
            truncated = False
            step_count = 0
            
            while not terminated and not truncated:
                # Choose an action
                if test:
                    # Choose a random action (0 = stick, 1 = hit)
                    action = env.action_space.sample()
                    game_state["llm_reasoning"] = "Using random action (test mode)"
                else:
                    try:
                        # Get action from LLM
                        llm_action = await agent.get_action(
                            action_model=BlackjackAction,
                            game_state=game_state,
                            game_name="Blackjack",
                            game_rules=BLACKJACK_RULES
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
                
                # Get the new state
                player_score, dealer_card, usable_ace = observation
                
                # Render the new state
                game_state = render_blackjack_state(
                    player_score, dealer_card, usable_ace, terminated, reward, step_count, action
                )
                game_state["game_number"] = game_num + 1
                game_state["total_games"] = num_games
                
                # Send the updated state
                yield f"data: {json.dumps(game_state)}\n\n"
                await asyncio.sleep(0.5)  # Short delay for visualization
            
            # Update statistics
            stats["games_played"] += 1
            if reward > 0:
                stats["wins"] += 1
            elif reward < 0:
                stats["losses"] += 1
            else:
                stats["draws"] += 1
            stats["total_reward"] += reward
            
            # Send the statistics
            yield f"data: {json.dumps({'stats': stats})}\n\n"
            
            # Short delay between games
            await asyncio.sleep(1.0)
        
        # Send a completion message
        yield f"data: {json.dumps({'completed': True, 'stats': stats})}\n\n"
        
    except Exception as e:
        logger.error(f"Error in blackjack_stream_generator: {str(e)}", exc_info=True)
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
