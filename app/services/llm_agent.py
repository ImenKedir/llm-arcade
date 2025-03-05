from typing import Any, Dict, Generic, List, Optional, Type, TypeVar, Union
import logging
from pydantic import BaseModel

from app.services.openrouter_client import OpenRouterClient

# Set up logger
logger = logging.getLogger(__name__)

# Type variable for the action model
T = TypeVar('T', bound=BaseModel)

class LLMGameAgent(Generic[T]):
    """
    A service that uses LLMs to play games by generating actions based on game state.
    """
    
    def __init__(self, openrouter_client: OpenRouterClient):
        """
        Initialize the LLM game agent.
        
        Args:
            openrouter_client: An instance of the OpenRouterClient
        """
        self.client = openrouter_client
    
    async def get_action(
        self, 
        action_model: Type[T], 
        game_state: Dict[str, Any],
        game_name: str,
        game_rules: str,
        model: str = "openai/gpt-4o",
        temperature: float = 0.7
    ) -> T:
        """
        Get an action from the LLM based on the current game state.
        
        Args:
            action_model: The Pydantic model class for the action
            game_state: The current state of the game
            game_name: The name of the game
            game_rules: A string describing the rules of the game
            model: The model to use (default: openai/gpt-4o)
            temperature: Controls randomness (default: 0.7)
            
        Returns:
            An instance of the provided action_model
        """
        try:
            # Create a system prompt that explains the game and the task
            system_prompt = f"""
            You are an expert AI agent playing the game {game_name}.
            
            Game Rules:
            {game_rules}
            
            Your task is to choose the best action based on the current game state.
            Analyze the game state carefully and choose an action that maximizes your chances of winning.
            Provide clear reasoning for your choice.
            """
            
            # Create a prompt that describes the current game state
            prompt = f"""
            Current Game State:
            {game_state}
            
            Based on this game state, what action would you take?
            Choose the best action and explain your reasoning.
            """
            
            # Get the action from the LLM
            action = await self.client.generate_structured_output(
                response_model=action_model,
                prompt=prompt,
                model=model,
                temperature=temperature,
                system_prompt=system_prompt
            )
            
            logger.info(f"Generated action for {game_name}: {action}")
            return action
            
        except Exception as e:
            logger.error(f"Error getting action for {game_name}: {e}", exc_info=True)
            raise
