from typing import Any, Dict, Generic, List, Optional, Type, TypeVar
import logging
import os
import json
from pydantic import BaseModel, create_model
from fastapi import Depends, HTTPException
import httpx

# Set up logger
logger = logging.getLogger(__name__)

# Type variable for the response model
T = TypeVar('T', bound=BaseModel)

class OpenRouterClient(Generic[T]):
    """
    A client for interacting with the OpenRouter API with structured output support.
    This client is designed to be used with dependency injection in FastAPI routes.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the OpenRouter client.
        
        Args:
            api_key: OpenRouter API key. If not provided, will try to get from environment variable.
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            logger.warning("OpenRouter API key not provided and not found in environment variables")
        
        self.base_url = "https://openrouter.ai/api/v1"
        self.client = httpx.AsyncClient(timeout=60.0)  # Longer timeout for LLM responses
    
    async def close(self):
        """Close the HTTP client session."""
        await self.client.aclose()
    
    async def generate_structured_output(
        self, 
        response_model: Type[T], 
        prompt: str,
        model: str = "openai/gpt-4o",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        system_prompt: Optional[str] = None
    ) -> T:
        """
        Generate a structured output from the OpenRouter API based on a Pydantic model.
        
        Args:
            response_model: A Pydantic model class that defines the expected structure
            prompt: The prompt to send to the model
            model: The model to use (default: openai/gpt-4o)
            temperature: Controls randomness (default: 0.7)
            max_tokens: Maximum tokens to generate (default: 1024)
            system_prompt: Optional system prompt to guide the model
            
        Returns:
            An instance of the provided response_model
        """
        try:
            # Create messages array
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            # Convert Pydantic model to JSON schema
            schema = response_model.model_json_schema()
            
            # Prepare the request payload
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": response_model.__name__,
                        "strict": True,
                        "schema": schema
                    }
                }
            }
            
            # Make the API request
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            response = await self.client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload
            )
            
            # Handle API errors
            response.raise_for_status()
            
            # Parse the response
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            # If content is a string, try to parse it as JSON
            if isinstance(content, str):
                try:
                    content_data = json.loads(content)
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse response as JSON: {content}")
                    raise HTTPException(status_code=500, detail="Invalid response format from LLM")
            else:
                content_data = content
                
            # Create and validate the model instance
            return response_model.model_validate(content_data)
            
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error during OpenRouter API call: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"OpenRouter API error: {str(e)}")
        except Exception as e:
            logger.error(f"Error generating structured output: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error generating structured output: {str(e)}")

# Factory function for dependency injection
def get_openrouter_client() -> OpenRouterClient:
    """
    Factory function to create an OpenRouterClient instance for dependency injection.
    """
    client = OpenRouterClient()
    return client
