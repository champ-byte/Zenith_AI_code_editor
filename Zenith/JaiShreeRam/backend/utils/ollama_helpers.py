import requests
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class OllamaHelper:
    @staticmethod
    def list_models(base_url: str = "http://127.0.0.1:11435") -> List[Dict]:
        """List available Ollama models"""
        try:
            response = requests.get(f"{base_url}/api/tags")
            if response.status_code == 200:
                return response.json().get("models", [])
            return []
        except Exception as e:
            logger.error(f"Error listing models: {str(e)}")
            return []

    @staticmethod
    def pull_model(model_name: str, base_url: str = "http://127.0.0.1:11435") -> bool:
        """Pull a model from Ollama"""
        try:
            response = requests.post(
                f"{base_url}/api/pull", json={"name": model_name}, stream=True
            )

            if response.status_code == 200:
                logger.info(f"Started pulling model: {model_name}")
                # You could process the stream here
                return True
            return False
        except Exception as e:
            logger.error(f"Error pulling model: {str(e)}")
            return False

    @staticmethod
    def check_model_exists(
        model_name: str, base_url: str = "http://127.0.0.1:11435"
    ) -> bool:
        """Check if a model exists locally"""
        models = OllamaHelper.list_models(base_url)
        return any(model["name"] == model_name for model in models)
