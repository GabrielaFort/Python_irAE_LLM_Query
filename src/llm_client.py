# This class contains methods for querying local and remote LLMs
import requests

class LLMClient:
    def __init__(self, model, api_url, temperature):
        self.model = model
        self.api_url = api_url
        self.temperature = temperature

    def generate(self, prompt):
        """
        Automativally handle both LM studio and Ollama APIs
        """
        if "11434" in self.api_url:
            # Ollama mode
            payload = {
                "model": self.model,
                "prompt":prompt,
                "temperature": self.temperature,
                "stream": False
            }
            response = requests.post(f"{self.api_url}/api/generate", json=payload)
            response.raise_for_status()
            content = response.json()
            if "response" in content:
                return content["response"].strip()
            else:
                raise ValueError(f"Unexpected response format from Ollama API: {content}")
            
        else:
            # LM Studio mode
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
                "max_tokens": 1024
            }
            
            response = requests.post(f"{self.api_url}/v1/chat/completions", json=payload)
            response.raise_for_status()
            content = response.json()

            return content["choices"][0]["message"]["content"]
