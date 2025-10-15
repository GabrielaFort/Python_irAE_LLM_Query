# This class contains methods for querying local and remote LLMs
import requests

class LLMClient:
    def __init__(self, model, api_url, temperature):
        self.model = model
        self.api_url = api_url
        self.temperature = temperature

    def generate(self, prompt):
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
