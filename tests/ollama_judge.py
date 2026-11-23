import json
import urllib.request
from deepeval.models.base_model import DeepEvalBaseLLM


class OllamaJudge(DeepEvalBaseLLM):
    def __init__(self, model_name="llama3.1"):
        self.model_name = model_name
        self.url = "http://localhost:11434/api/generate"

    def load_model(self):
        return self

    def _call_ollama(self, prompt: str) -> str:
        is_json_requested = "json" in prompt.lower()

        payload_dict = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0},
        }

        if is_json_requested:
            payload_dict["format"] = "json"

        payload = json.dumps(payload_dict).encode("utf-8")

        req = urllib.request.Request(
            self.url,
            data=payload,
            headers={"Content-Type": "application/json"},
        )

        with urllib.request.urlopen(req) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            return res_data.get("response", "")

    def generate(self, prompt: str) -> str:
        return self._call_ollama(prompt)

    async def a_generate(self, prompt: str) -> str:
        return self._call_ollama(prompt)

    def get_model_name(self):
        return f"Ollama-{self.model_name}"