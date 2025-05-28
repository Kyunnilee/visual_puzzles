from openai import OpenAI
from .base_solver import Solver, encode_image
from .utils import ResponseFormat
import time
import json
import logging


class PixtralSolver(Solver):
    def __init__(self, data_dir, **config):
        super().__init__(data_dir)
        self.solver_name = config.get("name", "mistralai/Pixtral-12B-2409")
        self.config = config
        self.client = OpenAI(
            api_key="EMPTY", # assume VLLM or SGLang is used,
            base_url=f"http://localhost:8000/v1" # TODO add a config for this
        )

    def run_sample(self, prompt, image_path):
        content = [{"type": "text", "text": prompt}]

        base64_image = encode_image(image_path)
        if self.config.get("low_res_mode", False):
            content += [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "low",
                    },
                }
            ]
        else:
            content += [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                }
            ]
        for _ in range(5):
            try:
                # Support after "gpt-4o-2024-08-06"
                completion = self.client.beta.chat.completions.parse(
                    model=self.config.get("model", "mistralai/Pixtral-12B-2409"),
                    messages=[{"role": "user", "content": content}],
                    temperature=self.config.get("temperature", 0.0),
                    max_tokens=self.config.get("max_tokens", 1024),
                    response_format=ResponseFormat,
                )
                response = completion.choices[0].message.parsed
                break
            except Exception as e:
                logging.error(f"Error occured when calling LLM API: {e}")
                time.sleep(60)

        return response.answer, response.reasoning
