"""
Reference for JSON decoding: https://github.com/anthropics/anthropic-cookbook/blob/main/misc/how_to_enable_json_mode.ipynb
"""

import anthropic
from .base_solver import Solver, encode_image
import time
import json


def parse_result(response):
    json_start = response.index("{")
    json_end = response.rfind("}")
    json_obj = json.loads(response[json_start:json_end + 1])
    return json_obj["answer"], json_obj["reasoning"]


class ClaudeSolver(Solver):
    def __init__(self, data_dir, **config):
        super().__init__(data_dir)
        self.solver_name = config.get("name", "Claude-3")
        self.config = config
        self.client = anthropic.Anthropic(api_key=config.get("api_key"))

    def run_sample(self, prompt, image_path):
        content = []
        base64_image = encode_image(image_path)
        content += [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "data": base64_image,
                    "media_type": "image/png",
                },
            }
        ]
        content += [{"type": "text", "text": prompt}]
        answer, reasoning = '', ''

        pred_ans = ""
        for _ in range(5):
            try:
                message = self.client.messages.create(
                    model=self.config.get("model", "gpt-4o"),
                    temperature=self.config.get("temperature", 0.0),
                    max_tokens=self.config.get("max_tokens", 1024),
                    messages=[
                        {
                            "role": "user",
                            "content": content,
                        }
                    ],
                )
                pred_ans = message.content[-1].text
                answer, reasoning = parse_result(pred_ans)
                break
            except Exception as e:
                print(f"Error occured when calling LLM API: {e}")
                time.sleep(60)

        return answer, reasoning