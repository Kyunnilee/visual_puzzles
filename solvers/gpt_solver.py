from openai import OpenAI
from .base_solver import Solver, encode_image
from .utils import ResponseFormat
import time
import json
import logging


class GPTSolver(Solver):
    def __init__(self, data_dir, **config):
        super().__init__(data_dir)
        self.solver_name = config.get("name", "gpt4o")
        self.config = config
        self.client = OpenAI(
            api_key=config.get("api_key"),
            organization=config.get("organization", None),
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
                if "4o" in self.solver_name:
                    completion = self.client.beta.chat.completions.parse(
                        model=self.config.get("model", "gpt4o"),
                        messages=[{"role": "user", "content": content}],
                        temperature=self.config.get("temperature", 0.0),
                        max_tokens=self.config.get("max_tokens", 1024),
                        response_format=ResponseFormat
                    )
                    response = completion.choices[0].message.parsed
                else:
                    completion = self.client.beta.chat.completions.parse(
                        model=self.config.get("model", "gpt-4o"),
                        messages=[{"role": "user", "content": content}],
                        max_completion_tokens=self.config.get("max_tokens", 1024),
                        response_format=ResponseFormat
                    )
                    response = completion.choices[0].message.parsed
                break
            except Exception as e:
                logging.error(f"Error occured when calling LLM API: {e}")
                time.sleep(60)

        return response.answer, response.reasoning
    
    def run_sample_icl(self, prompts, image_paths):
        content = []
        for prompt, image_path in zip(prompts, image_paths):
            content += [{"type": "text", "text": prompt}]

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
                if "4o" in self.solver_name:
                    completion = self.client.beta.chat.completions.parse(
                        model=self.config.get("model", "gpt4o"),
                        messages=[{"role": "user", "content": content}],
                        temperature=self.config.get("temperature", 0.0),
                        max_tokens=self.config.get("max_tokens", 1024),
                        response_format=ResponseFormat
                    )
                    response = completion.choices[0].message.parsed
                else:
                    completion = self.client.beta.chat.completions.parse(
                        model=self.config.get("model", "gpt4o"),
                        messages=[{"role": "user", "content": content}],
                        max_completion_tokens=self.config.get("max_tokens", 1024),
                        response_format=ResponseFormat
                    )
                    response = completion.choices[0].message.parsed
                break
            except Exception as e:
                logging.error(f"Error occured when calling LLM API: {e}")
                # raise e
                time.sleep(60)

        return response.answer, response.reasoning

    def run_sample_caption(self, prompt, answer_text):
        content = [{"type": "text", "text": prompt}]
        for _ in range(5):
            try:
                if "4o" in self.solver_name:
                    completion = self.client.beta.chat.completions.parse(
                        model=self.config.get("model", "gpt-4o"),
                        messages=[{"role": "user", "content": content}],
                        temperature=self.config.get("temperature", 0.0),
                        max_tokens=self.config.get("max_tokens", 1024),
                        response_format=ResponseFormat
                    )
                    response = completion.choices[0].message.parsed
                else:
                    completion = self.client.beta.chat.completions.parse(
                        model=self.config.get("model", "gpt4o"),
                        messages=content,
                        response_format=ResponseFormat
                    )
                    response = completion.choices[0].message.parsed
                if response:
                    return response.answer, response.reasoning
                else:
                    logging.error("Received NoneType response from API.")
            except Exception as e:
                logging.error(f"Error occurred when calling LLM API: {e}")
                time.sleep(60)
        return "N/A", "N/A"