from PIL import Image
from google import genai
import time
import logging
import os
from .base_solver import Solver
from .utils import ResponseFormat

from google.genai import types


class GeminiSolver(Solver):
    def __init__(self, data_dir, **config):
        super().__init__(data_dir)
        self.solver_name = config.get("name", "gemini-2.0-flash")
        self.config = config
        self.client = genai.Client(
            api_key=os.getenv("GOOGLE_API_KEY"),
            # http_options=types.HttpOptions(api_version='v1alpha')
        )
    
        # TODO: use vLLM or SGLang to speed up the inference...

    def run_sample(self, prompt:str, image_path:str):
        image = Image.open(image_path)
        content = [image, prompt]
        
        for _ in range(5):
            try:
                # Support after "gpt-4o-2024-08-06"
                completion = self.client.models.generate_content(
                    model=self.config.get("model", "gemini-2.0-flash"),
                    contents=content,
                    config=types.GenerateContentConfig(
                        max_output_tokens=self.config.get("max_tokens", 1024),
                        temperature=self.config.get("temperature", 0.0),
                        response_mime_type='application/json',
                        response_schema=ResponseFormat,
                    ),
                )
                response = completion.parsed
                answer = response.answer
                reasoning = response.reasoning
                if not answer:
                    logging.error(f"Empty answer from Gemini API. {completion} Retrying...")
                break
            except Exception as e:
                logging.error(f"Error occured when calling LLM API: {e}")
                time.sleep(60)
        return answer, reasoning
