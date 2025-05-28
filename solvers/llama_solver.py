from openai import OpenAI
from .base_solver import Solver, encode_image, llama_image
from .utils import ResponseFormat
import time
import json
import logging
from transformers import AutoProcessor, MllamaForConditionalGeneration
import torch

class LlamaSolver(Solver):
    def __init__(self, data_dir, **config):
        super().__init__(data_dir)
        self.solver_name = config.get("name", "meta-llama/Llama-3.2-11B-Vision")
        self.config = config
        self.model = MllamaForConditionalGeneration.from_pretrained(
            self.solver_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(self.solver_name)

    def run_sample(self, prompt, image_path):
       
        base64_image = llama_image(image_path)
        
        text = f"<|image|><|begin_of_text|>{prompt}"
        
        inputs = self.processor(base64_image, text, return_tensors="pt").to(self.model.device)

        
        for _ in range(5):
            try:
                # Generate response
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                )

                response = self.processor.batch_decode(outputs[:, inputs["input_ids"].shape[-1]:])[0]
                break
            except Exception as e:
                logging.error(f"Error occured when using Llama Generation: {e}")
                time.sleep(60)

        return response, None