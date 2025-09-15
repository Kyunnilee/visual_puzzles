from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
import torch
import json
import re
import logging
from .base_solver import Solver
from .utils import ResponseFormat

# note that transformer version should same or lower than 4.50 to avoid NoneType error
class MolmoSolver(Solver):
    def __init__(self, data_dir, **config):
        super().__init__(data_dir)
        self.solver_name = config.get("name", "molmo")
        self.config = config
        model_id = config.get("model", "allenai/Molmo-7B-D-0924")

        self.processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map='auto'
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map='auto'
        )

        # tie weights
        self.model.tie_weights()

    def run_sample(self, prompt, image_path):
        try:
            image = Image.open(image_path).convert("RGB")
            
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    inputs = self.processor.process(images=[image], text=prompt)
                    
                    # move inputs to the correct device and make a batch of size 1
                    inputs = {k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()}
                    
                    # generate output
                    output = self.model.generate_from_batch(
                        inputs,
                        GenerationConfig(
                            max_new_tokens=self.config.get("max_tokens", 1024),
                            stop_strings=["<|endoftext|>"],
                            temperature=self.config.get("temperature", 0.0),
                        ),
                        tokenizer=self.processor.tokenizer
                    )
            
            if output is None:
                return "Error", "No output generated from model"
            
            generated_tokens = output[0,inputs['input_ids'].size(1):]
            generated_text = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            print("generated_text: ", generated_text)
            
            parsed_response = self._parse_response(generated_text)
            return parsed_response
            
        except Exception as e:
            logging.error(f"Error occurred when calling LLM API: {e}")
            return "Error", f"Failed to process: {str(e)}"

    def run_sample_icl(self, prompts, image_paths):
        try:
            images = [Image.open(p).convert("RGB") for p in image_paths]
            text = " ".join(prompts)
            print("text: ", text)
            
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    inputs = self.processor.process(images=images, text=text)
                    
                    # move inputs to the correct device and make a batch of size 1
                    inputs = {k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()}
                    
                    # generate output
                    output = self.model.generate_from_batch(
                        inputs,
                        GenerationConfig(
                            max_new_tokens=self.config.get("max_tokens", 1024),
                            stop_strings=["<|endoftext|>"],
                            temperature=self.config.get("temperature", 0.0),
                        ),
                        tokenizer=self.processor.tokenizer
                    )
            
            if output is None:
                return "Error", "No output generated from model"
            
            generated_tokens = output[0,inputs['input_ids'].size(1):]
            generated_text = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            return self._parse_response(generated_text)
            
        except Exception as e:
            logging.error(f"Error occurred when calling LLM API: {e}")
            return "Error", f"Failed to process: {str(e)}"

    def run_sample_caption(self, prompt, answer_text):
        try:
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    inputs = self.processor.process(text=prompt)
                    
                    # move inputs to the correct device and make a batch of size 1
                    inputs = {k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()}
                    
                    print("inputs: ", inputs)
                    
                    # generate output
                    output = self.model.generate_from_batch(
                        inputs,
                        GenerationConfig(
                            max_new_tokens=self.config.get("max_tokens", 1024),
                            stop_strings=["<|endoftext|>"],
                            temperature=self.config.get("temperature", 0.0),
                        ),
                        tokenizer=self.processor.tokenizer
                    )

            if output is None:
                return "Error", "No output generated from model"
            
            generated_tokens = output[0,inputs['input_ids'].size(1):]
            generated_text = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            parsed_response = self._parse_response(generated_text)
            if parsed_response:
                return parsed_response
            else:
                logging.error("Received NoneType response from API.")
                
        except Exception as e:
            logging.error(f"Error occurred when calling LLM API: {e}")
            
        return "N/A", "N/A"

    def _parse_response(self, generated_text):
        try:
            match = re.search(r"\{[^}]*\"answer\"[^}]*\}", generated_text, re.DOTALL)
            if match:
                try:
                    obj = json.loads(match.group(0))
                    return obj.get("answer", ""), obj.get("reasoning", "")
                except json.JSONDecodeError:
                    pass
            answer, reasoning = "", ""
            for line in generated_text.split("\n"):
                if "answer" in line.lower() and ":" in line:
                    parts = line.split(":", 1)
                    if len(parts) > 1:
                        answer = parts[1].strip()
                elif "reasoning" in line.lower() and ":" in line:
                    parts = line.split(":", 1)
                    if len(parts) > 1:
                        reasoning = parts[1].strip()
            if not answer:
                for line in generated_text.split("\n"):
                    if line.strip():
                        answer = line.strip()
                        break
            return answer, reasoning
        except Exception as e:
            logging.error(f"Error parsing response: {e}")
            return generated_text, "Failed to parse response"
