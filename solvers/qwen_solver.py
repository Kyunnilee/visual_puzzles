from openai import OpenAI
from .base_solver import Solver, encode_image
from .utils import ResponseFormat
import time
import json
import logging
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import torch
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

class QwenSolver(Solver):
    def __init__(self, data_dir, **config):
        super().__init__(data_dir)
        self.solver_name = config.get("name", "Qwen/Qwen2.5-VL-7B-Instruct")
        self.config = config
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.solver_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(self.solver_name)

    def run_sample(self, prompt, image_path):
       
        base64_image = encode_image(image_path)
        content  = {
            "role": "user",
            "content": [
                {"type": "image", "image": f"data:image;base64,{base64_image}"},
                {"type": "text", "text": prompt}
            ]
        }
        conversations = [content]
        text = self.processor.apply_chat_template(
            conversations,
            tokenize=False, 
            add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(conversations)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        
        for _ in range(5):
            try:
                # Generate response
                generated_ids = self.model.generate(**inputs, max_new_tokens=256)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                response = self.processor.batch_decode(
                    generated_ids_trimmed, 
                    skip_special_tokens=True, 
                    clean_up_tokenization_spaces=False
                )[0]
                break
            except Exception as e:
                logging.error(f"Error occured when using Qwen Generation: {e}")
                time.sleep(60)

        return response, None

class QwenOmniSolver(Solver):
    def __init__(self, data_dir, **config):
        super().__init__(data_dir)
        self.solver_name = config.get("name", "Qwen/Qwen2.5-Omni-7B")
        self.config = config
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-Omni-7B", torch_dtype="auto", device_map="auto")
        self.processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")

    def run_sample(self, prompt, image_path):
       
        base64_image = encode_image(image_path)
        content  = {
            "role": "user",
            "content": [
                {"type": "image", "image": f"data:image;base64,{base64_image}"},
                {"type": "text", "text": prompt}
            ]
        }
        conversations = [content]
        text = self.processor.apply_chat_template(
            conversations,
            tokenize=False, 
            add_generation_prompt=True
        )

        USE_AUDIO_IN_VIDEO = True
        
        audios, images, videos = process_mm_info(conversations, use_audio_in_video=USE_AUDIO_IN_VIDEO)
        inputs = self.processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
        inputs = inputs.to(self.model.device).to(self.model.dtype)
        
        for _ in range(5):
            try:
                # Generate response
                generated_ids, _ = self.model.generate(**inputs, max_new_tokens=256)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                response = self.processor.batch_decode(
                    generated_ids_trimmed, 
                    skip_special_tokens=True, 
                    clean_up_tokenization_spaces=False
                )[0]
                break
            except Exception as e:
                logging.error(f"Error occured when using Qwen Generation: {e}")
                raise e
                time.sleep(60)

        return response, None