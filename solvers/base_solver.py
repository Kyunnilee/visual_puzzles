from tqdm import tqdm
import random
import base64
import json
import os
from .utils import PROMPT
from .utils import ICL_PROMPT, ICL_PATH
from .utils import CAPTION_PROMPT
from .utils import SKILLS_ID_TO_NAME, skill_informed_prompt
from PIL import Image

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def llama_image(image_path):
    image = Image.open(image_path)
    return image

class Solver:
    def __init__(self, data_dir):
        self.solver_name = "Base Solver"
        annotation_file = os.path.join(data_dir, "answers.json")
        self.data_dir = data_dir
        with open(annotation_file, "r") as f:
            self.data = json.load(f)
        

    def preprocessing(self):
        # Any preprocessing steps can be added here
        pass

    def postprocessing(self):
        # Any postprocessing steps can be added here
        pass

    def run_sample(self, prompt, image_path, **kwargs):
        # This should be implemented with the actual generation logic
        raise NotImplementedError
    
    def run_sample_icl(self, prompts, image_paths, **kwargs):
        # This should be implemented with the actual generation logic
        pass

    def run_sample_caption(self, prompt, answer_text, **kwargs):
        # This should be implemented with the actual generation logic
        pass

    def run_batch(self, output_dir, start_idx=None, end_idx=None, use_icl=False, use_skill_prompt=False, use_caption=False, caption_answers=None):
        # If use ICL, add the icl prompt before the PROMPT
        self.preprocessing()
        if start_idx is None:
            start_idx = 0
        if end_idx is None:
            end_idx = len(self.data)
        print(start_idx)
        test_data = self.data[start_idx:end_idx]
        os.makedirs(output_dir, exist_ok=True)

        for idx, entry in tqdm(enumerate(test_data), total=len(test_data)):
            data_id = idx + start_idx

            output_fname = os.path.join(output_dir, f"{data_id:03d}.json")

            # shuffle input image orderings
            image_path = os.path.join(self.data_dir, "image", entry["image"])
            caption_path = os.path.join(self.data_dir, "gpt4o-caption.json")
            with open(caption_path, "r") as f:
                captions = json.load(f)
            if use_icl:
                prompts = [ICL_PROMPT, PROMPT]
                icl_image_path = os.path.join(self.data_dir, "image", ICL_PATH)
                image_paths = [icl_image_path, image_path]
                pred, reasoning = self.run_sample_icl(prompts, image_paths)
                entry["result"] = {
                    "prompt": prompts,
                    "pred": pred,
                    "reasoning": reasoning,
                }
            elif use_skill_prompt:
                prompt = skill_informed_prompt(skills=entry["skills"], prompt=PROMPT)
                pred, reasoning = self.run_sample(prompt, image_path)
                entry["result"] = {
                    "prompt": prompt,
                    "pred": pred,
                    "reasoning": reasoning,
                }
            elif use_caption:
                answer_text = captions[data_id]["answer"]
                prompt = CAPTION_PROMPT.format(answer_text=answer_text)
                pred, reasoning = self.run_sample_caption(prompt, caption_path)
                entry["result"] = {
                    "prompt": prompt,
                    "pred": pred,
                    "reasoning": reasoning,
                }
            else:
                prompt = PROMPT
                pred, reasoning = self.run_sample(prompt, image_path)
                entry["result"] = {
                    "prompt": prompt,
                    "pred": pred,
                    "reasoning": reasoning,
                }
            with open(output_fname, "w") as f:
                json.dump(entry, f, indent=2)
        self.postprocessing()