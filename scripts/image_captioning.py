import os
import json
from openai import OpenAI
import base64

api_key = ""
MODE = "gpt-4o" # Tag caption using gpt-4o

images_dir = "PATH_TO_IMAGES"  # path to image directory
output_file = f"PATH/gpt4o-caption.json"

question = ""

class OpenAIAPI:
    def __init__(self, api_key):
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key)
    
    def encode_image(self, image_path):
        with open(image_path, "rb") as f:
            encoded_string = base64.b64encode(f.read()).decode('utf-8')
        return encoded_string

    def get_response(self, image_path, question):
        base64_image = self.encode_image(image_path)
        response = self.client.chat.completions.create(
            model=MODE,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        },
                        {
                            "type": "text",
                            "text": f"Provide a detailed caption for each image. Do not guess the words and expressions."
                        }
                    ]
                }
            ],
            max_completion_tokens=300, # MODE gpt-4o
        )
        return response.choices[0].message.content

if __name__ == "__main__":
    # Load the images
    images = sorted([
        os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith('.png')
    ], key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

    api_client = OpenAIAPI(api_key)
    with open(output_file, "w") as f:
        json.dump([], f)
    output_data = []

    for image_path in images:
        with open(image_path, 'rb') as image_file:
            image_data = image_file.read()

        response = api_client.get_response(image_path, question)
        output_data.append({
            "image": os.path.basename(image_path),
            "answer": response
        })

        print(f"[{os.path.basename(image_path)}]Answer: {response}")

        with open(output_file, 'w') as outfile:
            json.dump(output_data, outfile, indent=2)