import json
import re
import time
from openai import OpenAI

# Load the predictions from your current file
with open("./eval/compat.json", "r") as f:
    predictions = json.load(f)

# Sort predictions by numeric value in image filename
predictions.sort(key=lambda x: int(re.findall(r'\d+', x["image"])[0]))

# Load the ground truths and image names from the original answers file
with open("./dataset/answers.json", "r") as f:
    ground_truth_data = json.load(f)

# Create a quick lookup dictionary for ground truth answers by image
ground_truth_dict = {entry["image"]: entry["answer"] for entry in ground_truth_data}

def normalize_text(text):
    """Lowercase and remove spaces for naive matching."""
    return re.sub(r'\s+', '', text.lower())

# Create the final result format
results = []
for entry in predictions:
    image = entry["image"]
    prediction_raw = entry["answer"]
    ground_truth = ground_truth_dict.get(image, "")
    correct = prediction_raw == ground_truth
    results.append({
        "image": image,
        "ground_truth": ground_truth,
        "prediction": prediction_raw,
        "correct": correct
    })

# Initialize OpenAI API client
client = OpenAI(api_key="")  # Replace this with your actual API key

# Save naive matching results
with open("patrick_result_naive.json", "w") as f:
    json.dump(results, f, indent=2)


# Initialize or clear the GPT-4o results file properly
with open("patrick_result_llm.json", "w") as f:
    f.write("[")

first_entry = True  # Track whether we're writing the first JSON object

gpt4o_results = []

for entry in predictions:
    image = entry["image"]
    gt = ground_truth_dict.get(image, "")
    print(gt)
    pred = entry["answer"]
    prompt = (
        "Determine whether the predicted answer is semantically equivalent to the ground truth. "
        "Ignore differences in case and spacing, as well as minor errors in spelling. "
        "If there's a / in the ground truth, it means that either answer is acceptable.\n"
        f"Ground Truth: {gt}\n"
        f"Prediction: {pred}\n"
        "Respond with only 'yes' or 'no'."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an evaluator for a puzzle-solving task."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=1,
        )
        answer = response.choices[0].message.content.strip().lower()
        correct = answer == "yes"
    except Exception as e:
        print(f"Error on image {image}: {e}")
        correct = False

    result_entry = {
        "image": image,
        "ground_truth": gt,
        "prediction": pred,
        "correct": correct
    }
    gpt4o_results.append(result_entry)

    # Properly append JSON objects with commas between them
    with open("NAME_result_llm.json", "a") as f:
        if not first_entry:
            f.write(",\n")
        else:
            first_entry = False
        json.dump(result_entry, f, indent=2)

    time.sleep(1)  # To avoid hitting API rate limits

# Finalize the JSON array correctly
with open("NAME.json", "a") as f:
    f.write("\n]\n")
