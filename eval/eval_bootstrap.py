import os
import json
import time
import random
import openai
import functools
from pathlib import Path
from typing import List, Dict

client = openai.Client(
    api_key="PUT_YOUR_API_KEY_HERE"
)

# --- Configuration ---
ANSWERS_PATH = Path("./result/base_prompt/MODEL_NAME/result")
MODEL_NAME = ANSWERS_PATH.parent.name
LLM_AS_JUDGE = True  # Use LLM to judge correctness

if LLM_AS_JUDGE:
    EVAL_LOG_PATH = Path(f"./eval/eval_log/{MODEL_NAME}_llm.json")
else:
    # Naive matching evaluation
    EVAL_LOG_PATH = Path(f"./eval/eval_log/{MODEL_NAME}_naive.json")


def load_results(answers_path: Path) -> List[Dict]:
    results = []
    for file in sorted(answers_path.glob("*.json")):
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
            results.append(data)
    return results


def log_append(logs: List[Dict], image: str, gt: str, pred: str, is_correct: bool):
    logs.append({"image": image, "ground_truth": gt, "prediction": pred, "correct": is_correct})


def inference_matching(results: List[Dict]) -> List[Dict]:
    """
    Evaluate the results by checking if the predicted answer matches the ground truth.
    """
    logs = []
    for idx, result in enumerate(results):
        image = result.get("image", "Unknown Image")
        gt = result.get("answer", "").strip().lower() if result.get("answer") else None
        pred = result.get("result", {}).get("pred", "").strip().lower() if result.get("result", {}).get("pred") else None

        if gt is None or pred is None:
            is_correct = False
        else:
            possible_answers = [
                ans.strip().lower() for ans in gt.split("/")
            ]  # when there are multiple answers like apple/orange
            is_correct = pred in possible_answers  # check if either matches
        log_append(logs, image, gt, pred, is_correct)

        EVAL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(EVAL_LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=2)
    return logs
    
def inference_openai(results: List[Dict]) -> List[Dict]:
    logs = []
    for idx, result in enumerate(results):
        image = result.get("image", "Unknown Image")
        gt = result.get("answer")
        pred = result.get("result", {}).get("pred")
        reasoning = result.get("result", {}).get("reasoning")

        prompt = (
            "Determine whether the predicted answer is semantically equivalent to the ground truth. "
            "Ignore differences in case and spacing, as well as minor errors in spelling. "
            "If there's a / in the ground truth, it means that either answer is acceptable.\n"
            f"Ground Truth: {gt}\n"
            f"Prediction: {pred}\n"
            f"Reasoning: {reasoning}\n"
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
            is_correct = answer == "yes"
        except Exception as e:
            print(f"Error processing entry {idx}: {e}")
            is_correct = False
        log_append(logs, image, gt, pred, is_correct)

        EVAL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(EVAL_LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=2)
        time.sleep(1)
    return logs
    
def precompute_inference(results: List[Dict], use_llm=False):
    logs = []
    if use_llm:
        logs = inference_openai(results)
    else:
        logs = inference_matching(results)

    EVAL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EVAL_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2)

def bootstrap_from_log(log_path: Path, num_samples: int = 1000) -> Dict:
    """
    Bootstrap 95% confidence interval for the accuracy from the evaluation log.
    """
    accuracies = []
    with open(log_path, "r", encoding="utf-8") as f:
        logs = json.load(f)
    for _ in range(num_samples):
        sample = random.choices(logs, k=len(logs))
        correct = sum(1 for entry in sample if entry["correct"])
        accuracy = correct / len(sample)
        accuracies.append(accuracy)

    lower_bound = max(0, sorted(accuracies)[int(0.025 * num_samples)])
    upper_bound = min(1, sorted(accuracies)[int(0.975 * num_samples)])
    mean_accuracy = sum(accuracies) / num_samples

    return {"mean_accuracy": mean_accuracy, "confidence_interval": (lower_bound, upper_bound)}

def main():
    results = load_results(ANSWERS_PATH)
    
    if not EVAL_LOG_PATH.exists():
        print("Precomputing evaluations and saving to log...")
        precompute_inference(results, use_llm=LLM_AS_JUDGE)
    else:
        overwrite = input(f"Log file {EVAL_LOG_PATH} already exists. Overwrite? (y/n): ")
        if overwrite.lower() == "y":
            print("Overwriting existing log file...")
            precompute_inference(results, use_llm=LLM_AS_JUDGE)
        else:
            print("Using existing log file.")
    bootstrap_results = bootstrap_from_log(EVAL_LOG_PATH)
    with open(EVAL_LOG_PATH, "r", encoding="utf-8") as f:
        logs = json.load(f)
    
    total = len(logs)
    correct = sum(1 for entry in logs if entry["correct"])
    accuracy = correct / total if total > 0 else 0

    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Mean Accuracy (Bootstrapped): {bootstrap_results['mean_accuracy']:.2%}")
    print(f"95% Confidence Interval: {bootstrap_results['confidence_interval']}")

if __name__ == "__main__":
    main()