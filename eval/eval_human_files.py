import json
import time
import tqdm
import random
import openai
import functools
from pathlib import Path
from typing import List, Dict

# --- Configuration ---
RESULTS_FILE = Path("/Users/davidchan/Projects/visual-puzzles/output.json")  # Replace with your JSON file path
MODEL_NAME = RESULTS_FILE.stem
LLM_AS_JUDGE = True  # Set to True to use OpenAI as judge
EVAL_LOG_PATH = Path(f"./eval_log_{MODEL_NAME}.json")
HANDLE_SKIP = True  # Set to True to skip empty predictions

client = openai.Client(
    api_key="PUT_YOUR_API_KEY"
)


def _check_correctness(results: List[Dict], use_llm=False) -> Dict:
    if use_llm:
        return evaluate_openai(results)
    else:
        return evaluate_matching(results)


def load_results_from_file(file_path: Path) -> List[Dict]:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def log_append(logs: List[Dict], image: str, gt: str, pred: str, is_correct: bool):
    logs.append({"image": image, "ground_truth": gt, "prediction": pred, "correct": is_correct})


def evaluate_matching(results: List[Dict]) -> Dict:
    total = 0
    correct = 0
    seen_images = set()

    logs = []

    for result in results:
        image = result.get("image", "Unknown Image")
        if image in seen_images:
            continue
        seen_images.add(image)
        gt = result["answer"].strip().lower()
        pred = result["result"]["pred"].strip().lower()

        if HANDLE_SKIP:
            if pred == "<skip>" or pred == "":
                continue

        possible_answers = [ans.strip().lower().replace(" ", "") for ans in gt.split("/")]
        is_correct = pred and any(pred.replace(" ", "") in p for p in possible_answers)
        if is_correct:
            correct += 1
        # else:
        #     print(f"Incorrect for {image}: GT: {gt}, Pred: {pred}")

        total += 1

        log_append(logs, image, gt, pred, is_correct)

    with open(EVAL_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2)

    accuracy = correct / total if total > 0 else 0
    return {"total": total, "correct": correct, "accuracy": accuracy}


def evaluate_openai(results: List[Dict]) -> Dict:
    total = 0
    correct = 0
    seen_images = set()
    logs = []

    for result in tqdm.tqdm(results):
        image = result.get("image", "Unknown Image")
        if image in seen_images:
            continue
        seen_images.add(image)

        gt = result["answer"]
        pred = result["result"]["pred"]
        reasoning = result["result"].get("reasoning", "")

        if HANDLE_SKIP:
            if pred == "<skip>" or pred == "" or pred == "<SKIP>":
                continue

        elif pred == "" or pred == "<skip>":
            total += 1
            log_append(logs, image, gt, pred, False)
            continue

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
            response = _openai_call(prompt)
            reply = response.choices[0].message.content.strip().lower()
            is_correct = "yes" in reply
            if is_correct:
                correct += 1
            total += 1

            log_append(logs, image, gt, pred, is_correct)
        except Exception as e:
            print(f"Error evaluating image {image}: {e}")
        # time.sleep(1.2)

    with open(EVAL_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2)

    accuracy = correct / total if total > 0 else 0
    return {"total": total, "correct": correct, "accuracy": accuracy}


@functools.lru_cache()
def _openai_call(prompt):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an evaluator for a puzzle-solving task."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=1,
    )

    return response


def bootstrap_evaluation(results: List[Dict], num_samples: int = 1000, use_llm=False) -> Dict:
    accuracies = []
    for _ in range(num_samples):
        sample = random.choices(results, k=len(results))
        if use_llm:
            evaluation = evaluate_openai(sample)
        else:
            evaluation = evaluate_matching(sample)
        accuracies.append(evaluation["accuracy"])

    lower_bound = max(0, sorted(accuracies)[int(0.025 * num_samples)])
    upper_bound = min(1, sorted(accuracies)[int(0.975 * num_samples)])
    mean_accuracy = sum(accuracies) / num_samples

    return {"mean_accuracy": mean_accuracy, "confidence_interval": (lower_bound, upper_bound)}


def main():
    results = load_results_from_file(RESULTS_FILE)
    evaluation = _check_correctness(results, use_llm=LLM_AS_JUDGE)

    print(f"Total: {evaluation['total']}")
    print(f"Correct: {evaluation['correct']}")
    print(f"Accuracy: {evaluation['accuracy']:.2%}")

    bootstrap_results = bootstrap_evaluation(results, use_llm=LLM_AS_JUDGE)
    print(f"Mean Accuracy: {bootstrap_results['mean_accuracy']:.2%}")
    print(f"95% Confidence Interval: {bootstrap_results['confidence_interval']}")


if __name__ == "__main__":
    main()