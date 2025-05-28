import argparse
import json
import os
from PIL import Image
import torch
from tqdm import tqdm
import open_clip
import numpy as np
from sklearn.metrics import precision_score, recall_score
from scipy.stats import rankdata
from collections import defaultdict


def load_ground_truth(json_path):
    with open(json_path, "r") as f:
        return json.load(f)


def compute_metrics(ranks, k_values=[1, 5, 10]):
    metrics = {}
    ranks = np.array(ranks)

    for k in k_values:
        recall_at_k = np.mean(ranks < k)
        metrics[f"Recall@{k}"] = recall_at_k

    precision_at_1 = np.mean(ranks == 0)
    mrr = np.mean(1.0 / (ranks + 1))
    ndcg = np.mean(1.0 / np.log2(ranks + 2))

    metrics.update(
        {
            "Precision@1": precision_at_1,
            "MRR": mrr,
            "NDCG": ndcg,
            "Ground Truth Rank Distribution": ranks.tolist(),  # Convert to list for JSON
        }
    )

    # Convert everything to plain Python types
    cleaned_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, (np.float32, np.float64)):
            cleaned_metrics[key] = float(value)
        elif isinstance(value, np.ndarray):  # Should not happen with current logic but good practice
            cleaned_metrics[key] = value.tolist()
        else:
            cleaned_metrics[key] = value  # Handles lists and standard floats

    return cleaned_metrics


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model and preprocessing
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.checkpoint)
    tokenizer = open_clip.get_tokenizer(args.model)
    model.eval().to(device)

    # Load ground truth
    data = load_ground_truth(args.ground_truth)
    all_texts = [item["answer"] for item in data]
    text_tokens = tokenizer(all_texts).to(device)

    with torch.no_grad(), torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    ranks = []
    gt_similarities = []

    for idx, item in enumerate(tqdm(data, desc="Processing images")):
        image_path = os.path.join(args.input_folder, item["image"])
        image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)

        with torch.no_grad(), torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
            image_features = model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            sims = (100.0 * image_features @ text_features.T).squeeze(0).softmax(dim=-1)
            sims_np = sims.cpu().numpy()

            # Rank the scores (lower rank means more similar)
            ranked_indices = np.argsort(-sims_np)
            true_index = idx  # Ground truth index corresponds to data position
            rank = np.where(ranked_indices == true_index)[0][0]
            ranks.append(rank)
            gt_similarities.append(sims_np[true_index])

    metrics = compute_metrics(ranks)

    print("\n--- Evaluation Metrics ---")
    for k, v in metrics.items():
        if isinstance(v, list):
            print(f"{k}: [list of {len(v)} values]")
        else:
            print(f"{k}: {v:.4f}")

    # Optionally save the results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate TULIP model on image-text retrieval task.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the TULIP checkpoint.")
    parser.add_argument("--model", type=str, required=True, help="Model name, e.g., 'TULIP-so400m-14-384'")
    parser.add_argument("--input_folder", type=str, required=True, help="Folder containing the images.")
    parser.add_argument("--ground_truth", type=str, required=True, help="JSON file with image-text pairs.")
    parser.add_argument("--output", type=str, help="Output JSON file to save metrics.")

    args = parser.parse_args()
    main(args)


# python /home/davidchan/Projects/visual-puzzles/compute_clip_recall.py \
#   --checkpoint webli \
#   --model ViT-gopt-16-SigLIP2-384 \
#   --input_folder /home/davidchan/Projects/visual-puzzles/rebus_puzzle/image \
#   --ground_truth /home/davidchan/Projects/visual-puzzles/rebus_puzzle/answers.json \
#   --output ./metrics_siglip2_g.json
