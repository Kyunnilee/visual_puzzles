import os
import json
import argparse
from rich.console import Console
from rich.table import Table
from rich.text import Text

# Initialize console globally for easier access in helper functions
console = Console()


def get_metrics_from_file(filepath):
    """Reads a JSON file and extracts specified metrics."""
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
            metrics = {
                "Recall@1": data.get("Recall@1"),
                "Recall@5": data.get("Recall@5"),
                "Recall@10": data.get("Recall@10"),
                "Precision@1": data.get("Precision@1"),
                "MRR": data.get("MRR"),
                "NDCG": data.get("NDCG"),
            }
            return metrics
    except FileNotFoundError:
        console.print(f"[red]Error: File not found at {filepath}[/red]")
        return None
    except json.JSONDecodeError:
        console.print(f"[red]Error: Invalid JSON in file {filepath}. Skipping.[/red]")
        return None
    except Exception as e:
        console.print(f"[red]An unexpected error occurred while reading {filepath}: {e}. Skipping.[/red]")
        return None


def extract_model_identifier(filename):
    """
    Extracts a readable model identifier from the JSON filename.
    Assumes filenames are like 'metrics_model_name.json'.
    """
    identifier = filename.replace("metrics_", "").replace(".json", "")
    # You can add more sophisticated formatting here if needed,
    # e.g., replacing underscores with spaces, capitalizing words.
    return identifier


def main():
    parser = argparse.ArgumentParser(description="Discover and display CLIP recall metrics from JSON files.")
    parser.add_argument(
        "folder",
        nargs="?",  # Make the argument optional
        default=".",  # Default to the current directory
        help="The folder containing the JSON metric files (e.g., 'metrics_*.json'). Defaults to current directory.",
    )
    parser.add_argument(
        "-s",
        "--summary_file",
        type=str,
        help="Optional: Path to a JSON file where the summary of all metrics will be written.",
    )
    args = parser.parse_args()

    metrics_folder = args.folder

    if not os.path.isdir(metrics_folder):
        console.print(f"[red]Error: Folder '{metrics_folder}' not found.[/red]")
        return

    # Discover JSON files that start with 'metrics_'
    json_files = [f for f in os.listdir(metrics_folder) if f.startswith("metrics_") and f.endswith(".json")]

    if not json_files:
        console.print(f"[yellow]No 'metrics_*.json' files found in '{metrics_folder}'.[/yellow]")
        return

    results = []
    for filename in json_files:
        filepath = os.path.join(metrics_folder, filename)
        metrics = get_metrics_from_file(filepath)
        if metrics:
            model_identifier = extract_model_identifier(filename)
            results.append({"model": model_identifier, **metrics})

    if not results:
        console.print("[yellow]No valid metric data could be extracted from the JSON files.[/yellow]")
        return

    # Sort results by Model/Checkpoint name for consistent display
    results.sort(key=lambda x: x["model"])

    # --- Write summary JSON file if requested ---
    if args.summary_file:
        try:
            with open(args.summary_file, "w") as f:
                json.dump(results, f, indent=2)
            console.print(f"\n[green]Metrics summary successfully written to '{args.summary_file}'.[/green]")
        except Exception as e:
            console.print(f"[red]Error writing summary to '{args.summary_file}': {e}[/red]")

    # --- Determine the best value for each metric for table bolding ---
    best_metrics = {
        "Recall@1": None,
        "Recall@5": None,
        "Recall@10": None,
        "Precision@1": None,
        "MRR": None,
        "NDCG": None,
    }

    for metric_key in best_metrics.keys():
        current_max = -1.0  # Metrics are usually between 0 and 1, so -1 is a safe start
        has_valid_data = False
        for row_data in results:
            value = row_data.get(metric_key)
            if value is not None and isinstance(value, (int, float)):
                if not has_valid_data or value > current_max:
                    current_max = value
                    has_valid_data = True
        if has_valid_data:
            best_metrics[metric_key] = current_max
        else:
            best_metrics[metric_key] = None  # No valid numbers were found for this metric

    # Create a rich table
    table = Table(title="CLIP Model Recall and Precision Metrics")
    table.add_column("Model/Checkpoint", style="cyan", justify="left")
    table.add_column("Recall@1", style="magenta", justify="right")
    table.add_column("Recall@5", style="green", justify="right")
    table.add_column("Recall@10", style="yellow", justify="right")
    table.add_column("Precision@1", style="blue", justify="right")
    table.add_column("MRR", style="red", justify="right")
    table.add_column("NDCG", style="purple", justify="right")

    # Add rows to the table, bolding the best numbers
    for row_data in results:
        model = row_data["model"]

        # Helper to format and potentially bold
        def format_metric(value, best_value):
            if value is None:
                return "N/A"
            formatted_value = f"{value:.4f}"
            # Use a small tolerance for float comparison
            if best_value is not None and abs(value - best_value) < 1e-9:
                return f"[bold]{formatted_value}[/bold]"
            return formatted_value

        recall1 = format_metric(row_data["Recall@1"], best_metrics["Recall@1"])
        recall5 = format_metric(row_data["Recall@5"], best_metrics["Recall@5"])
        recall10 = format_metric(row_data["Recall@10"], best_metrics["Recall@10"])
        precision1 = format_metric(row_data["Precision@1"], best_metrics["Precision@1"])
        mrr = format_metric(row_data["MRR"], best_metrics["MRR"])
        ndcg = format_metric(row_data["NDCG"], best_metrics["NDCG"])

        table.add_row(model, recall1, recall5, recall10, precision1, mrr, ndcg)

    console.print(table)


if __name__ == "__main__":
    main()
