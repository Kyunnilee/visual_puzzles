import gradio as gr
import os
import json
from PIL import Image
import tempfile

# Configuration
IMAGE_FOLDER = "../dataset/image" # path to the image folder
ANNOTATION_FILE = "annotations.json"

# Load image paths
image_paths = sorted([f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(("png", "jpg", "jpeg"))])
annotations = []
current_index = 0


# Helper functions
def load_annotations(file_obj):
    global annotations, current_index
    file_path = file_obj.name if hasattr(file_obj, "name") else file_obj
    with open(file_path, "r") as f:
        annotations = json.load(f)
    done_images = {a["image"] for a in annotations}
    remaining = [img for img in image_paths if img not in done_images]
    current_index = 0 if not remaining else image_paths.index(remaining[0])
    return update_interface()


def save_annotations():
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json") as tmp:
        json.dump(annotations, tmp, indent=2)
        tmp_path = tmp.name
    return tmp_path


def update_interface():
    if current_index >= len(image_paths):
        skipped_count = sum(1 for a in annotations if a["answer"] == "<SKIP>")
        return None, "", f"All {len(image_paths)} images annotated. Skipped: {skipped_count}", 100
    image_path = os.path.join(IMAGE_FOLDER, image_paths[current_index])
    existing = next((a["answer"] for a in annotations if a["image"] == image_paths[current_index]), "")
    skipped_count = sum(1 for a in annotations if a["answer"] == "<SKIP>")
    return (
        image_path,
        existing if existing != "<SKIP>" else "",
        f"{len(annotations)} / {len(image_paths)} completed. Skipped: {skipped_count}",
        int((len(annotations) / len(image_paths)) * 100),
    )


def submit_answer(answer):
    global current_index
    if current_index < len(image_paths):
        img_name = image_paths[current_index]
        existing = next((a for a in annotations if a["image"] == img_name), None)
        if answer.strip():
            if existing:
                existing["answer"] = answer
            else:
                annotations.append({"image": img_name, "answer": answer})
        else:
            if not existing:
                annotations.append({"image": img_name, "answer": "<SKIP>"})

    # Find the next unannotated index
    for i in range(current_index + 1, len(image_paths)):
        next_img_name = image_paths[i]
        if not any(a["image"] == next_img_name and a["answer"] not in ("", "<SKIP>") for a in annotations):
            current_index = i
            break
    else:
        # If no unannotated images are found, set current_index to the end
        current_index = len(image_paths)

    return update_interface()


def go_previous():
    global current_index
    if current_index > 0:
        current_index -= 1
    return update_interface()


def go_next():
    global current_index
    if current_index < len(image_paths) - 1:
        current_index += 1
    return update_interface()


# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("## Rebus Puzzle Annotator")

    with gr.Row():
        image_display = gr.Image(label="Rebus Image", type="filepath")
        textbox = gr.Textbox(label="Your Answer", placeholder="Type the rebus answer here")

    with gr.Row():
        submit_btn = gr.Button("Submit Answer")
        prev_btn = gr.Button("Previous")
        next_btn = gr.Button("Next")

    with gr.Row():
        status = gr.Textbox(label="Progress")
        progress_bar = gr.Slider(label="Progress", minimum=0, maximum=100, step=1, interactive=False)

    with gr.Row():
        download_btn = gr.Button("Export Annotations")
        upload_btn = gr.File(label="Import Annotations", file_types=[".json"])
        output_file = gr.File(label="Download JSON")

    # Hook up events
    submit_btn.click(submit_answer, textbox, [image_display, textbox, status, progress_bar])
    textbox.submit(submit_answer, textbox, [image_display, textbox, status, progress_bar])
    prev_btn.click(go_previous, None, [image_display, textbox, status, progress_bar])
    next_btn.click(go_next, None, [image_display, textbox, status, progress_bar])
    download_btn.click(save_annotations, None, output_file)
    upload_btn.change(load_annotations, upload_btn, [image_display, textbox, status, progress_bar])

    demo.load(update_interface, None, [image_display, textbox, status, progress_bar])


if __name__ == "__main__":
    demo.launch()
