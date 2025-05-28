import json
from pathlib import Path
from typing import Dict, List, Tuple
from textwrap import dedent as _d

import gradio as gr
from PIL import Image

# Configurable constants
DATA_JSON_PATH = Path("./answers.json")
IMAGE_DIR = Path("./image")
CATEGORY = [
    "Image Recognition",
    "Text Recognition (OCR + Typography/Layout)",
    "Font Style/Size",
    "Text Orientation",
    "Spatial and Positional Reasoning",
    "Phonetics and Wordplay",
    "Symbolic Substitution",
    "Visual Metaphors and Cultural References",
    "Letter and Word Manipulation",
    "Absence or Negation",
    "Quantitative or Mathematical Reasoning",
]

RUBRIC_VISUAL_REASONING_MD = _d(
    """
### **Visual Word Puzzle Concepts**
| Concept | Description |
| --- | --- |
| **Image Recognition** | Identifying objects, people, actions, or symbols in the image. |
| **Text Recognition (OCR + Typography/Layout)** | Detecting written words, fonts, capitalization, or stylized text. |
| **Font Style/Size** | Recognizing different font styles, sizes, or colors. |
| **Text Orientation** | Understanding text direction (e.g., upside down, rotated) and how it affects meaning. |
| **Spatial and Positional Reasoning** | Understanding (multi-)object layout or relationships (e.g., above/below, inside/outside) and how that changes meaning ("man in the moon"). |
| **Phonetics and Wordplay** | Homophones, puns, mondegreens ("10 issues" → "tennis shoes"). |
| **Symbolic Substitution** | Replacing with numbers, letters, or emojis (e.g., "4" → "for"). |
| **Visual Metaphors and Cultural References** | Idioms, memes, or visual sayings ("water" shaped like a waterfall). |
| **Letter and Word Manipulation** | Overlapping, hiding, or repeating letters to form new meanings. |
| **Absence or Negation** | Missing elements or crossed-out text (e.g., a gap = "invisible"). |
| **Quantitative or Mathematical Reasoning** | Math symbols, object counting (e.g., "1 2 3" + foot = "three feet"). |
"""
)


# Load image-label JSONL
def load_data() -> List[Dict]:
    with DATA_JSON_PATH.open("r", encoding="utf-8") as f:
        labels = json.load(f)
    return labels


def load_image(idx: int) -> Image.Image:
    return Image.open(IMAGE_DIR / f"{idx+1}.png")


# Update UI on load or step
def update_display(data, idx, annotations):
    if idx < 0 or idx >= len(data):
        return [gr.update(visible=False)] * 4
    img = load_image(idx)
    label = data[idx]["answer"]
    selected = annotations.get(idx, [])
    return img, label, f"Sample {idx + 1} / {len(data)}", selected


# Save and move forward/backward
def save_and_step(data, idx, annotations, selected_tags, direction):
    annotations[idx] = selected_tags
    if direction == "next":
        idx = min(idx + 1, len(data) - 1)
    elif direction == "prev":
        idx = max(idx - 1, 0)
    return data, idx, annotations


# Wrapper for buttons
def prev_cb(data, idx, ann, selected_tags):
    return save_and_step(data, idx, ann, selected_tags, "prev")


def next_cb(data, idx, ann, selected_tags):
    return save_and_step(data, idx, ann, selected_tags, "next")


def goto_sample(data, idx_str, annotations):
    try:
        idx = max(0, min(int(idx_str) - 1, len(data) - 1))
    except:
        idx = 0
    return data, idx, annotations


# Save to file
def download_annotations(data, annotations):
    out_path = Path("image_annotations.jsonl")
    with out_path.open("w", encoding="utf-8") as f:
        for idx, entry in enumerate(data):
            line = {
                "id": idx + 1,
                "annotation": annotations.get(idx, []),
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    return str(out_path)


# Build Gradio UI
def build_interface():
    with gr.Blocks(title="Image Annotation Tool") as demo:
        gr.Markdown("## 🖼️ Image Annotation Tool")

        data_state = gr.State([])
        idx_state = gr.State(0)
        ann_state = gr.State({})

        with gr.Row():
            img_display = gr.Image(label="Image", type="pil", height=400)
            with gr.Column():
                label_display = gr.Textbox(label="Ground Truth", interactive=False)
                idx_display = gr.Label(label="Label")
                intent_checkboxes = gr.CheckboxGroup(CATEGORY, label="Categories (You can select multiple)")
        # 📚 Rubric reference section
        with gr.Accordion("📚 Rubrics & Visual Concepts", open=False):
            gr.Markdown(RUBRIC_VISUAL_REASONING_MD)
        with gr.Row():
            prev_btn = gr.Button("⬅️ Previous")
            next_btn = gr.Button("Save / Next ➡️", variant="primary")
            go_input = gr.Textbox(label="Go to sample #", placeholder="e.g., 12", scale=1)
            go_btn = gr.Button("Go", scale=0)
            dl_btn = gr.Button("📥 Download Annotations")

        # Load data and init display
        demo.load(load_data, inputs=[], outputs=[data_state]).then(
            fn=update_display,
            inputs=[data_state, idx_state, ann_state],
            outputs=[img_display, label_display, idx_display, intent_checkboxes],
        )

        # Button logic
        prev_btn.click(
            prev_cb,
            inputs=[data_state, idx_state, ann_state, intent_checkboxes],
            outputs=[data_state, idx_state, ann_state],
        ).then(
            update_display,
            inputs=[data_state, idx_state, ann_state],
            outputs=[img_display, label_display, idx_display, intent_checkboxes],
        )

        next_btn.click(
            next_cb,
            inputs=[data_state, idx_state, ann_state, intent_checkboxes],
            outputs=[data_state, idx_state, ann_state],
        ).then(
            update_display,
            inputs=[data_state, idx_state, ann_state],
            outputs=[img_display, label_display, idx_display, intent_checkboxes],
        )

        go_btn.click(
            goto_sample, inputs=[data_state, go_input, ann_state], outputs=[data_state, idx_state, ann_state]
        ).then(
            update_display,
            inputs=[data_state, idx_state, ann_state],
            outputs=[img_display, label_display, idx_display, intent_checkboxes],
        )

        dl_btn.click(download_annotations, inputs=[data_state, ann_state], outputs=gr.File())

    return demo


if __name__ == "__main__":
    build_interface().launch()
