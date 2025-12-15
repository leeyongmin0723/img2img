import gradio as gr
import torch
from PIL import Image
from torchvision import transforms

from src.cyclegan_turbo import CycleGAN_Turbo
from src.my_utils.training_utils import build_transform


# =========================
# Device
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# Model registry (UNPAIRED only)
# =========================
UNPAIRED_MODELS = {
    "Day → Night": "day_to_night",
    "Night → Day": "night_to_day",
    "Clear → Rain": "clear_to_rainy",
    "Rain → Clear": "rainy_to_clear",
}

_loaded_models = {}


# =========================
# Lazy model loader
# =========================
def load_model(task):
    if task in _loaded_models:
        return _loaded_models[task]

    model = CycleGAN_Turbo(
        pretrained_name=UNPAIRED_MODELS[task]
    ).to(DEVICE)
    model.eval()

    _loaded_models[task] = model
    return model


# =========================
# Inference
# =========================
def run(task, image):
    if image is None:
        return None

    model = load_model(task)

    image = image.convert("RGB")

    # GitHub original preprocessing
    T = build_transform("resize_512x512")
    img_t = T(image)

    x = transforms.ToTensor()(img_t)
    x = transforms.Normalize([0.5]*3, [0.5]*3)(x)
    x = x.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = model(x)

    result = transforms.ToPILImage()(out[0].cpu() * 0.5 + 0.5)
    return result.resize(image.size)


# =========================
# UI
# =========================
with gr.Blocks(title="img2img-turbo (Stable Unpaired)") as demo:
    gr.Markdown("## 🧠 img2img-turbo (GitHub Original · Stable)")
    gr.Markdown("### Day/Night · Weather Translation")

    task = gr.Radio(
        choices=list(UNPAIRED_MODELS.keys()),
        value="Day → Night",
        label="변환 기능"
    )

    image = gr.Image(
        label="입력 이미지",
        type="pil"
    )

    output = gr.Image(label="결과 이미지")

    run_btn = gr.Button("Run")

    run_btn.click(
        fn=run,
        inputs=[task, image],
        outputs=output
    )

demo.launch()
