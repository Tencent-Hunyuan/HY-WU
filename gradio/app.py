import os
import re
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
from wu import WUPipeline

base_model_path = "tencent/HunyuanImage-3.0-Instruct"
pg_model_path = "tencent/HY-WU"

pipeline = WUPipeline(
    base_model_path=base_model_path,
    pg_model_path=pg_model_path,
    device_map="auto",
    moe_impl="eager",
    moe_drop_tokens=False,
)


def generate(prompt, imgs, image_size, diff_infer_steps, seed):
    if not prompt:
        raise gr.Error("请输入 prompt")
    if not imgs or len(imgs) == 0:
        raise gr.Error("请上传至少一张图片")

    image_size_str = str(image_size).strip().lower()
    if image_size_str != "auto":
        if not re.match(r"^\d+x\d+$", image_size_str):
            raise gr.Error("image_size 格式需为 auto 或 WxH（例如 1024x1024）")

    imgs_input = [img.name if hasattr(img, "name") else img for img in imgs]

    sample = pipeline.generate(
        prompt=prompt,
        imgs_input=imgs_input,
        image_size=image_size_str,
        diff_infer_steps=int(diff_infer_steps),
        seed=int(seed),
        verbose=2,
    )
    return sample


with gr.Blocks(title="HunyuanImage WU", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎨 HunyuanImage WU — 图像生成")
    gr.Markdown("上传一张或多张参考图，输入 prompt，即可生成图像。")

    with gr.Row():
        with gr.Column(scale=1):
            prompt = gr.Textbox(
                label="Prompt",
                placeholder="请输入 prompt …",
                lines=4,
            )
            imgs = gr.File(
                label="输入图片（支持多张）",
                file_count="multiple",
                file_types=["image"],
            )
            image_size = gr.Textbox(
                label="Image Size",
                value="auto",
                placeholder="auto 或 1024x1024",
            )
            with gr.Row():
                diff_infer_steps = gr.Slider(
                    label="Diffusion Steps",
                    minimum=1,
                    maximum=100,
                    value=50,
                    step=1,
                )
                seed = gr.Number(
                    label="Seed",
                    value=42,
                    precision=0,
                )
            run_btn = gr.Button("🚀 生成", variant="primary")

        with gr.Column(scale=1):
            output_image = gr.Image(label="生成结果", type="pil")

    run_btn.click(
        fn=generate,
        inputs=[prompt, imgs, image_size, diff_infer_steps, seed],
        outputs=output_image,
    )

    gr.Examples(
        examples=[
            [
                "以图1为底图，将图2公仔穿的衣物换到图1人物身上；保持图1人物、姿态和背景不变，自然贴合并融合。",
                ["./assets/input_1_1.png", "./assets/input_1_2.png"],
                "auto",
                50,
                42,
            ],
        ],
        inputs=[prompt, imgs, image_size, diff_infer_steps, seed],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
