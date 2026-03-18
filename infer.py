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

prompt = "以图1为底图，将图2公仔穿的衣物换到图1人物身上；保持图1人物、姿态和背景不变，自然贴合并融合。"
# prompt_en = Using Figure 1 as the base image, replace the clothing on the character in Figure 1 with the outfit worn by the figurine in Figure 2. Keep the character, pose, and background of Figure 1 unchanged, ensuring the new clothing fits naturally and blends seamlessly.
imgs_input = ["./assets/input_1_1.png", "./assets/input_1_2.png"]

sample = pipeline.generate(
    prompt=prompt, imgs_input=imgs_input, diff_infer_steps=50, seed=42, verbose=2, image_size="1024x1024"
)

sample.save("./output.png")
