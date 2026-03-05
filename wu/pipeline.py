from typing import Optional

import torch
from PIL import Image

from .generator import ParameterGenerator
from .mixin import inject_pg
from .models import HunyuanImage3ForCausalMM


class WUPipeline:
    def __init__(
        self,
        base_model_path: str,
        pg_model_path: str,
        device_map: str = "auto",
        moe_impl: str = "eager",
        moe_drop_tokens: bool = False,
    ):
        # Load parameter generator
        self.parameter_generator = ParameterGenerator.from_pretrained(
            pg_model_path,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
        )
        self.parameter_generator.eval()

        # Load model with PG injection
        model_kwargs = dict(
            attn_implementation="sdpa",
            trust_remote_code=True,
            torch_dtype="auto",
            device_map=device_map,
            moe_impl=moe_impl,
            moe_drop_tokens=moe_drop_tokens,
        )
        self.model: HunyuanImage3ForCausalMM = inject_pg(
            self.parameter_generator.config, HunyuanImage3ForCausalMM
        ).from_pretrained(base_model_path, **model_kwargs)
        self.model.load_tokenizer(base_model_path)
        self.model.enable_pg()
        self.model.eval()

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        imgs_input: list[str | Image.Image],
        cot_text: Optional[str] = None,
        diff_infer_steps: int = 50,
        seed: int = 42,
        verbose: int = 2,
    ) -> Image.Image:
        # Step 1: Generate COT
        if cot_text is None:
            cot_text = self.model.generate_cot(
                prompt=prompt,
                image=imgs_input,
                seed=seed,
                image_size="auto",
                use_system_prompt="en_unified",
                bot_task="think_recaption",
                infer_align_image_size=True,
                diff_infer_steps=diff_infer_steps,
                verbose=verbose,
            )

        # Step 2: Extract hidden states
        condition = self.model.generate_image(
            prompt=prompt,
            image=imgs_input,
            seed=seed,
            image_size="auto",
            use_system_prompt="en_unified",
            bot_task="img_ratio",
            infer_align_image_size=True,
            diff_infer_steps=diff_infer_steps,
            cot_text=cot_text,
            return_hidden_states=True,
            verbose=0,
        )

        # Step 3: PG generates LoRA parameters
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pg_output = self.parameter_generator(condition=condition)
        pg_state_dict = {k: v for k, v in pg_output.items()}

        # Step 4: Generate image with LoRA
        _, samples = self.model.generate_image(
            prompt=prompt,
            image=imgs_input,
            seed=seed,
            image_size="auto",
            use_system_prompt="en_unified",
            bot_task="img_ratio",
            infer_align_image_size=True,
            diff_infer_steps=diff_infer_steps,
            cot_text=cot_text,
            pg_state_dict=pg_state_dict,
            verbose=verbose,
        )

        return samples[0]
