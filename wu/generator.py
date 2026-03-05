from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor
from transformers import PreTrainedModel

from .config import ParameterGeneratorConfig
from .model import TransformerModel
from .tokenizer import Tokenizer2DBatchedLoRA


class ParameterGenerator(PreTrainedModel):
    config_class = ParameterGeneratorConfig
    _no_split_modules = ["TransformerBlock"]

    def __init__(self, config: ParameterGeneratorConfig):
        super().__init__(config)

        self.tokenizer = Tokenizer2DBatchedLoRA(
            token_dim=config.token_dim,
            rank=config.rank,
            alpha=config.alpha,
            pg_mapping=config.pg_mapping,
        )
        self.lora_A_token_count = self.tokenizer.lora_A_token_count
        self.lora_B_token_count = self.tokenizer.lora_B_token_count

        self.model = TransformerModel(
            d_model=config.d_model,
            num_base_model_layers=config.num_base_model_layers,
            num_token_per_layer=self.lora_A_token_count + self.lora_B_token_count,
            lora_rank=config.rank,
            output_dim=config.output_dim,
            head_dim=config.head_dim,
            num_blocks=config.num_pg_layers,
            dim_accumulation=config.dim_accumulation,
            lora_A_token_count=self.lora_A_token_count,
            lora_B_token_count=self.lora_B_token_count,
        )

        self.layer_num = config.num_base_model_layers
        self.prefix = config.prefix
        self.lora_rank = config.rank

        self.hidden_in = nn.Linear(config.input_dim, config.d_model)

    def get_lora_count(self):
        return (
            self.config.output_dim
            * (self.lora_A_token_count + self.lora_B_token_count)
            * self.config.num_base_model_layers
            * self.lora_rank
        )

    def forward(self, condition: Tensor) -> Dict[str, Tensor]:
        embeddings = self.hidden_in(condition)

        # (batch_size, num_layers, token_num, token_size_0, token_size_1)
        output = self.model(encoder_hidden_states=embeddings)

        all_layer_state_dict = {}
        shape_state_dict = self.tokenizer.shape_state_dict
        for layer_index in range(self.layer_num):
            layer_output = output[:, layer_index]  # (batch_size, token_num, token_size_0, token_size_1)
            layer_state_dict = self.tokenizer.detokenize(
                shape_state_dict, layer_output
            )  # OrderedDict of (param_name: (batch_size, *param_shape))
            for key, value in layer_state_dict.items():
                new_key = f"{self.prefix}{layer_index}.{key}"
                all_layer_state_dict[new_key] = value

        all_layer_state_dict["scale"] = torch.tensor(self.tokenizer.lora_scale).to(
            device=output.device, dtype=output.dtype
        )

        return all_layer_state_dict
