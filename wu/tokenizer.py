from collections import OrderedDict
from typing import Any, Dict

import numpy as np
import torch
from torch import Tensor


class Tokenizer2DBatchedLoRA:
    def __init__(
        self,
        token_dim: int,
        rank: int,
        alpha: int,
        pg_mapping: Dict[str, Dict[str, int]],
        padding_value: float = torch.nan,
    ):
        # Tokenization parameters
        self.token_size = [rank, token_dim]

        # LoRA parameters
        assert rank > 0, "rank should be a positive integer."
        assert alpha > 0, "alpha should be a positive integer."
        self.rank = rank
        self.alpha = alpha

        # Expand grouped entries in pg_mapping
        # Grouped entry format: {"type": "grouped", "num_experts": E, "sub_weights": {"w1": {...}, ...}}
        # Expanded into individual sub-weight entries with num_experts field
        self.pg_mapping = {}
        for module, dims in pg_mapping.items():
            if isinstance(dims, dict) and dims.get("type") == "grouped":
                num_experts = dims["num_experts"]
                for sub_name, sub_dims in dims["sub_weights"].items():
                    expanded_name = f"{module}.{sub_name}"
                    self.pg_mapping[expanded_name] = {
                        **sub_dims,
                        "num_experts": num_experts,
                    }
            else:
                self.pg_mapping[module] = dims

        self._module_names = list(self.pg_mapping.keys())

        # Padding value for incompatible shapes
        self.padding_value = padding_value

    def get_key_order(self, key: str) -> int:
        """
        key: str -- parameter key in state_dict
        return: int -- order index for sorting keys
        To ensure the order of tokens are the same during tokenization and detokenization,
        we need to define a consistent ordering of keys.
        """
        # Determine minor_order from module index in pg_mapping
        minor_order = None
        for idx, module_name in enumerate(self._module_names):
            if module_name in key:
                minor_order = idx
                break
        if minor_order is None:
            raise ValueError(f"Key {key} does not match any known module in pg_mapping.")

        # lora_As first, then lora_Bs
        if "lora_A" in key:
            major_order = 0
        elif "lora_B" in key:
            major_order = 1
        else:
            raise ValueError(f"Key {key} does not have lora_A or lora_B.")

        return major_order * len(self._module_names) + minor_order

    def tokenize(self, state_dict: Dict[str, Tensor]) -> Tensor:
        state_dict = OrderedDict(sorted(state_dict.items(), key=lambda item: self.get_key_order(item[0])))

        token_area = self.token_size[0] * self.token_size[1]
        all_tokens = []

        # Process each parameter separately to avoid mixing elements from different params
        for key, weight in state_dict.items():
            batch_size = weight.shape[0]

            # Flatten each batch's parameters for this specific weight
            flattened_weight = weight.view(batch_size, -1)  # [batch_size, param_count]
            param_count = flattened_weight.shape[1]

            # Calculate number of tokens needed for this parameter
            num_tokens_for_param = (param_count + token_area - 1) // token_area  # Ceiling division

            # Pad this parameter to fit exactly into tokens
            padded_length = num_tokens_for_param * token_area
            if param_count < padded_length:
                padding = torch.full(
                    (batch_size, padded_length - param_count),
                    self.padding_value,
                    dtype=flattened_weight.dtype,
                    device=flattened_weight.device,
                )
                flattened_weight = torch.cat([flattened_weight, padding], dim=1)

            # Reshape to tokens for this parameter: [batch_size, num_tokens_for_param, token_height, token_width]
            param_tokens = flattened_weight.view(
                batch_size, num_tokens_for_param, self.token_size[0], self.token_size[1]
            )
            all_tokens.append(param_tokens)

        # Concatenate all parameter tokens along the token dimension
        # Final shape: [batch_size, total_num_tokens, token_height, token_width]
        tokens = torch.cat(all_tokens, dim=1)

        return tokens

    def detokenize(self, shape_state_dict: Dict[str, tuple[int, int, int]], tokens: Tensor) -> OrderedDict[str, Tensor]:
        shape_state_dict = OrderedDict(sorted(shape_state_dict.items(), key=lambda item: self.get_key_order(item[0])))

        batch_size = tokens.shape[0]

        token_area = self.token_size[0] * self.token_size[1]
        detokenized_state_dict = OrderedDict()
        token_start = 0

        # Process each parameter separately
        for key, weight_shape in shape_state_dict.items():
            # Calculate parameter count for this weight (excluding batch dimension)
            param_count = np.prod(weight_shape)

            # Calculate number of tokens used for this parameter
            num_tokens_for_param = (param_count + token_area - 1) // token_area  # Ceiling division

            # Extract tokens for this parameter
            token_end = token_start + num_tokens_for_param
            param_tokens = tokens[
                :, token_start:token_end
            ]  # [batch_size, num_tokens_for_param, token_height, token_width]

            # Flatten tokens back to parameters
            flattened_param_tokens = param_tokens.view(batch_size, -1)  # [batch_size, padded_param_count]

            # Extract only the needed parameters (remove padding)
            param_data = flattened_param_tokens[:, :param_count]

            # Reshape to original weight shape
            weight = param_data.view(batch_size, *weight_shape)
            detokenized_state_dict[key] = weight

            token_start = token_end

        return detokenized_state_dict

    @property
    def lora_scale(self) -> float:
        return self.alpha / self.rank

    @property
    def shape_state_dict(self) -> OrderedDict[str, tuple]:
        """
        Generate shape state dict from pg_mapping config.
        Handles both regular entries and grouped entries (with num_experts).
        """
        shape_state_dict = {}
        for module in self._module_names:
            dims = self.pg_mapping[module]
            if "num_experts" in dims:
                E = dims["num_experts"]
                lora_A_shape = (E, self.rank, dims["lora_A_dim"])
                lora_B_shape = (E, dims["lora_B_dim"], self.rank)
            else:
                lora_A_shape = (self.rank, dims["lora_A_dim"])
                lora_B_shape = (dims["lora_B_dim"], self.rank)
            shape_state_dict[f"{module}.lora_A"] = lora_A_shape
            shape_state_dict[f"{module}.lora_B"] = lora_B_shape

        shape_state_dict = OrderedDict(sorted(shape_state_dict.items(), key=lambda x: self.get_key_order(x[0])))
        return shape_state_dict

    def _count_tokens(self, prefix: str) -> int:
        token_area = self.token_size[0] * self.token_size[1]
        count = 0
        for key, shape in self.shape_state_dict.items():
            if prefix in key:
                param_count = np.prod(shape)
                count += (param_count + token_area - 1) // token_area
        return count

    @property
    def lora_A_token_count(self) -> int:
        return self._count_tokens("lora_A")

    @property
    def lora_B_token_count(self) -> int:
        return self._count_tokens("lora_B")

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        # Make the tokenizer callable for checkpointing function
        return self.detokenize(*args, **kwds)
