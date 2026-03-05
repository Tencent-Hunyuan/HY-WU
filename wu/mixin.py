from typing import Dict

import torch
from torch import Tensor

from .config import ParameterGeneratorConfig


def batched_lora_forward_bmm(hidden: Tensor, loraAs: Tensor, loraBs: Tensor) -> Tensor:
    # Handle 2D case by adding sequence dimension
    if hidden.dim() == 2:
        input = hidden.unsqueeze(1)
    else:
        input = hidden

    # Transpose LoRA matrices for proper matrix multiplication
    # hidden: (B, L, D_in)
    # loraAs: (B, R, D_in) -> need (B, D_in, R) for matmul
    # loraBs: (B, D_out, R) -> need (B, R, D_out) for matmul

    loraAs_transposed = loraAs.transpose(1, 2)  # (B, R, D_in) -> (B, D_in, R)
    loraBs_transposed = loraBs.transpose(1, 2)  # (B, D_out, R) -> (B, R, D_out)

    # Use torch.matmul instead of torch.bmm to support broadcasting
    intermediate = torch.matmul(input, loraAs_transposed)  # (B, L, D_in) × (B/1, D_in, R) -> (B, L, R)
    output = torch.matmul(intermediate, loraBs_transposed)  # (B, L, R) × (B/1, R, D_out) -> (B, L, D_out)

    # Remove sequence dimension if original was 2D
    if hidden.dim() == 2:
        output = output.squeeze(1)

    return output


class PGMixin:
    """Mixin class to add PG functionality to any module

    This mixin should be used with nn.Module or its subclasses.
    It provides LoRA-based parameter generation functionality.

    Supports two types of target modules:
    1. Standard nn.Linear modules: LoRA is applied as an additive term
       output = original_forward(x) + (x @ lora_A^T @ lora_B^T) * scale
    2. Custom modules with pg_lora_forward: the module provides its own
       LoRA-aware forward that replaces the standard forward when LoRA is active.
    """

    def convert_to_pg(self, module_pg_mapping: Dict[str, str]) -> None:
        """
        Convert specified modules to PG-enabled modules

        Args:
            module_pg_mapping: Dict mapping module names to their corresponding PG keys
        """

        # Ensure this mixin is used with nn.Module
        if not hasattr(self, "get_submodule"):
            raise TypeError("PGMixin must be used with nn.Module or its subclasses")

        self._current_pg_state_dict = None  # Store current pg_state_dict
        self._activated_map = {}
        self._original_forwards = {}  # Store original forward methods

        for module_name, pg_key in module_pg_mapping.items():
            module = self.get_submodule(module_name)

            # Save original forward method
            self._original_forwards[module_name] = module.forward

            if hasattr(module, "pg_lora_forward"):
                # Custom LoRA forward — module provides its own strategy
                def create_pg_custom_forward(mod, orig_forward, key):
                    def pg_forward_func(*args, **kwargs):
                        return self._pg_custom_forward_wrapper(mod, orig_forward, key, *args, **kwargs)

                    return pg_forward_func

                module.forward = create_pg_custom_forward(module, module.forward, pg_key)
            else:
                # Default: nn.Linear-style additive LoRA patching
                def create_pg_forward(orig_forward, key):
                    def pg_forward_func(x):
                        return self._pg_forward_wrapper(x, orig_forward, key)

                    return pg_forward_func

                module.forward = create_pg_forward(module.forward, pg_key)

    def _remove_postfix(self, key: str) -> str:
        return key.replace(".lora_A", "").replace(".lora_B", "")

    def _lazy_device_transfer(self, key: str, device: torch.device) -> Tensor:
        """Move a tensor in pg_state_dict to the target device if needed, caching the result."""
        t = self._current_pg_state_dict[key]
        if t.device != device:
            t = t.to(device)
            self._current_pg_state_dict[key] = t
        return t

    def _pg_forward_wrapper(self, x: Tensor, original_forward, pg_key: str) -> Tensor:
        """Wrapper function for standard nn.Linear PG-enabled forward pass"""

        if self._current_pg_state_dict is not None:  # and f"{pg_key}.lora_A" in self._current_pg_state_dict:
            self._activated_map[pg_key] = True

            lora_A = self._lazy_device_transfer(f"{pg_key}.lora_A", x.device)
            lora_B = self._lazy_device_transfer(f"{pg_key}.lora_B", x.device)
            scaling = self._lazy_device_transfer("scale", x.device)

            lora_output = batched_lora_forward_bmm(x, lora_A, lora_B) * scaling

            return original_forward(x) + lora_output
        else:
            return original_forward(x)

    def _pg_custom_forward_wrapper(self, module, original_forward, pg_key: str, *args, **kwargs) -> Tensor:
        """Wrapper for modules that provide their own pg_lora_forward.

        When LoRA is active, calls module.pg_lora_forward with the original args
        plus the collected lora_dict and scale. The module's pg_lora_forward replaces
        the entire forward (it contains both base computation + LoRA).
        When LoRA is inactive, calls the original forward as-is.
        """
        if self._current_pg_state_dict is not None:
            # Collect all LoRA weights whose keys start with this module's pg_key
            # Infer target device from the first positional arg (input tensor)
            target_device = args[0].device if args and isinstance(args[0], Tensor) else None
            prefix = f"{pg_key}."
            lora_dict = {
                k[len(prefix) :]: self._lazy_device_transfer(k, target_device)
                for k in self._current_pg_state_dict
                if k.startswith(prefix)
            }
            for k in lora_dict.keys():
                self._activated_map[self._remove_postfix(prefix + k)] = True
            scale = self._lazy_device_transfer("scale", target_device)

            return module.pg_lora_forward(*args, lora_dict=lora_dict, scale=scale, **kwargs)
        else:
            return original_forward(*args, **kwargs)

    def set_pg_state_dict(self, pg_state_dict: Dict[str, Tensor]) -> None:
        """Set the pg_state_dict for all PG-enabled submodules"""

        self._current_pg_state_dict = pg_state_dict
        self._activated_map = {}
        for key in pg_state_dict.keys():
            if key == "scale":
                continue
            self._activated_map[self._remove_postfix(key)] = False

    def clear_pg_state_dict(self) -> None:
        """Clear the pg_state_dict for all PG-enabled submodules"""
        for key, activated in self._activated_map.items():
            if not activated:
                print(f"[WARNING] PG key {key} is not activated!")
        self._activated_map = {}
        self._current_pg_state_dict = None


def create_pg_module(base_class, module_pg_mapping: Dict[str, str] = {}, auto_convert: bool = False):
    """Factory function to create PG-enabled version of any module class.

    By default, conversion is NOT automatic; call instance.enable_pg() explicitly.
    Set auto_convert=True to preserve previous behavior.
    """

    class PGModule(PGMixin, base_class):
        _pg_mapping = module_pg_mapping

        def __init__(self, *args, **kwargs):
            # Initialize parent with all original parameters
            base_class.__init__(self, *args, **kwargs)

            if auto_convert and self._pg_mapping:
                self.convert_to_pg(self._pg_mapping)

        def enable_pg(self):
            if self._pg_mapping:
                self.convert_to_pg(self._pg_mapping)

    return PGModule


def inject_pg(pg_config: ParameterGeneratorConfig, model_class):
    """Create a PG-enabled HunyuanImage3ForCausalMM class from config.

    Args:
        pg_config: dict with "prefix" and "pg_mapping" keys.
    """
    prefix = pg_config.prefix
    num_layers = pg_config.num_base_model_layers
    modules = pg_config.pg_mapping
    all_layer_pg_mapping = {}
    for i in range(num_layers):
        for m in modules.keys():
            full_path = f"{prefix}{i}.{m}"
            all_layer_pg_mapping[full_path] = full_path
    return create_pg_module(model_class, module_pg_mapping=all_layer_pg_mapping)
