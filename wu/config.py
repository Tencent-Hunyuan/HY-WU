from transformers import PretrainedConfig


class ParameterGeneratorConfig(PretrainedConfig):
    model_type = "parameter_generator"

    def __init__(
        self,
        token_dim: int = 1024,
        rank: int = 16,
        alpha: int = 64,
        # model config
        input_dim: int = 4096,
        dim_accumulation: int = 4,
        d_model: int = 4096,
        head_dim: int = 128,
        num_base_model_layers: int = 32,
        output_dim: int = 1024,
        num_pg_layers: int = 24,
        # pg injection config
        prefix: str = "model.layers.",
        pg_mapping: dict = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.token_dim = token_dim
        self.rank = rank
        self.alpha = alpha
        # model hyper-parameters
        self.input_dim = input_dim
        self.dim_accumulation = dim_accumulation
        self.d_model = d_model
        self.head_dim = head_dim
        self.num_base_model_layers = num_base_model_layers
        self.output_dim = output_dim
        self.num_pg_layers = num_pg_layers
        # pg injection
        self.prefix = prefix
        self.pg_mapping = (
            pg_mapping
            if pg_mapping is not None
            else {
                "self_attn.qkv_proj": {"lora_A_dim": 4096, "lora_B_dim": 6144},
                "self_attn.o_proj": {"lora_A_dim": 4096, "lora_B_dim": 4096},
                "mlp.shared_mlp.gate_and_up_proj": {"lora_A_dim": 4096, "lora_B_dim": 6144},
                "mlp.shared_mlp.down_proj": {"lora_A_dim": 3072, "lora_B_dim": 4096},
                "mlp.experts": {
                    "type": "grouped",
                    "num_experts": 64,
                    "sub_weights": {
                        "w1": {"lora_A_dim": 4096, "lora_B_dim": 3072},
                        "w2": {"lora_A_dim": 3072, "lora_B_dim": 4096},
                        "w3": {"lora_A_dim": 4096, "lora_B_dim": 3072},
                    },
                },
            }
        )

    def to_pg_config_dict(self) -> dict:
        """Convert back to the legacy pg_config dict format for backward compatibility."""
        return {
            "token_dim": self.token_dim,
            "rank": self.rank,
            "alpha": self.alpha,
            "model": {
                "input_dim": self.input_dim,
                "dim_accumulation": self.dim_accumulation,
                "d_model": self.d_model,
                "head_dim": self.head_dim,
                "num_base_model_layers": self.num_base_model_layers,
                "output_dim": self.output_dim,
                "num_pg_layers": self.num_pg_layers,
            },
            "prefix": self.prefix,
            "pg_mapping": self.pg_mapping,
        }

    @classmethod
    def from_pg_config_dict(cls, pg_config: dict, **kwargs) -> "ParameterGeneratorConfig":
        """Create config from legacy pg_config dict format."""
        model_cfg = pg_config["model"]
        return cls(
            token_dim=pg_config["token_dim"],
            rank=pg_config["rank"],
            alpha=pg_config["alpha"],
            input_dim=model_cfg["input_dim"],
            dim_accumulation=model_cfg["dim_accumulation"],
            d_model=model_cfg["d_model"],
            head_dim=model_cfg["head_dim"],
            num_base_model_layers=model_cfg["num_base_model_layers"],
            output_dim=model_cfg["output_dim"],
            num_pg_layers=model_cfg["num_pg_layers"],
            prefix=pg_config["prefix"],
            pg_mapping=pg_config["pg_mapping"],
            **kwargs,
        )
