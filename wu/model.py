import torch
from einops import rearrange, repeat
from torch import Tensor, nn
from torch.nn import functional as F


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=pos.dtype, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


def apply_cross_rope(xq: Tensor, xk: Tensor, q_freqs: Tensor, k_freqs: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = q_freqs[..., 0] * xq_[..., 0] + q_freqs[..., 1] * xq_[..., 1]
    xk_out = k_freqs[..., 0] * xk_[..., 0] + k_freqs[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


class PosEmbedND(nn.Module):
    def __init__(self, dim: int, theta: int):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(
        self,
        ids: Tensor = None,
        axes_dim: list[int] = None,
        axes_lengths: list[int] = None,
        device: torch.device = None,
    ) -> Tensor:
        assert sum(axes_dim) == self.dim, f"The sum of axes_dim {axes_dim} must be equal to dim {self.dim}"
        if ids is None:
            ids = self._create_ids(axes_dim, axes_lengths, device)

        emb = torch.cat(
            [rope(ids[..., i], axes_dim[i], self.theta) for i in range(len(axes_dim))],
            dim=-3,
        )
        return emb.unsqueeze(1)

    def _create_ids(self, axes_dim: list[int], axes_lengths: list[int], device: torch.device = None) -> Tensor:
        assert len(axes_lengths) == len(axes_dim)
        ranges = [torch.arange(length, device=device) for length in axes_lengths]
        grids = torch.meshgrid(*ranges, indexing="ij")
        ids = torch.stack([g.flatten() for g in grids], dim=-1)
        return ids.unsqueeze(0)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout

        self.W_q = nn.Linear(self.d_model, self.d_model)
        self.W_k = nn.Linear(self.d_model, self.d_model)
        self.W_v = nn.Linear(self.d_model, self.d_model)
        self.W_o = nn.Linear(self.d_model, self.d_model)

    def forward(self, x: Tensor, freqs: Tensor) -> Tensor:
        batch_size, seq_length, _ = x.shape

        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        q = q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rope(q, k, freqs)

        attn_output = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            dropout_p=self.dropout if self.training else 0.0,
        )

        output = attn_output.transpose(1, 2).reshape(batch_size, seq_length, self.d_model)
        output = self.W_o(output)

        return output


class MultiHeadCrossAttention(MultiHeadAttention):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__(d_model=d_model, num_heads=num_heads, dropout=dropout)

    def forward(self, q: Tensor, kv: Tensor, q_freqs: Tensor, kv_freqs: Tensor) -> Tensor:
        batch_size, q_seq_length, _ = q.shape
        _, kv_seq_length, _ = kv.shape

        q = self.W_q(q)
        k = self.W_k(kv)
        v = self.W_v(kv)

        q = q.view(batch_size, q_seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, kv_seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, kv_seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        q, k = apply_cross_rope(q, k, q_freqs, kv_freqs)

        attn_output = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            dropout_p=self.dropout if self.training else 0.0,
        )

        output = attn_output.transpose(1, 2).reshape(batch_size, q_seq_length, self.d_model)
        output = self.W_o(output)

        return output


class PositionWiseFeedForward(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
    ):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.act = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        seq_len_n: int,
        seq_len_hw: int,
    ):
        super(TransformerBlock, self).__init__()
        self.self_attn_n = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.self_attn_hw = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.self_attn_cross = MultiHeadCrossAttention(d_model=d_model, num_heads=num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model=d_model, d_ff=d_ff)
        self.norm1 = nn.RMSNorm(d_model, elementwise_affine=True, eps=1e-6)
        self.norm2 = nn.RMSNorm(d_model, elementwise_affine=True, eps=1e-6)
        self.norm3 = nn.RMSNorm(d_model, elementwise_affine=True, eps=1e-6)
        self.norm4 = nn.RMSNorm(d_model, elementwise_affine=True, eps=1e-6)

    def forward(
        self,
        x: Tensor,
        encoder_hidden_states: Tensor,
        hw_freqs: Tensor,
        n_freqs: Tensor,
        l_freqs: Tensor,
        all_freqs: Tensor,
    ) -> Tensor:
        batch, n, hw, _ = x.shape

        norm_x = self.norm1(x)
        norm_x = rearrange(norm_x, "b n hw c -> (b hw) n c")
        out = self.self_attn_n(norm_x, freqs=n_freqs)
        out = rearrange(out, "(b hw) n c -> b n hw c", b=batch, hw=hw)
        x = x + out

        norm_x = self.norm2(x)
        norm_x = rearrange(norm_x, "b n hw c -> (b n) hw c")
        out = self.self_attn_hw(norm_x, freqs=hw_freqs)
        out = rearrange(out, "(b n) hw c -> b n hw c", b=batch, n=n)
        x = x + out

        norm_x = self.norm3(x)
        norm_x = rearrange(norm_x, "b n hw c -> b (n hw) c")
        out = self.self_attn_cross(norm_x, encoder_hidden_states, q_freqs=all_freqs, kv_freqs=l_freqs)
        out = rearrange(out, "b (n hw) c -> b n hw c", b=batch, n=n)
        x = x + out

        norm_x = self.norm4(x)
        out = self.feed_forward(norm_x)
        x = x + out

        return x


class TransformerModel(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_base_model_layers: int,
        num_token_per_layer: int,
        lora_rank: int,
        dim_accumulation: int,
        output_dim: int,
        head_dim: int,
        num_blocks: int,
        lora_A_token_count: int,
        lora_B_token_count: int,
    ):
        super().__init__()
        self.d_model = d_model
        self.head_dim = head_dim
        self.num_heads = self.d_model // self.head_dim
        self.dim_accumulation = dim_accumulation
        self.dim_wo_accumulation = self.d_model // self.dim_accumulation

        self.seq_n = num_base_model_layers
        self.seq_h = num_token_per_layer
        self.seq_w = lora_rank
        self.output_dim = output_dim

        self.lora_A_token_count = lora_A_token_count
        self.lora_B_token_count = lora_B_token_count

        self.pos_embed = PosEmbedND(dim=self.head_dim, theta=10000)

        self.norm_input = nn.RMSNorm(self.d_model, elementwise_affine=True, eps=1e-6)
        self.norm_tokens = nn.RMSNorm(self.d_model, elementwise_affine=True, eps=1e-6)

        self.hw_pos = nn.Embedding(
            num_token_per_layer * lora_rank,
            self.dim_wo_accumulation,
        )
        self.layer_pos = nn.Embedding(
            num_base_model_layers,
            self.dim_wo_accumulation,
        )

        self.norm_final = nn.RMSNorm(self.d_model, elementwise_affine=True, eps=1e-6)

        module_list = []
        for _ in range(num_blocks):
            module_list.append(
                TransformerBlock(
                    d_model=self.d_model,
                    num_heads=self.num_heads,
                    d_ff=self.d_model * 4,
                    seq_len_n=self.seq_n,
                    seq_len_hw=self.seq_h * self.seq_w,
                )
            )
        self.blocks = nn.ModuleList(module_list)

        self.proj_out_A = nn.Linear(self.dim_wo_accumulation, self.output_dim)
        self.proj_out_B = nn.Linear(self.dim_wo_accumulation, self.output_dim)

    @torch.no_grad()
    def _get_pos_embed(self, device: torch.device, cond_seq_length: int) -> Tensor:
        # hw attention: 2D grid (h, w)
        hw_freqs = self.pos_embed(
            axes_dim=[self.head_dim // 2, self.head_dim // 2],
            axes_lengths=[self.seq_h, self.seq_w // self.dim_accumulation],
            device=device,
        )
        # n attention: 1D sequence
        n_freqs = self.pos_embed(axes_dim=[self.head_dim], axes_lengths=[self.seq_n], device=device)

        # cross attention Q: 2D grid (n, hw)
        all_freqs = self.pos_embed(
            axes_dim=[self.head_dim // 2, self.head_dim // 2],
            axes_lengths=[self.seq_n, self.seq_h * self.seq_w // self.dim_accumulation],
            device=device,
        )

        # cross attention KV: diagonal positions
        diag_ids = torch.arange(cond_seq_length, device=device)
        diag_ids = torch.stack([diag_ids, diag_ids], dim=-1).unsqueeze(0)  # (1, L, 2)
        diag_ids = diag_ids + max(self.seq_h, self.seq_w // self.dim_accumulation)
        l_freqs = self.pos_embed(
            ids=diag_ids,
            axes_dim=[self.head_dim // 2, self.head_dim // 2],
        )
        return hw_freqs, n_freqs, l_freqs, all_freqs

    def get_tokens(self, batch_size: int) -> Tensor:
        hw_pos = repeat(self.hw_pos.weight, "hw d -> b n hw d", b=batch_size, n=self.seq_n).clone()  # need clone
        layer_pos = repeat(self.layer_pos.weight, "n d -> b n hw d", b=batch_size, hw=self.seq_h * self.seq_w).clone()
        tokens = hw_pos + layer_pos
        tokens = rearrange(
            tokens,
            "b n (h w a) d -> b n (h w) (a d)",
            a=self.dim_accumulation,
            h=self.seq_h,
            w=self.seq_w // self.dim_accumulation,
        )
        tokens = self.norm_tokens(tokens)
        return tokens

    def post_process(self, x: Tensor) -> Tensor:
        x = self.norm_final(x)

        x = rearrange(
            x,
            "b n (h w) (a c) -> b n h (w a) c",
            a=self.dim_accumulation,
            h=self.seq_h,
            w=self.seq_w // self.dim_accumulation,
        )

        x_A, x_B = torch.split(x, [self.lora_A_token_count, self.lora_B_token_count], dim=2)

        x_A = self.proj_out_A(x_A)
        x_B = self.proj_out_B(x_B)
        x = torch.cat([x_A, x_B], dim=2)

        return x

    def forward(self, encoder_hidden_states: Tensor) -> Tensor:
        # x: (batch, 1, sequence_length, dim)
        # out: (batch, num_layers, token_per_layer, rank, dimension)

        encoder_hidden_states = encoder_hidden_states.squeeze(1)
        encoder_hidden_states = self.norm_input(encoder_hidden_states)
        hw_freqs, n_freqs, l_freqs, all_freqs = self._get_pos_embed(
            device=encoder_hidden_states.device,
            cond_seq_length=encoder_hidden_states.shape[1],
        )
        x = self.get_tokens(batch_size=encoder_hidden_states.shape[0])

        for block in self.blocks:
            x = block(
                x=x,
                encoder_hidden_states=encoder_hidden_states,
                hw_freqs=hw_freqs,
                n_freqs=n_freqs,
                l_freqs=l_freqs,
                all_freqs=all_freqs,
            )

        x = self.post_process(x)

        return x
