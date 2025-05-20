import math

import mlx.core as mx
import mlx.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        n_head: int,
        n_feat: int,
        bias=True,
    ):
        super().__init__()

        self.n_head = n_head
        self.head_dim = n_feat // n_head
        self.scale = self.head_dim**-0.5

        self.linear_q = nn.Linear(n_feat, n_feat, bias=bias)
        self.linear_k = nn.Linear(n_feat, n_feat, bias=bias)
        self.linear_v = nn.Linear(n_feat, n_feat, bias=bias)
        self.linear_out = nn.Linear(n_feat, n_feat, bias=bias)

    def __call__(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        pos_emb: mx.array | None = None,
        mask: mx.array | None = None,
        cache=None,
    ) -> mx.array:
        q, k, v = self.linear_q(q), self.linear_k(k), self.linear_v(v)

        batch, q_seq, _ = q.shape
        _, k_seq, _ = k.shape

        q = q.reshape(batch, q_seq, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch, k_seq, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch, k_seq, self.n_head, self.head_dim).transpose(0, 2, 1, 3)

        if cache:
            k, v = cache.update_and_fetch_kv(k, v)

        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        o = o.transpose(0, 2, 1, 3).reshape(batch, q_seq, self.n_feat)

        return self.linear_out(o)


class RelPositionMultiHeadAttention(MultiHeadAttention):
    def __init__(
        self,
        n_head: int,
        n_feat: int,
        bias: bool = True,
        pos_bias_u: mx.array | None = None,
        pos_bias_v: mx.array | None = None,
    ):
        super().__init__(
            n_head=n_head,
            n_feat=n_feat,
            bias=bias,
        )

        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)

        if pos_bias_u is None:
            self._pos_bias_u_init = mx.zeros((self.n_head, self.head_dim))
        else:
            self._pos_bias_u_init = pos_bias_u

        if pos_bias_v is None:
            self._pos_bias_v_init = mx.zeros((self.n_head, self.head_dim))
        else:
            self._pos_bias_v_init = pos_bias_v

        self.pos_bias_u = self._pos_bias_u_init
        self.pos_bias_v = self._pos_bias_v_init

    def rel_shift(self, x: mx.array) -> mx.array:
        B, H, Tq, pos_len = x.shape
        padding = [(0, 0)] * (x.ndim - 1) + [(1, 0)]

        x = mx.pad(x, padding)
        x = x.reshape(B, H, pos_len + 1, Tq)
        x = x[:, :, 1:, :]
        x = x.reshape(B, H, Tq, pos_len)

        return x

    def __call__(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        pos_emb: mx.array | None = None,
        mask: mx.array | None = None,
        cache=None,
    ) -> mx.array:
        if pos_emb is None:
            raise ValueError("pos_emb is necessary!")

        q, k, v = self.linear_q(q), self.linear_k(k), self.linear_v(v)

        p = self.linear_pos(pos_emb)  # p stands for position

        batch, q_seq, _ = q.shape
        _, k_seq, _ = k.shape
        _, pos_len, _ = p.shape

        q = q.reshape(batch, q_seq, self.n_head, self.head_dim)
        q_u = (q + self.pos_bias_u).transpose(0, 2, 1, 3)
        q_v = (q + self.pos_bias_v).transpose(0, 2, 1, 3)

        k = k.reshape(batch, k_seq, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch, k_seq, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        p = p.reshape(batch, pos_len, self.n_head, self.head_dim).transpose(0, 2, 1, 3)

        if cache is not None:
            k, v = cache.update_and_fetch_kv(k, v)

        matrix_bd = mx.matmul(q_v, p.swapaxes(-2, -1))
        matrix_bd = self.rel_shift(matrix_bd)
        matrix_bd = matrix_bd[:, :, :, : k.shape[-2]] * self.scale

        if mask is not None:
            mask = mx.expand_dims(mask, 0)
            matrix_bd[mask] = -mx.inf

        o = mx.fast.scaled_dot_product_attention(
            q_u, k, v, scale=self.scale, mask=matrix_bd
        )
        o = o.transpose(0, 2, 1, 3).reshape(batch, q_seq, -1)

        return self.linear_out(o)


class RelPositionMultiHeadLocalAttention(RelPositionMultiHeadAttention):
    def __init__(
        self,
        n_head: int,
        n_feat: int,
        bias: bool = True,
        pos_bias_u: mx.array | None = None,
        pos_bias_v: mx.array | None = None,
        context_size: tuple[int, int] = (256, 256),
    ):
        super().__init__(n_head, n_feat, bias, pos_bias_u, pos_bias_v)

        self.context_size = context_size

        if min(context_size) <= 0:
            raise ValueError(
                "Context size for RelPositionMultiHeadLocalAttention must be > 0."
            )

    def __call__(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        pos_emb: mx.array | None = None,
        mask: mx.array | None = None,
        cache=None,
    ) -> mx.array:
        if pos_emb is None:
            raise ValueError("pos_emb is necessary!")

        q, k, v = self.linear_q(q), self.linear_k(k), self.linear_v(v)

        p = self.linear_pos(pos_emb)  # p stands for position

        batch, q_seq, _ = q.shape
        _, k_seq, _ = k.shape
        _, pos_len, _ = p.shape

        q = q.reshape(batch, q_seq, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch, k_seq, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch, k_seq, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        p = p.reshape(batch, pos_len, self.n_head, self.head_dim).transpose(0, 2, 1, 3)

        if cache is not None:
            k, v = cache.update_and_fetch_kv(k, v)

        # pad to fit context size
        w = max(self.context_size)
        pad_len = (2 * w - q.shape[2] % (2 * w)) % (2 * w)

        q = mx.pad(q, ((0, 0), (0, 0), (0, pad_len), (0, 0)))
        k = mx.pad(k, ((0, 0), (0, 0), (0, pad_len), (0, 0)))
        v = mx.pad(v, ((0, 0), (0, 0), (0, pad_len), (0, 0)))

        q_u = q + mx.expand_dims(self.pos_bias_u, 1)
        q_v = q + mx.expand_dims(self.pos_bias_v, 1)

        # lets not handle mask for now

        matrix_ac = self.matmul_qk(q_u, k, w)  # (batch, head, seq, 2w + 1)
        matrix_bd = mx.matmul(q_v, p.swapaxes(-2, -1))  # (batch, head, seq, 2w + 1)

        # we only add stuff in range and mask off unnecessaries
        matrix_ac[:, :, :, : self.context_size[0]] += matrix_bd[
            :, :, :, : self.context_size[0]
        ]
        matrix_ac[:, :, :, -(self.context_size[1] + 1) :] += matrix_bd[
            :, :, :, self.context_size[0] :
        ]
        matrix_ac[:, :, :, : (w - self.context_size[0])] = -mx.inf
        matrix_ac[:, :, :, (w + self.context_size[1] + 1) :] = -mx.inf

        scores = matrix_ac * self.scale

        attn = mx.softmax(scores, -1)
        out = self.matmul_pv(attn, v, w)

        out = out.reshape(batch, -1, self.n_head * self.head_dim)[:, :q_seq]

        return self.linear_out(out)

    def matmul_qk(self, q: mx.array, k: mx.array, w: int) -> mx.array:
        # TODO: very early and proof of concept. VERY MUCH MEMORY INEFFICIENT i should write metal kernel or whatever
        B, H, S_q, D = q.shape
        _, _, S_k, _ = k.shape

        k_pad = mx.pad(k, ((0, 0), (0, 0), (w, w), (0, 0)))

        # very much naive and consumes A LOT of memory right now.
        raw_idx = mx.arange(S_q)[:, None] + mx.arange(2 * w + 1)[None, :]
        idx = mx.clip(raw_idx, 0, S_k + 2 * w - 1)

        windows = k_pad[:, :, idx]
        scores = mx.einsum("bhsd,bhskd->bhsk", q, windows)

        mask = (raw_idx < w) | (raw_idx >= (S_k + w))
        mask = mask.reshape((1, 1, S_q, 2 * w + 1))

        scores = mx.where(mask, -mx.inf, scores)

        return scores

    def matmul_pv(self, prob: mx.array, v: mx.array, w: int) -> mx.array:
        # TODO: same with matmul_qk, very inefficient and skewing makes readibility really bad. i should write metal kernel for this too.
        B, H, S, _ = prob.shape
        L_chunks = S // w
        N = B * H  # merge batchs and heads temporary

        chunk_prob = prob.reshape(B * H, L_chunks, w, 2 * w + 1)

        # skewing like nemo implementation
        prob_pad = mx.pad(chunk_prob, ((0, 0), (0, 0), (0, 0), (0, w + 1)))
        prob_pad = prob_pad.reshape(N, L_chunks, -1)
        prob_pad = prob_pad[:, :, :-w]
        prob_pad = prob_pad.reshape(N, L_chunks, w, 3 * w + 1)
        skewed_prob = prob_pad[:, :, :, :-1]

        v_pad = mx.pad(v.reshape(N, S, -1), ((0, 0), (w, w), (0, 0)))

        # very much naive
        starts = mx.arange(0, L_chunks * w, w)
        idx = starts[:, None] + mx.arange(3 * w)[None, :]

        chunk_v = v_pad[:, idx, :]

        context = mx.einsum(
            "bcwd,bcdh->bcwh",
            skewed_prob,
            chunk_v,
        )

        return context.reshape(B, H, S, -1).transpose(0, 2, 1, 3)


class RelPositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        scale_input: bool = True,
    ):
        assert d_model % 2 == 0 and max_len > 0
        super().__init__()

        self.d_model = d_model
        self.max_len = max_len
        self.scale = math.sqrt(self.d_model) if scale_input else 1.0
        self.calculate_pe()

    def calculate_pe(self):
        positions = mx.arange(self.max_len - 1, -self.max_len, -1, dtype=mx.int32)
        positions = mx.expand_dims(positions, axis=1).astype(mx.float32)

        div_term = mx.exp(
            mx.arange(0, self.d_model, 2, dtype=mx.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe = mx.zeros((2 * self.max_len - 1, self.d_model), dtype=mx.float32)

        pe[:, 0::2] = mx.sin(positions * div_term)
        pe[:, 1::2] = mx.cos(positions * div_term)

        self._pe = mx.expand_dims(pe, axis=0).astype(mx.float32)

        mx.eval(self._pe)

    def __call__(self, x: mx.array, offset: int = 0) -> tuple[mx.array, mx.array]:
        input_len = x.shape[1] + offset

        if input_len > self.max_len:
            self.max_len = input_len + 1
            self.calculate_pe()

        x = x * self.scale

        buffer_len = self._pe.shape[1]
        start_idx = buffer_len // 2 - (input_len - 1)
        end_idx = buffer_len // 2 + (input_len - 1) + 1

        pos_emb = self._pe[:, start_idx:end_idx].astype(x.dtype)

        return x, pos_emb


class LocalRelPositionalEncoding(RelPositionalEncoding):
    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        scale_input: bool = True,
        context_size: tuple[int, int] = (256, 256),
    ):
        self.left_context, self.right_context = context_size

        super().__init__(d_model, max_len, scale_input)

    def calculate_pe(self):
        positions = mx.arange(
            self.left_context, -self.right_context - 1, -1, dtype=mx.int32
        )
        positions = mx.expand_dims(positions, axis=1).astype(mx.float32)

        div_term = mx.exp(
            mx.arange(0, self.d_model, 2, dtype=mx.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe = mx.zeros(
            (self.left_context + self.right_context + 1, self.d_model), dtype=mx.float32
        )

        pe[:, 0::2] = mx.sin(positions * div_term)
        pe[:, 1::2] = mx.cos(positions * div_term)

        self._pe = mx.expand_dims(pe, axis=0).astype(mx.float32)

        mx.eval(self._pe)

    def __call__(self, x: mx.array, offset: int = 0) -> tuple[mx.array, mx.array]:
        x = x * self.scale

        end_idx = self.left_context + self.right_context + 1
        pos_emb = self._pe[:, :end_idx].astype(x.dtype)

        return x, pos_emb
