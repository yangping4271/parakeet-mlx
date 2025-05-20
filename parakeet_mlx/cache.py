import mlx.core as mx


class ConformerCache:
    keys: mx.array | None
    values: mx.array | None
    conv: mx.array | None

    offset: int
    step = 256

    def __init__(self):
        self.keys = None
        self.values = None
        self.offset = 0

    def update_and_fetch_kv(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        # k, v is [batch, head, seq, dim]
        prev = self.offset
        if (
            self.keys is None
            or self.values is None
            or (prev + keys.shape[2]) > self.keys.shape[2]
        ):
            B, H, S, D_KEYS = keys.shape
            _, _, _, D_VALUES = values.shape
            S_CACHE = ((self.step + S - 1) // self.step) * self.step

            new_k = mx.zeros((B, H, S_CACHE, D_KEYS), keys.dtype)
            new_v = mx.zeros((B, H, S_CACHE, D_VALUES), keys.dtype)

            if self.keys is None or self.values is None:  # type safety!
                self.keys, self.values = new_k, new_v
            else:
                if prev % self.step != 0:
                    self.keys = self.keys[..., :prev, :]
                    self.values = self.values[..., :prev, :]
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)

        self.offset += keys.shape[2]
        self.keys[..., prev : self.offset, :] = keys
        self.values[..., prev : self.offset, :] = values

        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]

    def update_and_fetch_conv(self, x: mx.array, padding: int = 0) -> mx.array:
        if padding == 0:
            return x

        B, S, D = x.shape

        if self.conv is None:
            self.conv = mx.zeros((B, padding, D), x.dtype)

        tokens_to_cache = min(padding, S)

        cache_update = x[:, S - tokens_to_cache : S, :]

        if tokens_to_cache < padding:
            self.conv = mx.concatenate(
                [self.conv[:, tokens_to_cache:, :], cache_update], axis=1
            )
        else:
            self.conv = cache_update

        result = mx.concatenate([self.conv, x], axis=1)
        result = mx.pad(result, ((0, 0), (0, padding)))

        return result


class RotatingConformerCache(ConformerCache):
    capacity: int
    cache_drop_size: int

    def __init__(self, capacity: int, cache_drop_size: int = 0):
        super().__init__()

        self.capacity = capacity
        self.cache_drop_size = cache_drop_size

    def update_and_fetch_kv(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        if self.keys is None or self.values is None:
            B, H, _, D_KEYS = keys.shape
            _, _, _, D_VALUES = values.shape
            self.keys = mx.zeros((B, H, self.capacity, D_KEYS), keys.dtype)
            self.values = mx.zeros((B, H, self.capacity, D_VALUES), values.dtype)

        S = keys.shape[2]

        if S <= self.cache_drop_size:
            pass
        else:
            tokens_to_cache = S - self.cache_drop_size
            keys_to_cache = keys[..., :tokens_to_cache, :]
            values_to_cache = values[..., :tokens_to_cache, :]

            pos = self.offset % self.capacity
            space_left = self.capacity - pos

            if tokens_to_cache <= space_left:
                self.keys[..., pos : pos + tokens_to_cache, :] = keys_to_cache
                self.values[..., pos : pos + tokens_to_cache, :] = values_to_cache
            else:
                self.keys[..., pos:, :] = keys_to_cache[..., :space_left, :]
                self.values[..., pos:, :] = values_to_cache[..., :space_left, :]

                remaining_s = tokens_to_cache - space_left
                self.keys[..., :remaining_s, :] = keys_to_cache[..., space_left:, :]
                self.values[..., :remaining_s, :] = values_to_cache[..., space_left:, :]

            self.offset += tokens_to_cache

        if self.offset <= self.capacity:
            k_out = self.keys[..., : self.offset, :]
            v_out = self.values[..., : self.offset, :]
        else:
            shift_amount = -(self.offset % self.capacity)

            k_out = mx.roll(self.keys, shift_amount, 2)
            v_out = mx.roll(self.values, shift_amount, 2)

        return k_out, v_out

    def update_and_fetch_conv(self, x: mx.array, padding: int = 0) -> mx.array:
        if padding == 0:
            return x

        B, S, D = x.shape

        if self.conv is None:
            self.conv = mx.zeros((B, padding, D), x.dtype)

        if S > self.cache_drop_size:
            tokens_to_cache = min(padding, S - self.cache_drop_size)
            cache_update = x[:, S - tokens_to_cache : S, :]

            if tokens_to_cache < padding:
                self.conv = mx.concatenate(
                    [self.conv[:, tokens_to_cache:, :], cache_update], axis=1
                )
            else:
                self.conv = cache_update

        result = mx.concatenate([self.conv, x], axis=1)
        result = mx.pad(result, ((0, 0), (0, padding)))

        return result
