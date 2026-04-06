from __future__ import annotations

from typing import Optional, Tuple
import heapq
import torch


class ReplayBuffer:
    """
    Streaming-friendly replay buffer for tensors shaped [B, D].

    Features:
      - Main buffer: reservoir sampling => ~uniform over all samples seen so far.
      - Optional hard buffer: keeps high-loss samples via a min-heap.
      - Sampling can mix main/hard with a fixed ratio.

    Stores data on CPU with configurable dtype (float64 by default).
    """

    def __init__(
        self,
        *,
        dim: int,
        capacity: int = 200_000,  # rows
        hard_capacity: int = 50_000,  # rows
        hard_mix: float = 0.5,  # fraction of replay drawn from hard buffer
        pin_memory: bool = True,
        seed: int = 0,
        dtype: torch.dtype = torch.float64,
    ):
        self.dim = int(dim)
        self.capacity = int(capacity)
        self.hard_capacity = int(hard_capacity)
        self.hard_mix = float(hard_mix)
        self.pin_memory = bool(pin_memory)
        self.dtype = dtype

        if self.dim <= 0:
            raise ValueError("dim must be > 0")
        if self.capacity < 0:
            raise ValueError("capacity must be >= 0")
        if self.hard_capacity < 0:
            raise ValueError("hard_capacity must be >= 0")
        if not (0.0 <= self.hard_mix <= 1.0):
            raise ValueError("hard_mix must be in [0,1]")

        self._g = torch.Generator(device="cpu")
        self._g.manual_seed(int(seed))

        # Main reservoir
        self._main = torch.empty(
            (max(1, self.capacity), self.dim), dtype=self.dtype, device="cpu"
        )
        self._main_size = 0
        self._seen = 0  # total rows ever offered to reservoir

        # Hard buffer
        self._hard = torch.empty(
            (max(1, self.hard_capacity), self.dim), dtype=self.dtype, device="cpu"
        )
        self._hard_heap: list[Tuple[float, int, int]] = []  # (loss, id, idx)
        self._hard_size = 0
        self._hard_next_id = 0

    @property
    def seen(self) -> int:
        return self._seen

    @property
    def size(self) -> int:
        return self._main_size

    @property
    def hard_size(self) -> int:
        return self._hard_size

    def add(
        self, batch_xy: torch.Tensor, *, losses: Optional[torch.Tensor] = None
    ) -> None:
        """
        Add raw samples to replay.

        batch_xy: [B, D] (D==dim). Can be CPU/CUDA, any float dtype.
        losses:   optional [B] float tensor, used for hard buffer.
        """
        if batch_xy.ndim != 2 or batch_xy.size(1) != self.dim:
            raise ValueError(
                f"ReplayBuffer.add expects [B,{self.dim}], got {tuple(batch_xy.shape)}"
            )

        # Early-out if nothing enabled
        if self.capacity == 0 and self.hard_capacity == 0:
            return

        # Move to CPU target dtype once
        x = batch_xy.detach()
        if x.is_cuda:
            x = x.to("cpu", non_blocking=False)
        x = x.to(self.dtype).contiguous()

        B = x.size(0)

        # Prepare losses on CPU if provided
        l_cpu: Optional[torch.Tensor] = None
        if losses is not None and self.hard_capacity > 0:
            if losses.ndim != 1 or losses.size(0) != B:
                raise ValueError(f"losses must be [B], got {tuple(losses.shape)}")
            loss = losses.detach()
            if loss.is_cuda:
                loss = loss.to("cpu", non_blocking=False)
            l_cpu = loss.to(torch.float32).contiguous()

        # --- main reservoir update ---
        if self.capacity > 0:
            for i in range(B):
                self._seen += 1
                if self._main_size < self.capacity:
                    self._main[self._main_size].copy_(x[i])
                    self._main_size += 1
                else:
                    j = int(
                        torch.randint(
                            low=0, high=self._seen, size=(1,), generator=self._g
                        ).item()
                    )
                    if j < self.capacity:
                        self._main[j].copy_(x[i])
        else:
            # still count as "seen" if you want, but it's not very meaningful w/ capacity==0
            self._seen += B

        # --- hard buffer update ---
        if l_cpu is not None and self.hard_capacity > 0:
            for i in range(B):
                self._hard_offer(x[i], float(l_cpu[i].item()))

    def _hard_offer(self, row: torch.Tensor, loss: float) -> None:
        if self.hard_capacity <= 0:
            return

        # Fill first
        if self._hard_size < self.hard_capacity:
            idx = self._hard_size
            self._hard[idx].copy_(row)
            heapq.heappush(self._hard_heap, (loss, self._hard_next_id, idx))
            self._hard_next_id += 1
            self._hard_size += 1
            return

        # Replace smallest-loss item if this is larger
        smallest_loss, _, idx = self._hard_heap[0]
        if loss <= smallest_loss:
            return

        heapq.heapreplace(self._hard_heap, (loss, self._hard_next_id, idx))
        self._hard_next_id += 1
        self._hard[idx].copy_(row)

    def reset(self, keep_main: bool = False, keep_hard: bool = False) -> None:
        """
        Flush replay contents. Optionally keep main/hard.
        """
        if not keep_main:
            self._main_size = 0
            self._seen = 0  # TODO: maybe keep this?
        if not keep_hard:
            self._hard_size = 0
            self._hard_heap.clear()
            self._hard_next_id = 0

    def can_sample(self, n: int) -> bool:
        n = int(n)
        return n > 0 and (self._main_size > 0 or self._hard_size > 0)

    def sample(
        self,
        n: int,
        *,
        device: Optional[torch.device] = None,
        non_blocking: bool = True,
    ) -> torch.Tensor:
        """
        Returns raw replay samples [n, D] (dtype).
        If device is provided, returns on that device (e.g. cuda).
        """
        n = int(n)
        dev = device if device is not None else torch.device("cpu")

        if n <= 0 or (self._main_size == 0 and self._hard_size == 0):
            return torch.empty((0, self.dim), dtype=self.dtype, device=dev)

        # Choose how many from hard vs main
        hard_n = 0
        if self._hard_size > 0 and self.hard_mix > 0.0:
            hard_n = int(round(n * self.hard_mix))
            hard_n = max(0, min(hard_n, n, self._hard_size))

        main_n = n - hard_n
        main_n = max(0, min(main_n, self._main_size))

        # If one side is empty, take from the other
        if main_n == 0 and self._hard_size > 0:
            hard_n = min(n, self._hard_size)
        if hard_n == 0 and self._main_size > 0:
            main_n = min(n, self._main_size)

        parts = []

        if main_n > 0 and self._main_size > 0:
            idx = torch.randint(0, self._main_size, (main_n,), generator=self._g)
            parts.append(self._main.index_select(0, idx))

        if hard_n > 0 and self._hard_size > 0:
            idx = torch.randint(0, self._hard_size, (hard_n,), generator=self._g)
            parts.append(self._hard.index_select(0, idx))

        out = torch.cat(parts, dim=0) if len(parts) > 1 else parts[0]

        # Optionally pin *only* when user requests GPU output (helps H2D copy)
        if self.pin_memory and device is not None and device.type == "cuda":
            out = out.pin_memory()

        if device is not None:
            out = out.to(device, non_blocking=non_blocking)

        return out
