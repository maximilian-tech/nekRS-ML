#!/usr/bin/env python3
"""
PyTorch DataLoader Consumer (MPMD)

- Launch together with producers in a single MPMD `mpiexec` run.
- This script assumes it is the last rank in MPI.COMM_WORLD and thus the
  owner of shard 0. It wraps the ringbuffer as a torch IterableDataset and
  consumes tensors until the shard is drained.
"""

from mpi4py import MPI
import os
import sys
import copy
try:
    import torch
    from torch.utils.data import IterableDataset, DataLoader
    import torch.nn as nn
    import torch.optim as optim
except Exception as e:
    raise SystemExit(f"Please install torch: {e}")

try:
    import rdqpy
except Exception as e:
    raise SystemExit(
        f"Failed to import rdqpy. Ensure PYTHONPATH points to the built extension. Error: {e}"
    )

from replay import ReplayBuffer

REPLAY_ENABLE = True
REPLAY_CAPACITY = 200_000
REPLAY_HARD_CAPACITY = 50_000
REPLAY_HARD_MIX = 0.15
REPLAY_BATCH = 2048
REPLAY_WARMUP = 6_000
REPLAY_SEED = 54321
VALIDATION_SPLIT = 0.2
VALIDATION_MIN_SAMPLES = 1

def stream(ctx, shard, *, want=1, allow_partial=False, prefer_zerocopy=False):
    while True:
        try:
            yield ctx.recv_batch_dl(shard, want, allow_partial, prefer_zerocopy)
        except StopIteration:
            return

class DDQIterableDataset(IterableDataset):
    """Streams tensors from a DDQ shard. Use DataLoader(num_workers=0)."""
    def __init__(self, ctx, shard=0, *, want=1, allow_partial=False, prefer_zerocopy=False):
        super().__init__()
        self.ctx = ctx
        self.shard = shard
        self.want = want
        self.allow_partial = allow_partial
        self.prefer_zerocopy = prefer_zerocopy

    def __iter__(self):
        for prod in stream(self.ctx, self.shard,
                           want=self.want,
                           allow_partial=self.allow_partial,
                           prefer_zerocopy=self.prefer_zerocopy):
            #yield torch.from_dlpack(prod)  # `prod` is your DlpackProducer
            yield torch.from_dlpack(prod)  # `prod` is your DlpackProducer

class FCN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FCN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class DataBuffer:
    """Simple append-only buffer for accumulated training samples."""

    def __init__(self):
        self._chunks = []
        self._cached = None
        self._nrows = 0

    @property
    def nrows(self):
        return self._nrows

    def add(self, batch):
        if batch is None or batch.numel() == 0:
            return
        chunk = batch.detach()
        if chunk.is_cuda:
            chunk = chunk.to("cpu", non_blocking=False)
        chunk = chunk.contiguous()
        self._chunks.append(chunk)
        self._cached = None
        self._nrows += chunk.shape[0]

    def as_tensor(self):
        if self._cached is None:
            if not self._chunks:
                return None
            self._cached = torch.cat(self._chunks, dim=0)
        return self._cached


def split_train_validation(batch, validation_split):
    batch_size = batch.shape[0]
    if batch_size <= 1 or validation_split <= 0.0:
        return batch, None

    n_val = int(batch_size * validation_split)
    n_val = max(VALIDATION_MIN_SAMPLES, n_val)
    n_val = min(n_val, batch_size - 1)
    if n_val <= 0:
        return batch, None

    perm = torch.randperm(batch_size, device=batch.device)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    return batch[train_idx], batch[val_idx]


def regression_accuracy(prediction, target):
    target_mean = target.mean(dim=0, keepdim=True)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - prediction) ** 2)
    eps = torch.finfo(target.dtype).eps
    if ss_tot.abs() <= eps:
        return torch.tensor(
            1.0 if ss_res.abs() <= eps else 0.0,
            dtype=target.dtype,
            device=target.device,
        )
    return 1.0 - (ss_res / ss_tot)


def evaluate_buffer_loss(model, loss_fn, data_buffer, ndIn):
    buffer_tensor = data_buffer.as_tensor()
    if buffer_tensor is None or buffer_tensor.numel() == 0:
        return None

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    buffer_tensor = buffer_tensor.to(device=device, dtype=dtype, non_blocking=False)
    buffer_features = buffer_tensor[:, :ndIn]
    buffer_target = buffer_tensor[:, ndIn:]

    model.eval()
    with torch.no_grad():
        buffer_output = model.forward(buffer_features)
        buffer_loss = loss_fn(buffer_output, buffer_target).item()
    model.train()

    return buffer_loss


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    mpi_appnum = comm.Get_attr(MPI.APPNUM)
    if (mpi_appnum is None):
        print("MPI_APPNUM not available")
        comm.Abort(-1)
    ml_comm = comm.Split(color=mpi_appnum)
    if (ml_comm == MPI.COMM_NULL):
        print("ml_comm is NULL: Unexpected")
        comm.Abort(-2)
    # Treat this rank as the owner (consumer); producers should agree.
    owners = [rank]
    shard = 0
    
    ctx = rdqpy.Context(comm, nshards=1, owners=owners)
    ctx.set_log(level=2, categories=0xFFFF, json=False, color=False)
    intercomms, n_intercomm = rdqpy.create_intercomm(ml_comm)
    assert len(intercomms) == n_intercomm
    if n_intercomm != 1:
        raise RuntimeError(f"Expected exactly one intercommunicator, got {n_intercomm}")

    intercomm = intercomms[0]
    ch = rdqpy.GlobalFeedbackChannel(ml_comm, intercomm, is_sender_group=True, root_rank=0)
    # Wrap DDQ as a streaming dataset
    torch.manual_seed(0)
    dataset = DDQIterableDataset(ctx, shard=shard)
    # Use a single-worker DataLoader to avoid multiprocessing pickling issues
    loader = DataLoader(dataset, batch_size=1, num_workers=0)
    nNeurons = 20  # number of neuronsining settings
    ndIn = 1
    ndOut = 1
    
    learning_rate = 0.001  # learning rate

    model = FCN(input_size=ndIn, hidden_size=nNeurons, output_size=ndOut).to(torch.float64)
    loss_fn = nn.functional.mse_loss
    
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate * size, weight_decay=1e-3
    )

    replay = None
    data_buffer = DataBuffer()
    if REPLAY_ENABLE:
        replay = ReplayBuffer(
            dim=ndIn + ndOut,
            capacity=REPLAY_CAPACITY,
            hard_capacity=REPLAY_HARD_CAPACITY,
            hard_mix=REPLAY_HARD_MIX,
            pin_memory=True,
            seed=REPLAY_SEED + rank,
            dtype=torch.float64,
        )

    print("here=")
    count = 0
    running_train_loss = 0.0
    running_val_loss = 0.0
    running_val_mae = 0.0
    running_val_r2 = 0.0
    n_val_steps = 0
    for iteration, batch_l in enumerate(loader):
        # `batch` is a 1-element batch of tensors (shape: [1, rows, cols, ...])
        # Access the tensor content as batch[0]
        batch = batch_l[0]
        # Simple side-effect to confirm progress
        
        print(f"[consumer rank {rank}] received tensor {count}: shape={tuple(batch.shape)}, dtype={batch.dtype}",flush=True)
        #t = None
        
        #print(f"{batch.shape=}",flush=True)
        train_batch, val_batch = split_train_validation(batch, VALIDATION_SPLIT)
        cur_batch = train_batch
        B = cur_batch.shape[0]

        rep_n = min(REPLAY_BATCH, max(0, int(2 * B)))
        rep_batch = None
        if (
            REPLAY_ENABLE
            and replay is not None
            and replay.seen >= REPLAY_WARMUP
            and replay.can_sample(rep_n)
        ):
            print("Sampling from replay buffer")
            rep_batch = replay.sample(rep_n, device=cur_batch.device, non_blocking=True)

        if rep_batch is None or rep_batch.numel() == 0:
            mixed_batch = cur_batch
        else:
            print("Using replay buffer")
            mixed_batch = torch.cat([cur_batch, rep_batch], dim=0)

        features = mixed_batch[:, :ndIn]
        #print(f"{features.shape=}",flush=True)
        target = mixed_batch[:, ndIn:]
        #print(f"{target.shape=}",flush=True)
        
        optimizer.zero_grad()
        output = model.forward(features)
        #print(f"{output.shape=}",flush=True)

        cur_output = output[:B]
        cur_target = target[:B]
        cur_elem_loss = loss_fn(cur_output, cur_target, reduction="none")
        per_sample_loss_cur = cur_elem_loss.reshape(B, -1).mean(dim=1)
        if REPLAY_ENABLE and replay is not None:
            with torch.no_grad():
                replay.add(cur_batch, losses=per_sample_loss_cur.detach())

        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        train_loss = loss.item()
        running_train_loss += train_loss
        data_buffer.add(cur_batch)
        buffer_loss = evaluate_buffer_loss(model, loss_fn, data_buffer, ndIn)

        metrics = [
            f"iteration={iteration}",
            f"train_loss={train_loss:.6e}",
            f"train_loss_avg={running_train_loss / (iteration + 1):.6e}",
        ]

        if buffer_loss is not None:
            metrics.extend(
                [
                    f"buffer_loss={buffer_loss:.6e}",
                    f"buffer_rows={data_buffer.nrows}",
                ]
            )

        if val_batch is not None:
            val_features = val_batch[:, :ndIn]
            val_target = val_batch[:, ndIn:]
            model.eval()
            with torch.no_grad():
                val_output = model.forward(val_features)
                val_loss = loss_fn(val_output, val_target).item()
                val_mae = torch.mean(torch.abs(val_output - val_target)).item()
                val_r2 = regression_accuracy(val_output, val_target).item()
            model.train()

            running_val_loss += val_loss
            running_val_mae += val_mae
            running_val_r2 += val_r2
            n_val_steps += 1

            metrics.extend(
                [
                    f"val_loss={val_loss:.6e}",
                    f"val_mae={val_mae:.6e}",
                    f"val_r2={val_r2:.6e}",
                    f"val_loss_avg={running_val_loss / n_val_steps:.6e}",
                    f"val_mae_avg={running_val_mae / n_val_steps:.6e}",
                    f"val_r2_avg={running_val_r2 / n_val_steps:.6e}",
                ]
            )

        print(" ".join(metrics))

        count += 1
        if False:
        #if loss.item() < 1e-4:
            done = ch.send_progress(
                {
                    "seq": 1,
                    "epoch": iteration,
                    "command": 1,
                    "payload_kind": 0,
                    "source_rank": 0,
                    "target_group": 7,
                }
            )
            print("!!!!!!!!!! MESSAGE POSTED !!!!!!!!!!!!!!!")
            if done:
                print("!!!!!!!!!! MESSAGE RECIEVED !!!!!!!!!!!!!!!")


    print(f"[consumer rank {rank}] drained shard {shard}; total tensors: {count}")
    ch.close()
    ml_comm.Free()
    ctx.close()


if __name__ == "__main__":
    main()
