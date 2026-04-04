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
    ctx.set_log(level=4, categories=0xFFFF, json=False, color=False)
    intercomms, n_intercomm = rdqpy.create_intercomm(ml_comm)
    assert len(intercomms) == n_intercomm
    if n_intercomm != 1:
        raise RuntimeError(f"Expected exactly one intercommunicator, got {n_intercomm}")

    intercomm = intercomms[0]
    print("here=")
    ch = rdqpy.GlobalFeedbackChannel(ml_comm, intercomm, is_sender_group=True, root_rank=0)
    print("here=")
    # Wrap DDQ as a streaming dataset
    dataset = DDQIterableDataset(ctx, shard=shard)
    print("here=")
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
    print("here=")
    count = 0
    for iteration, batch_l in enumerate(loader):
        # `batch` is a 1-element batch of tensors (shape: [1, rows, cols, ...])
        # Access the tensor content as batch[0]
        batch = batch_l[0]
        # Simple side-effect to confirm progress
        
        print(f"[consumer rank {rank}] received tensor {count}: shape={tuple(batch.shape)}, dtype={batch.dtype}",flush=True)
        #t = None
        
        #print(f"{batch.shape=}",flush=True)
        features = batch[:, :ndIn]
        #print(f"{features.shape=}",flush=True)
        target = batch[:, ndIn:]
        #print(f"{target.shape=}",flush=True)
        
        optimizer.zero_grad()
        output = model.forward(features)
        #print(f"{output.shape=}",flush=True)
        
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        print(f"{iteration=} {loss.item()=}")

        count += 1
        #if False:
        if loss.item() < 1e-4:
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

