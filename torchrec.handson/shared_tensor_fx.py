# https://github.com/pytorch/pytorch/issues/55207
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.fx import symbolic_trace, replace_pattern


def all_reduce(inp):
    dist.all_reduce(inp)
    return inp


def all_to_all(out, inp):
    dist.all_to_all_single(out, inp)
    return out


torch.fx.wrap("all_reduce")
torch.fx.wrap("all_to_all")


def print_with_rank(rank, msg):
    print("[RANK {}] {}".format(rank, msg))


def shard_model(model, rank, world_size):
    """
    Apply a sharding spec to the model.
    """
    inp = model.weight
    print(model)
    print(model.weight)
    output = torch.empty_like(inp.t().contiguous())
    all_to_all(output, inp.t().contiguous())

    # Weight consisting of shards from other processes.
    model.weight = torch.nn.Parameter(output.t())


def shard_and_apply_fx(model, rank, world_size, batch_size):
    shard_model(model, rank, world_size)

    traced_model = symbolic_trace(model)
    if rank == 0:
        print_with_rank(rank, "Trace Before: {}".format(traced_model))
    dist.barrier()

    traced_model = apply_fx(traced_model, rank, world_size, batch_size)

    dist.barrier()
    if rank == 0:
        print_with_rank(rank, "Trace After: {}".format(traced_model))

    return traced_model


def apply_fx(traced_model, rank, world_size, batch_size):

    shard_size = traced_model.weight.size(1) // world_size

    def pattern(input_1, weight, bias):
        return torch.nn.functional.linear(input_1, weight, bias=bias)

    def replacement(input_1, weight, bias):
        # Move inputs across processes to appropriate shards.
        inp = input_1.t().contiguous()
        gathered_input = torch.empty_like(inp)
        gathered_input = all_to_all(gathered_input, inp)
        gathered_input = gathered_input.t()

        # Hack to create empty tensor since torch.empty doesn't work.
        # https://github.com/pytorch/pytorch/issues/53935
        out = weight.as_strided((0,), (0,))
        for r in range(world_size):
            inp = torch.narrow(gathered_input, 1, r * shard_size, shard_size)
            w = torch.narrow(weight, 1, r * shard_size, shard_size)
            res = inp.matmul(w.t().contiguous())
            res = all_reduce(res)
            out = torch.cat((out, res))

        # Need to know batch size here since torch.fx doesn't support dynamic
        # non-Tensor parameters: https://github.com/pytorch/pytorch/issues/53937
        return torch.narrow(out, 0, rank * batch_size, batch_size) + bias

    replace_pattern(traced_model, pattern, replacement)

    # Don't forget to recompile!
    traced_model.recompile()

    return traced_model


def run_worker(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)

    # Different weights, inputs for each rank.
    torch.manual_seed(rank)
    batch_size = 100

    # Initialize model and record result without sharding.
    model = torch.nn.Linear(64, 24).cuda(rank)
    input = torch.rand(batch_size, 64).cuda(rank)
    result_without_sharding = model(input)

    # Shard the model, apply fx and execute the sharded model.
    # shard_and_apply_fx could be a standard PyTorch API like torch.shard_model()
    # which takes a model and a sharding spec and does all the necessary
    # sharding and fx rewrites for the user.
    traced_model = shard_and_apply_fx(model, rank, world_size, batch_size)
    result_with_sharding = traced_model(input)

    assert torch.allclose(result_with_sharding, result_without_sharding, atol=1e-05)
    print_with_rank(rank, "PASSED")


if __name__ == "__main__":
    world_size = 4
    mp.spawn(run_worker, args=(world_size,), nprocs=world_size, join=True)
