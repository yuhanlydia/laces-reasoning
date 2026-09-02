from __future__ import annotations

# pyright: reportAny=false, reportArgumentType=false, reportInvalidCast=false, reportUnknownMemberType=false, reportUnknownVariableType=false

import argparse
import datetime
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_simple import DirectFileDataset
from models.state_prefill_block_diffusion import RWKVLike, StatePrefillBlockDiffusion
from utils import parse_dtype


class SaveableRWKV(RWKVLike, Protocol):
    def to(self, device: torch.device) -> nn.Module: ...

    def train(self, mode: bool = True) -> nn.Module: ...

    def save_pretrained(self, save_directory: str | Path) -> None: ...


class SaveableTokenizer(Protocol):
    def __len__(self) -> int: ...

    def save_pretrained(self, save_directory: str | Path) -> None: ...


@dataclass(frozen=True)
class DistributedContext:
    local_rank: int
    global_rank: int
    world_size: int
    is_main_process: bool
    is_distributed: bool


@dataclass(frozen=True)
class TrainArgs:
    rwkv_path: str
    token_dir: str
    save_dir: str
    batch_size: int
    max_steps: int
    block_size: int
    lr: float
    weight_decay: float
    dtype: str
    device: str
    max_samples: int | None
    num_workers: int
    grad_accum_steps: int
    mask_id: int | None
    pad_id: int | None
    eos_id: int
    min_mask_ratio: float
    max_mask_ratio: float
    full_mask_prob: float
    lambda_state: float
    lambda_logit: float
    lambda_ar: float
    state_mask_ratio_weight: float
    freeze_rwkv: bool
    save_every: int
    log_every: int
    micro_log_every: int


def init_distributed() -> DistributedContext:
    env_has_ddp = all(key in os.environ for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
    if not env_has_ddp:
        return DistributedContext(
            local_rank=0,
            global_rank=0,
            world_size=1,
            is_main_process=True,
            is_distributed=False,
        )

    local_rank = int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=datetime.timedelta(minutes=60),
            device_id=torch.device("cuda", local_rank),
        )
    else:
        dist.init_process_group(
            backend="gloo",
            init_method="env://",
            timeout=datetime.timedelta(minutes=60),
        )
    return DistributedContext(
        local_rank=local_rank,
        global_rank=dist.get_rank(),
        world_size=dist.get_world_size(),
        is_main_process=dist.get_rank() == 0,
        is_distributed=True,
    )


def model_vocab_size(rwkv: object, tokenizer: SaveableTokenizer) -> int:
    config = getattr(rwkv, "config", None)
    vocab_size = getattr(config, "vocab_size", None)
    if isinstance(vocab_size, int) and vocab_size > 0:
        return vocab_size
    return len(tokenizer)


def default_mask_id(rwkv: object, tokenizer: SaveableTokenizer) -> int:
    vocab_size = model_vocab_size(rwkv, tokenizer)
    if vocab_size > len(tokenizer):
        return vocab_size - 1
    return len(tokenizer) - 1


def default_pad_id(tokenizer: SaveableTokenizer) -> int | None:
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if isinstance(pad_token_id, int):
        return pad_token_id
    return None


def parse_args() -> TrainArgs:
    parser = argparse.ArgumentParser()
    def add_arg(*name_or_flags: str, **kwargs: object) -> None:
        _ = parser.add_argument(*name_or_flags, **kwargs)

    add_arg("--rwkv_path", required=True, help="Local HF RWKV checkpoint directory")
    add_arg("--token_dir", required=True, help="Directory containing token .npz/.pkl shards")
    add_arg("--save_dir", default="outputs_state_prefill_block", help="Checkpoint output directory")
    add_arg("--batch_size", type=int, default=1)
    add_arg("--max_steps", type=int, default=1000)
    add_arg("--block_size", type=int, default=32)
    add_arg("--lr", type=float, default=1e-5)
    add_arg("--weight_decay", type=float, default=0.0)
    add_arg("--dtype", default="bf16", choices=["fp32", "fp16", "bf16"])
    add_arg("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    add_arg("--max_samples", type=int, default=None)
    add_arg("--num_workers", type=int, default=0)
    add_arg("--grad_accum_steps", type=int, default=1, help="Optimizer gradient accumulation steps")
    add_arg("--mask_id", type=int, default=None, help="Default: model vocab_size - 1 when the RWKV vocab has unused tail ids")
    add_arg("--pad_id", type=int, default=None, help="Default: tokenizer pad_token_id when available")
    add_arg("--eos_id", type=int, default=0)
    add_arg("--min_mask_ratio", type=float, default=0.0)
    add_arg("--max_mask_ratio", type=float, default=1.0)
    add_arg("--full_mask_prob", type=float, default=0.10)
    add_arg("--lambda_state", type=float, default=0.0, help="Weight for masked-state to clean-state distillation")
    add_arg("--lambda_logit", type=float, default=0.0, help="Weight for masked-logit to clean-logit distillation")
    add_arg("--lambda_ar", type=float, default=0.0, help="Weight for clean next-token CE preservation")
    add_arg("--state_mask_ratio_weight", type=float, default=0.0, help="State loss multiplier uses 1 + weight * block_mask_ratio")
    add_arg("--freeze_rwkv", action="store_true", help="Debug only: freeze all RWKV params")
    add_arg("--save_every", type=int, default=0, help="0 disables intermediate saves")
    add_arg("--log_every", type=int, default=10)
    add_arg("--micro_log_every", type=int, default=0, help="0 disables gradient-accumulation heartbeat logs")
    ns = parser.parse_args()
    return TrainArgs(
        rwkv_path=str(ns.rwkv_path),
        token_dir=str(ns.token_dir),
        save_dir=str(ns.save_dir),
        batch_size=int(ns.batch_size),
        max_steps=int(ns.max_steps),
        block_size=int(ns.block_size),
        lr=float(ns.lr),
        weight_decay=float(ns.weight_decay),
        dtype=str(ns.dtype),
        device=str(ns.device),
        max_samples=None if ns.max_samples is None else int(ns.max_samples),
        num_workers=int(ns.num_workers),
        grad_accum_steps=int(ns.grad_accum_steps),
        mask_id=None if ns.mask_id is None else int(ns.mask_id),
        pad_id=None if ns.pad_id is None else int(ns.pad_id),
        eos_id=int(ns.eos_id),
        min_mask_ratio=float(ns.min_mask_ratio),
        max_mask_ratio=float(ns.max_mask_ratio),
        full_mask_prob=float(ns.full_mask_prob),
        lambda_state=float(ns.lambda_state),
        lambda_logit=float(ns.lambda_logit),
        lambda_ar=float(ns.lambda_ar),
        state_mask_ratio_weight=float(ns.state_mask_ratio_weight),
        freeze_rwkv=bool(ns.freeze_rwkv),
        save_every=int(ns.save_every),
        log_every=int(ns.log_every),
        micro_log_every=int(ns.micro_log_every),
    )


def main() -> None:
    args = parse_args()
    if args.grad_accum_steps < 1:
        raise ValueError(f"grad_accum_steps must be >= 1, got {args.grad_accum_steps}")
    if args.micro_log_every < 0:
        raise ValueError(f"micro_log_every must be >= 0, got {args.micro_log_every}")
    ddp = init_distributed()
    device = (
        torch.device("cuda", ddp.local_rank)
        if ddp.is_distributed and torch.cuda.is_available()
        else torch.device(args.device)
    )
    dtype = parse_dtype(args.dtype)
    if ddp.is_main_process:
        ddp_message = f"DDP: rank={ddp.global_rank} world_size={ddp.world_size} local_rank={ddp.local_rank} device={device}"
        print(ddp_message, flush=True)

    tokenizer = cast(
        SaveableTokenizer,
        AutoTokenizer.from_pretrained(args.rwkv_path, trust_remote_code=True, local_files_only=True),
    )
    rwkv = cast(
        SaveableRWKV,
        cast(
            object,
            AutoModelForCausalLM.from_pretrained(
                args.rwkv_path,
                trust_remote_code=True,
                local_files_only=True,
                torch_dtype=dtype,
            ),
        ),
    )
    _ = rwkv.to(device)
    _ = rwkv.train()

    vocab_size = model_vocab_size(rwkv, tokenizer)
    mask_id = default_mask_id(rwkv, tokenizer) if args.mask_id is None else args.mask_id
    pad_id = default_pad_id(tokenizer) if args.pad_id is None else args.pad_id
    if not 0 <= mask_id < vocab_size:
        raise ValueError(f"mask_id must be in [0, {vocab_size}), got {mask_id}")
    if pad_id is not None and not 0 <= pad_id < vocab_size:
        raise ValueError(f"pad_id must be in [0, {vocab_size}), got {pad_id}")
    if ddp.is_main_process:
        id_message = f"tokenizer_len={len(tokenizer)} model_vocab_size={vocab_size} mask_id={mask_id} pad_id={pad_id}"
        print(id_message, flush=True)
    model = StatePrefillBlockDiffusion(
        rwkv_model=rwkv,
        mask_id=mask_id,
        block_size=args.block_size,
        pad_id=pad_id,
        eos_id=args.eos_id,
        min_mask_ratio=args.min_mask_ratio,
        max_mask_ratio=args.max_mask_ratio,
        full_mask_prob=args.full_mask_prob,
        lambda_state=args.lambda_state,
        lambda_logit=args.lambda_logit,
        lambda_ar=args.lambda_ar,
        state_mask_ratio_weight=args.state_mask_ratio_weight,
        freeze_rwkv=args.freeze_rwkv,
    ).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters. Remove --freeze_rwkv for real training.")
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    ddp_model = (
        DDP(
            model,
            device_ids=[ddp.local_rank] if device.type == "cuda" else None,
            output_device=ddp.local_rank if device.type == "cuda" else None,
            find_unused_parameters=False,
            static_graph=True,
            gradient_as_bucket_view=True,
        )
        if ddp.is_distributed
        else model
    )

    dataset = DirectFileDataset(
        token_dir=args.token_dir,
        latent_dir="",
        max_samples=args.max_samples,
        use_external_latents=False,
        verbose=ddp.is_main_process,
    )
    sampler = DistributedSampler(
        dataset,
        num_replicas=ddp.world_size,
        rank=ddp.global_rank,
        shuffle=True,
        drop_last=True,
    ) if ddp.is_distributed else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    if len(loader) == 0:
        raise RuntimeError("No batches available; lower batch_size or check token_dir")
    if ddp.is_distributed and len(dataset) < args.batch_size * ddp.world_size:
        raise RuntimeError("Dataset is smaller than global batch; lower batch_size or add more samples")

    save_dir = Path(args.save_dir)
    if ddp.is_main_process:
        save_dir.mkdir(parents=True, exist_ok=True)
        effective_batch = args.batch_size * ddp.world_size * args.grad_accum_steps
        batch_message = (
            f"effective_batch={effective_batch} grad_accum_steps={args.grad_accum_steps} "
            f"log_every={args.log_every} micro_log_every={args.micro_log_every}"
        )
        print(batch_message, flush=True)
    if ddp.is_distributed:
        _ = dist.barrier()

    step = 0
    epoch = 0
    accum_count = 0
    accumulated_loss = torch.zeros((), device=device)
    accumulated_tokens = torch.zeros((), device=device)
    optimizer.zero_grad(set_to_none=True)
    while step < args.max_steps:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            tokens = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device).bool()

            out = ddp_model(tokens, attention_mask=attention_mask)
            loss = out.loss / args.grad_accum_steps
            _ = loss.backward()
            accumulated_loss = accumulated_loss + out.loss.detach()
            accumulated_tokens = accumulated_tokens + out.n_loss_tokens.detach()
            accum_count += 1
            if (
                ddp.is_main_process
                and args.micro_log_every > 0
                and accum_count % args.micro_log_every == 0
                and accum_count < args.grad_accum_steps
            ):
                micro_loss = accumulated_loss / accum_count
                micro_message = (
                    f"micro_step={accum_count}/{args.grad_accum_steps} "
                    f"next_optimizer_step={step + 1} loss_so_far={micro_loss.item():.4f} "
                    f"masked_tokens_so_far={int(accumulated_tokens.item())}"
                )
                print(micro_message, flush=True)
            if accum_count < args.grad_accum_steps:
                continue

            step += 1
            _ = torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            _ = optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            should_log = step == 1 or step % args.log_every == 0
            if should_log:
                log_loss = accumulated_loss / args.grad_accum_steps
                log_tokens = accumulated_tokens
                if ddp.is_distributed:
                    log_loss = log_loss.clone()
                    log_tokens = log_tokens.clone()
                    _ = dist.all_reduce(log_loss, op=dist.ReduceOp.AVG)
                    _ = dist.all_reduce(log_tokens, op=dist.ReduceOp.SUM)
                if ddp.is_main_process:
                    log_message = f"step={step} loss={log_loss.item():.4f} masked_tokens={int(log_tokens.item())}"
                    if args.lambda_state > 0.0 or args.lambda_logit > 0.0 or args.lambda_ar > 0.0:
                        log_message = (
                            f"{log_message} denoise={out.denoise_loss.item():.4f} "
                            f"state={out.state_loss.item():.4f} logit={out.logit_loss.item():.4f} "
                            f"ar={out.ar_loss.item():.4f}"
                        )
                    print(log_message, flush=True)
            accum_count = 0
            accumulated_loss = torch.zeros((), device=device)
            accumulated_tokens = torch.zeros((), device=device)
            if args.save_every > 0 and step % args.save_every == 0:
                ckpt_dir = save_dir / f"step_{step:08d}"
                if ddp.is_main_process:
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    rwkv.save_pretrained(ckpt_dir)
                    tokenizer.save_pretrained(ckpt_dir)
                if ddp.is_distributed:
                    _ = dist.barrier()
            if step >= args.max_steps:
                break
        epoch += 1

    final_dir = save_dir / "final"
    if ddp.is_main_process:
        final_dir.mkdir(parents=True, exist_ok=True)
        rwkv.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        print(f"saved final checkpoint to {final_dir}")
    if ddp.is_distributed:
        _ = dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
