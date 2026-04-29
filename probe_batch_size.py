"""Empirically find the largest batch size that fits in VRAM.

For each configuration of (precision, grad_checkpointing, intermediate_size),
ascends through candidate batch sizes, building model + Adam optimizer,
running one forward + backward + step at seq_len=512, and reporting
peak VRAM usage. Stops a config at the first OOM.
"""

import gc
import torch
import transformers


def try_batch(intermediate_size, batch, dtype, grad_ckpt, seq_len=512):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    config = transformers.GPTNeoXConfig(
        vocab_size=5,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=intermediate_size,
        max_position_embeddings=seq_len,
    )
    model = transformers.GPTNeoXForCausalLM(config).cuda()
    if grad_ckpt:
        model.gradient_checkpointing_enable()
    optim = torch.optim.AdamW(model.parameters(), lr=3e-4)
    input_ids = torch.randint(0, 5, (batch, seq_len), device="cuda")

    if dtype == torch.float32:
        out = model(input_ids=input_ids, labels=input_ids)
        out.loss.backward()
    else:
        with torch.amp.autocast("cuda", dtype=dtype):
            out = model(input_ids=input_ids, labels=input_ids)
        out.loss.backward()
    optim.step()

    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    del model, optim, out, input_ids
    gc.collect()
    torch.cuda.empty_cache()
    return peak_gb


def sweep(name, intermediate_size, dtype, grad_ckpt, candidates):
    print(f"\n=== {name} ===")
    last_ok = None
    for b in candidates:
        try:
            peak = try_batch(intermediate_size, b, dtype, grad_ckpt)
            print(f"  batch={b:4d}: OK   peak={peak:.2f} GB")
            last_ok = (b, peak)
        except torch.cuda.OutOfMemoryError as e:
            print(f"  batch={b:4d}: OOM ({str(e).splitlines()[0][:80]})")
            break
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  batch={b:4d}: OOM")
                break
            raise
    if last_ok is not None:
        b, peak = last_ok
        print(f"  → max OK at batch={b} (peak {peak:.2f} GB)")
    return last_ok


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}  total={torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Standard candidate ladder.
    ladder = [9, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]

    # Current model: 481M params (intermediate_size=24576).
    sweep("current model (intermediate_size=24576) | fp32",     24576, torch.float32, False, ladder)
    sweep("current model (intermediate_size=24576) | bf16",     24576, torch.bfloat16, False, ladder)
    sweep("current model (intermediate_size=24576) | fp32 + grad_ckpt", 24576, torch.float32, True, ladder)
    sweep("current model (intermediate_size=24576) | bf16 + grad_ckpt", 24576, torch.bfloat16, True, ladder)

    # Corrected model: 85M params (intermediate_size=3072).
    sweep("corrected model (intermediate_size=3072) | fp32",     3072, torch.float32, False, ladder)
    sweep("corrected model (intermediate_size=3072) | bf16",     3072, torch.bfloat16, False, ladder)
    sweep("corrected model (intermediate_size=3072) | bf16 + grad_ckpt", 3072, torch.bfloat16, True, ladder)


if __name__ == "__main__":
    main()
