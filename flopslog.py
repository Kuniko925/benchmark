from pathlib import Path
import torch
import time
from flops_profiler.profiler import FlopsProfiler

def profile_forward_flops(
    model,
    tokenizer,
    output_path: Path,
    device: torch.device,
    sequence_length: int = 512,
) -> dict:
    sample_text = (
        "Below is an instruction that describes a task.\n\n"
        "### Instruction:\n"
        "Explain why cooling is important for GPU training.\n\n"
        "### Response:\n"
    )

    encoded = tokenizer(
        sample_text,
        return_tensors="pt",
        truncation=True,
        max_length=sequence_length,
        padding="max_length",
    )

    inputs = {
        key: value.to(device)
        for key, value in encoded.items()
    }

    previous_training_state = model.training
    model.eval()

    with torch.no_grad():
        for _ in range(3):
            model(
                **inputs,
                use_cache=False,
            )

    torch.cuda.synchronize()

    profiler = FlopsProfiler(model)
    profiler.start_profile()

    start = time.perf_counter()

    try:
        with torch.no_grad():
            model(
                **inputs,
                use_cache=False,
            )

        torch.cuda.synchronize()

        elapsed_sec = time.perf_counter() - start

        flops = profiler.get_total_flops(
            as_string=False
        )
        macs = profiler.get_total_macs(
            as_string=False
        )
        params = profiler.get_total_params(
            as_string=False
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        profiler.print_model_profile(
            profile_step=0,
            module_depth=3,
            top_modules=10,
            detailed=True,
            output_file=str(output_path),
        )

    finally:
        profiler.stop_profile()
        profiler.end_profile()

        if previous_training_state:
            model.train()

    flops_per_sec = (
        flops / elapsed_sec
        if elapsed_sec > 0
        else 0.0
    )

    print(f"Forward FLOPs: {flops:,.0f}")
    print(f"Forward MACs: {macs:,.0f}")
    print(f"Parameters: {params:,.0f}")
    print(f"Forward latency: {elapsed_sec:.6f} sec")
    print(f"Estimated FLOP/s: {flops_per_sec:,.0f}")
    print(f"Profile saved to: {output_path.resolve()}")

    return {
        "forward_flops": float(flops),
        "forward_macs": float(macs),
        "parameters": float(params),
        "forward_latency_sec": elapsed_sec,
        "estimated_flops_per_sec": flops_per_sec,
    }
