# Shared settings and CLEAR's measurement loop, for the CLEAR skeleton adapted to Qwen3-8B (see run.sh).
# The loop is CLEAR's own (clear_repo/module_measure.py measure_energy): 0.5 s pause, energy counter read before and
# after N calls, no warm-up, no torch.cuda.synchronize(). The only addition is the wall time of each trial
# (read next to the counter), so that time and average power can be analysed; it does not change what is measured.
import os
import time

import torch
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetTotalEnergyConsumption
from tqdm import tqdm

model_name = os.environ.get("MODEL", "models/Qwen--Qwen3-8B")
device_index = int(os.environ.get("GPU", "0"))   # NVML index; run.sh also sets CUDA_VISIBLE_DEVICES to it
num_trials = int(os.environ.get("TRIALS", "20"))
repeat_parts = 10000                              # CLEAR: N = 10,000 for components
repeat_model = 1000                               # CLEAR: N = 1,000 for the full model
layer_to_capture = 5                              # as in CLEAR's BERT scripts
dtype = torch.float16                             # CLEAR's default precision (Qwen3's native dtype is BF16)
prompt = "The capital of France is" * 40          # CLEAR's prompt, repeated 40x (not 10x) so 128 tokens need no padding

nvmlInit()
handle = nvmlDeviceGetHandleByIndex(device_index)


def get_gpu_energy_mj():
    try:
        return nvmlDeviceGetTotalEnergyConsumption(handle)  # millijoules since boot
    except Exception as e:
        print(f"NVML Error: {e}")
        return 0


def measure_energy(fn, repeat_count, num_trials, label):
    if os.environ.get("TAIL") == "1":
        return measure_energy_tail(fn, repeat_count, num_trials, label)
    energy_usages, times = [], []
    for trial in tqdm(range(num_trials), desc=label):
        with torch.no_grad():
            time.sleep(0.5)
            start_energy = get_gpu_energy_mj(); t0 = time.time()
            for _ in range(repeat_count):
                _ = fn()
            end_energy = get_gpu_energy_mj(); t1 = time.time()
        energy_usages.append((end_energy - start_energy) / repeat_count)
        times.append(t1 - t0)
    vals = torch.tensor(energy_usages, dtype=torch.float64)
    return {
        "all_trials": [round(x, 6) for x in energy_usages],
        "average_mJ": round(vals.mean().item(), 6),
        "variance_mJ2": round(vals.var(unbiased=False).item(), 6),
        "std_dev_mJ": round(vals.std(unbiased=False).item(), 6),
        "repeat_count": repeat_count,
        "trial_seconds": [round(t, 4) for t in times],
    }


def tokenize(tokenizer, seq_len):
    tokens = tokenizer(prompt)["input_ids"][:seq_len]
    assert len(tokens) == seq_len, f"prompt too short for {seq_len} tokens"
    ids = torch.tensor([tokens], device="cuda")
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


# ------------------------------
# Run A (TAIL=1): CLEAR's loop unchanged, plus a record of the GPU work it never counts.
# CLEAR reads E_end as soon as the CPU has *queued* the N calls; without a sync the GPU may still be running up to
# ~1,000 queued operations. That tail runs into the 0.5 s pause before the next trial. The part that runs during a
# pause is counted by no trial; the part still running when the next trial reads E_start is counted by that trial.
# Additions, none of which wait on the GPU inside CLEAR's window:
#   - a CUDA event recorded before the first and after the last call (GPU time of the trial, and when it finished);
#   - the pause polls the previous trial's end event and reads the counter when the GPU finishes;
#   - clock, temperature and clock-limit reasons read right after E_end;
#   - after the last trial, a wait until the GPU is done (its tail is lost in CLEAR too).
# ------------------------------
from pynvml import (nvmlDeviceGetClockInfo, nvmlDeviceGetCurrentClocksEventReasons, nvmlDeviceGetTemperature,
                    NVML_CLOCK_SM, NVML_TEMPERATURE_GPU, nvmlClocksEventReasonSwPowerCap)


def _pause(sec, ev, e_end, t_end):
    """CLEAR's time.sleep(sec), polling (not waiting on) the previous trial's end event."""
    stop = time.time() + sec
    done = None
    while time.time() < stop:
        if done is None and ev is not None and ev.query():
            done = (time.time() - t_end, get_gpu_energy_mj() - e_end)
        time.sleep(0.0005)
    return done


def measure_energy_tail(fn, repeat_count, num_trials, label):
    rows = []
    prev = None
    for trial in tqdm(range(num_trials), desc=label):
        with torch.no_grad():
            done = _pause(0.5, prev and prev["ev1"], prev and prev["E_end"], prev and prev["t1"])
            start_energy = get_gpu_energy_mj(); t0 = time.time()
            ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            ev0.record()
            for _ in range(repeat_count):
                _ = fn()
            ev1.record()
            end_energy = get_gpu_energy_mj(); t1 = time.time()
        reasons = nvmlDeviceGetCurrentClocksEventReasons(handle)
        if prev is not None:  # what happened to the previous trial's tail during this pause
            prev["tail"] = _tail(done, prev, start_energy, t0)
        prev = {"E_start": start_energy, "E_end": end_energy, "t0": t0, "t1": t1, "ev0": ev0, "ev1": ev1,
                "sm_MHz": nvmlDeviceGetClockInfo(handle, NVML_CLOCK_SM),
                "temp_C": nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU),
                "power_cap": bool(reasons & nvmlClocksEventReasonSwPowerCap), "reasons": int(reasons)}
        rows.append(prev)
    torch.cuda.synchronize()  # after the last trial; CLEAR never counts this tail either
    s_last = time.time() - prev["t1"]
    prev["tail"] = {"done_within_pause": True, "s_after_E_end": s_last, "s_lost": s_last,
                    "mJ_lost": get_gpu_energy_mj() - prev["E_end"], "mJ_into_next": 0}
    out = []
    for r in rows:
        gpu_s = r["ev0"].elapsed_time(r["ev1"]) / 1000
        out.append({k: r[k] for k in ("E_start", "E_end", "t0", "t1", "sm_MHz", "temp_C", "power_cap", "reasons")}
                   | {"gpu_s": gpu_s, "cpu_s": r["t1"] - r["t0"], **{f"tail_{k}": v for k, v in r["tail"].items()}})
    counted = [(r["E_end"] - r["E_start"]) / repeat_count for r in out]
    lost = sum(r["tail_mJ_lost"] for r in out)
    vals = torch.tensor(counted, dtype=torch.float64)
    gpu_per_call = sum(r["gpu_s"] for r in out) / (num_trials * repeat_count)
    lost_s = sum(r["tail_s_lost"] for r in out)
    return {
        "all_trials": [round(x, 6) for x in counted],              # CLEAR's numbers, as their loop computes them
        "average_mJ": round(vals.mean().item(), 6),
        "variance_mJ2": round(vals.var(unbiased=False).item(), 6),
        "std_dev_mJ": round(vals.std(unbiased=False).item(), 6),
        "repeat_count": repeat_count,
        "trial_seconds": [round(r["cpu_s"], 4) for r in out],
        # Run A additions
        "average_with_tail_mJ": round((sum(counted) * repeat_count + lost) / (num_trials * repeat_count), 6),
        "lost_mJ_total": round(lost, 3),
        "lost_gpu_s_total": round(lost_s, 4),
        "gpu_s_per_call": gpu_per_call,
        "calls_lost_per_trial": round(lost_s / gpu_per_call / num_trials, 1) if gpu_per_call > 0 else None,
        "trials": out,
    }


def _tail(done, prev, next_start_energy, next_t0):
    """Split the previous trial's queued tail into the part lost in the pause and the part counted by the next trial."""
    if done is not None:  # GPU finished during the pause: everything after E_end until then is lost
        s, mj = done
        return {"done_within_pause": True, "s_after_E_end": s, "s_lost": s, "mJ_lost": mj, "mJ_into_next": 0}
    # still running at the next E_start: the whole pause is lost, the rest lands in the next trial
    return {"done_within_pause": False, "s_after_E_end": None, "s_lost": next_t0 - prev["t1"],
            "mJ_lost": next_start_energy - prev["E_end"], "mJ_into_next": None}
