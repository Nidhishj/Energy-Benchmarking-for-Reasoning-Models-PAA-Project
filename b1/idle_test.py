"""Idle / awake-power test for B1 (BERT-base, one block, 8 tokens). Results: findings_20261027.md, section 5.

  (1) Per-call energy of one block for N = 300 ... 30,000: does more repetition shrink anything?
  (2) Power while the GPU is awake but has almost nothing to do (tiny kernel back to back / every ~300 us),
      compared with idle (asleep) before and after.

  CUDA_VISIBLE_DEVICES=<free gpu> python b1/idle_test.py      # ~3 min
"""
import sys, time, torch
sys.path.insert(0, ".")
from energy import GpuMeter
from transformers import AutoModel
m = GpuMeter(0)

def window(fn, n):
    m.check_free(); e0, t0 = m.read(); 
    for _ in range(n): fn()
    e1, t1 = m.read(); return (e1 - e0), (t1 - t0)

def idle(sec, label):
    e0, t0 = m.read(sync=False); time.sleep(sec); e1, t1 = m.read(sync=False)
    print(f"{label:38s} {(e1-e0)/1000/(t1-t0):6.1f} W   state {m.state()}")

idle(15, "idle before loading anything")
bert = AutoModel.from_pretrained("models/google-bert--bert-base-uncased", dtype=torch.float16, attn_implementation="sdpa").cuda().eval()
layer = bert.encoder.layer[5]
x = torch.randn(1, 8, 768, device="cuda", dtype=torch.float16)
with torch.no_grad():
    for _ in range(2000): layer(x)
    print("\n(1) block, 8 tokens: per-call energy vs N")
    for n in (300, 1000, 3000, 10000, 30000):
        r = []
        for _ in range(3):
            time.sleep(0.5); e, t = window(lambda: layer(x), n); r.append((e / n, e / 1000 / t, t))
        print(f"  N={n:6d}  window {r[0][2]:6.2f} s   {sum(a for a,_,_ in r)/3:6.2f} mJ/call   {sum(b for _,b,_ in r)/3:5.1f} W")

    print("\n(2) GPU switched on but (almost) nothing to compute, 3 s each")
    a = torch.zeros(1, device="cuda")
    e, t = window(lambda: a.add_(1), 60000)
    print(f"  tiny kernel back-to-back       {e/1000/t:6.1f} W  ({t/60000*1e6:.1f} us per launch)")
    def spaced():
        a.add_(1); t1 = time.perf_counter() + 300e-6
        while time.perf_counter() < t1: pass
    e, t = window(spaced, 10000)
    print(f"  tiny kernel every ~300 us      {e/1000/t:6.1f} W")
idle(15, "\nidle right after the work")
idle(30, "idle 15-45 s after the work")
