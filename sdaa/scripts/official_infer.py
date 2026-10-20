"""Official-inference check: recompute gen_desc on the exact SMILES behind
gen_desc/aryl_scope_cat_emb.npy and compare with the authors' precomputed output."""
import sys, time
import numpy as np
import torch
from catemb import CatEmb

device = sys.argv[1] if len(sys.argv) > 1 else "sdaa"
ref = np.load("gen_desc/aryl_scope_cat_emb.npy", allow_pickle=True).item()
smis = list(ref["smi"])
ref_desc = np.asarray(ref["desc"], dtype=np.float64)
print("n_smiles", len(smis), "ref_shape", ref_desc.shape)

if device == "sdaa":
    torch.sdaa.reset_peak_memory_stats()
elif device == "cuda":
    torch.cuda.reset_peak_memory_stats()
t0 = time.time()
calc = CatEmb(device=device)
our = calc.gen_desc(smis).astype(np.float64)
if device == "sdaa":
    torch.sdaa.synchronize()
elif device == "cuda":
    torch.cuda.synchronize()
print(f"gen_s={time.time()-t0:.2f}")
if device == "sdaa":
    print(f"peak_mem_gib={torch.sdaa.max_memory_allocated()/2**30:.3f}")
elif device == "cuda":
    print(f"peak_mem_gib={torch.cuda.max_memory_allocated()/2**30:.3f}")

d = np.abs(our - ref_desc)
print(f"max={d.max():.3e} mean={d.mean():.3e} p99={np.quantile(d,0.99):.3e}")
print(f"frac<1e-4={np.mean(d<1e-4):.4f} frac<1e-5={np.mean(d<1e-5):.4f}")
print("nan_our", np.isnan(our).sum(), "nan_ref", np.isnan(ref_desc).sum())
np.save(f"/tmp/catemb_aryl_{device}.npy", our)
print("OK", device)
