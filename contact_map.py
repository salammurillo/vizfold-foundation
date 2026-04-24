import numpy as np
import matplotlib.pyplot as plt
import torch
from Bio import PDB
import sys

sys.path.insert(0, "/storage/ice1/6/0/salammurillo3/openfold")

from openfold.config import model_config
from openfold.model.model import AlphaFold
from openfold.utils.import_weights import import_jax_weights_
from openfold.data.data_pipeline import DataPipeline
from openfold.data.feature_pipeline import FeaturePipeline
from openfold.model.triangular_attention import TriangleAttention
from openfold.model.primitives import Attention, softmax_no_cast, permute_final_dims
import openfold.model.primitives as primitives

# CONFIG
LAYER_START = 38
LAYER_END   = 47

FASTA_PATH    = "/home/hice1/salammurillo3/scratch/openfold/examples/monomer/fasta_dir/6kwc.fasta"
ALIGNMENT_DIR = "/home/hice1/salammurillo3/scratch/openfold/examples/monomer/alignments/6KWC_1"
PDB_PATH      = "/home/hice1/salammurillo3/scratch/openfold/outputs/predictions/6KWC_1_model_1_ptm_relaxed.pdb"
PARAMS_PATH   = "openfold/resources/params/params_model_1_ptm.npz"


# 1 Contact map from PDB
def get_distance_matrix(pdb_path, chain_id="A"):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    coords = []
    for model_ in structure:
        for chain in model_:
            if chain.id != chain_id:
                continue
            for residue in chain:
                if PDB.is_aa(residue, standard=True) and "CA" in residue:
                    coords.append(residue["CA"].get_vector().get_array())
    coords = np.array(coords)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))

dist_matrix = get_distance_matrix(PDB_PATH)
print(f"Protein length: {len(dist_matrix)} residues")


# 2 Patch _attention to capture softmax weights 
_capture = False
_attn_weights_store = []

_orig_attention = primitives._attention

def _patched_attention(query, key, value, biases):
    key_t = permute_final_dims(key, (1, 0))
    a = torch.matmul(query, key_t)
    for b in biases:
        a += b
    a = softmax_no_cast(a, -1)
    if _capture:
        _attn_weights_store.append(a.detach().cpu())
    return torch.matmul(a, value)

primitives._attention = _patched_attention
Attention.forward.__globals__["_attention"] = _patched_attention

print(f"Patch live in primitives: {primitives._attention is _patched_attention}")
print(f"Patch live in Attention.forward globals: {Attention.forward.__globals__['_attention'] is _patched_attention}")


# 3 Load model
config = model_config("model_1_ptm")
model  = AlphaFold(config).eval().cuda()
import_jax_weights_(model, PARAMS_PATH, version="model_1_ptm")
print("Model loaded.")


# 4 Register pre/post hooks to toggle capture for target layers only
def get_layer_idx(name):
    parts = name.split(".")
    for i, p in enumerate(parts):
        if p == "blocks":
            try:
                return int(parts[i + 1])
            except (ValueError, IndexError):
                pass
    return -1

hooks = []

for name, module in model.named_modules():
    if not isinstance(module, TriangleAttention):
        continue
    if not name.startswith("evoformer"):
        continue
    layer_idx = get_layer_idx(name)
    if not (LAYER_START <= layer_idx <= LAYER_END):
        continue

    def make_hooks(n):
        def pre_hook(module, input):
            global _capture
            _capture = True
        def post_hook(module, input, output):
            global _capture
            _capture = False
        return pre_hook, post_hook

    pre, post = make_hooks(name)
    hooks.append(module.register_forward_pre_hook(pre))
    hooks.append(module.register_forward_hook(post))

print(f"Hooks registered on {len(hooks) // 2} triangle attention modules (layers {LAYER_START}–{LAYER_END}).")


# 5 Run inference
data_proc    = DataPipeline(template_featurizer=None)
feature_proc = FeaturePipeline(config.data)
features     = data_proc.process_fasta(fasta_path=FASTA_PATH, alignment_dir=ALIGNMENT_DIR)
processed    = feature_proc.process_features(features, mode="predict")
batch        = {k: v.unsqueeze(0).cuda() if isinstance(v, torch.Tensor) else v
                for k, v in processed.items()}

print("Running inference...")
with torch.no_grad():
    model(batch)

for h in hooks:
    h.remove()

primitives._attention = _orig_attention
Attention.forward.__globals__["_attention"] = _orig_attention

print(f"Captured {len(_attn_weights_store)} attention weight tensors.")
if _attn_weights_store:
    print(f"Sample shape: {_attn_weights_store[0].shape}")


# 6 Aggregate: mean over heads and recycles, then over layers
if not _attn_weights_store:
    raise RuntimeError("No attention weights captured.")

N = dist_matrix.shape[0]
tensors = []
for t in _attn_weights_store:
    # Shape: (1, N_res, H, N_res, N_res) — squeeze batch, then mean over everything
    # except the last two (N, N) dims
    t = t.squeeze(0)                         # drop batch dim
    shape = t.shape
    print(f"  Processing tensor shape: {list(shape)}")
    if shape[-1] == N and shape[-2] == N:
        # Flatten all leading dims and mean → (N, N)
        t = t.reshape(-1, N, N).float().mean(dim=0).numpy()
    else:
        raise ValueError(f"Unexpected shape {shape}, expected last two dims to be {N}")
    tensors.append(t)

attn_map = np.stack(tensors).mean(axis=0)
attn_map = (attn_map + attn_map.T) / 2
print(f"Final attention map shape: {attn_map.shape}")


# 7 Plot
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Distance map
im1 = axes[0].imshow(dist_matrix, cmap="viridis_r", origin="lower")
axes[0].set_title("Cα Distance Map")
axes[0].set_xlabel("Residue index")
axes[0].set_ylabel("Residue index")
plt.colorbar(im1, ax=axes[0], label="Distance (Å)")

# Binary contact map
axes[1].imshow(dist_matrix < 8.0, cmap="Greys", origin="lower")
axes[1].set_title("Contact Map (Cα < 8Å)")
axes[1].set_xlabel("Residue index")
axes[1].set_ylabel("Residue index")

# True attention weights overlaid on contact map
contacts_i, contacts_j = np.where(dist_matrix < 8.0)
im3 = axes[2].imshow(attn_map, cmap="hot_r", origin="lower")
axes[2].scatter(contacts_j, contacts_i, s=0.3, c="cyan", alpha=0.5, label="Cα < 8Å")
axes[2].set_title(f"Triangle Attention Weights (layers {LAYER_START}–{LAYER_END})\nMean over heads + recycles + layers")
axes[2].set_xlabel("Residue index")
axes[2].set_ylabel("Residue index")
plt.colorbar(im3, ax=axes[2], label="Attention weight")
axes[2].legend(markerscale=10, fontsize=8)

plt.suptitle("6KWC_1 — Contact Map + True Triangle Attention Weights", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"contact_attention_{LAYER_START}-{LAYER_END}.png", dpi=150, bbox_inches="tight")
print(f"Saved to contact_attention_{LAYER_START}-{LAYER_END}.png")
plt.show()

 