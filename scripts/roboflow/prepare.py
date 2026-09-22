import os, json, re, sys, random, shutil
from collections import defaultdict, Counter
import numpy as np
from PIL import Image

NAMES = ["plastic","paper","tissue","bottle","cup","can","cardboard","organic","wood"]
IDX = {n:i for i,n in enumerate(NAMES)}

def norm(s): return re.sub(r"[^a-z0-9]", "", s.lower())

CANON_RAW = {
 "plastic":   ["Plastic Wrapper","Plastic-Wrapper","PlasticWrapper","Plastic Bag","Plastic-Bag",
               "Styrofoam","Face Mask","Face-Mask"],
 "paper":     ["Paper Bag","PaperBag"],
 "cardboard": ["Cardboard","Paperboard","Tetra Pak","Tetra-Pak","TetraPak","Tetra Pack"],
 "bottle":    ["Plastic Bottle","Plastic-Bottle","Glass Bottle","Colored-Glass-Bottles",
               "Colored Glass Bottles"],
 "can":       ["Cans"],
 "cup":       ["Plastic Cup","PlasticCups","Paper Cup","Paper-Cup"],
 "organic":   ["Pile of Leaves","Peel"],
 None:        ["Rags"],
}
CANON = {}
for target, names in CANON_RAW.items():
    for n in names: CANON[norm(n)] = target

def find_src():
    if len(sys.argv) > 1: return sys.argv[1]
    base = r"D:\Y3\dip"
    for d in sorted(os.listdir(base)):
        p = os.path.join(base, d)
        if os.path.isdir(p) and os.path.isdir(os.path.join(p, "train", "images")):
            if "rf_trash" in d or "Trash" in d or "yolov" in d.lower(): return p
    raise SystemExit("source not found; pass path as arg")

SRC = find_src()
DST = r"D:\Y3\dip\rf_trash_clean"
VAL_FRAC = 0.15
SEED = 42
print("SRC:", SRC)

# ---- read source class list ----
y = open(os.path.join(SRC,"data.yaml"), encoding="utf-8", errors="ignore").read()
m = re.search(r"names:\s*\[(.*?)\]", y, re.S)
src_names = [s.strip().strip("'\"") for s in m.group(1).split(",")] if m else []
if not src_names:
    m2 = re.findall(r"^\s*(\d+):\s*(.+)$", y, re.M)
    src_names = [n.strip().strip("'\"") for _, n in sorted(m2, key=lambda t:int(t[0]))]
print("source classes:", len(src_names))
unknown = [n for n in src_names if norm(n) not in CANON]
if unknown: print("  !! unmapped source classes:", unknown)

src2tgt = {}
for i, n in enumerate(src_names):
    t = CANON.get(norm(n))
    src2tgt[i] = None if t is None else IDX[t]
print("mapping:", {src_names[i]: (NAMES[v] if v is not None else "DROP") for i,v in src2tgt.items() if i < len(src_names)})

# ---- scan images ----
items = []
for split in ("train","valid","test"):
    idir = os.path.join(SRC, split, "images")
    if not os.path.isdir(idir): continue
    for f in sorted(os.listdir(idir)):
        items.append((split, f))
print("images found:", len(items))

box_hist = Counter(); drop_hist = Counter(); neg = []; rows = []
for split, f in items:
    stem = os.path.splitext(f)[0]
    lp = os.path.join(SRC, split, "labels", stem + ".txt")
    out = []
    if os.path.exists(lp):
        for line in open(lp):
            t = line.split()
            if len(t) != 5: continue
            ci = int(float(t[0]))
            tgt = src2tgt.get(ci)
            if tgt is None:
                drop_hist[src_names[ci] if ci < len(src_names) else ci] += 1
                continue
            out.append(f"{tgt} " + " ".join(f"{float(v):.6f}" for v in t[1:]))
            box_hist[NAMES[tgt]] += 1
    if not out: neg.append((split,f))
    rows.append({"split":split,"file":f,"stem":stem,"lines":out})
print(f"boxes kept: {sum(box_hist.values())}   boxes dropped: {sum(drop_hist.values())} {dict(drop_hist)}")
print("per class:", dict(box_hist))
print("negatives (no box after remap):", len(neg))

# ---- dedup by image content (dHash prefilter, template confirm) ----
H, W = 9, 8
bits = np.zeros((len(rows), H*W), dtype=np.uint8)
for i, r in enumerate(rows):
    p = os.path.join(SRC, r["split"], "images", r["file"])
    im = Image.open(p).convert("L").resize((W+1, H), Image.LANCZOS)
    a = np.asarray(im, dtype=np.int16)
    bits[i] = (a[:, 1:] > a[:, :-1]).reshape(-1)
parent = list(range(len(rows)))
def find(a):
    while parent[a] != a: parent[a] = parent[parent[a]]; a = parent[a]
    return a
def union(a,b):
    ra, rb = find(a), find(b)
    if ra != rb: parent[rb] = ra
cand = 0
for s in range(0, len(rows), 256):
    e = min(s+256, len(rows))
    d = (bits[s:e, None, :] != bits[None, :, :]).sum(axis=2)
    for ii, jj in zip(*np.where(d <= 10)):
        i = s+ii
        if i < jj:
            union(i, int(jj)); cand += 1
print("dhash candidate duplicate pairs:", cand)
groups = defaultdict(list)
for i in range(len(rows)): groups[find(i)].append(i)
dups = [g for g in groups.values() if len(g) > 1]
print("duplicate groups:", len(dups), " images involved:", sum(len(g) for g in dups))

# ---- stratified split at group level ----
def signature(gi):
    sig = tuple(sorted({int(l.split()[0]) for i in groups[gi] for l in rows[i]["lines"]}))
    return sig
allg = list(groups.keys())
bysig = defaultdict(list)
for g in allg: bysig[signature(g)].append(g)
rng = random.Random(SEED)
assign = {}
for sig, gs in bysig.items():
    gs = sorted(gs, key=lambda g: rows[groups[g][0]]["file"])
    rng.shuffle(gs)
    nval = max(1, round(len(gs)*VAL_FRAC)) if len(gs) > 3 else (1 if len(gs) > 1 and rng.random() < VAL_FRAC else 0)
    for k, g in enumerate(gs): assign[g] = "val" if k < nval else "train"
print("groups:", len(allg), " by signature:", len(bysig))

# ---- write ----
if os.path.isdir(DST): shutil.rmtree(DST)
for sp in ("train","val"):
    os.makedirs(os.path.join(DST, sp, "images"), exist_ok=True)
    os.makedirs(os.path.join(DST, sp, "labels"), exist_ok=True)
stats = defaultdict(lambda: Counter())
nimg = Counter(); nbox = Counter()
for g, items_idx in groups.items():
    sp = assign[g]
    for i in items_idx:
        r = rows[i]
        shutil.copy2(os.path.join(SRC, r["split"], "images", r["file"]),
                     os.path.join(DST, sp, "images", r["file"]))
        with open(os.path.join(DST, sp, "labels", r["stem"] + ".txt"), "w") as fh:
            fh.write("\n".join(r["lines"]) + ("\n" if r["lines"] else ""))
        nimg[sp] += 1; nbox[sp] += len(r["lines"])
        for l in r["lines"]: stats[sp][NAMES[int(l.split()[0])]] += 1
with open(os.path.join(DST, "data.yaml"), "w") as fh:
    fh.write("path: .\ntrain: train/images\nval: val/images\n\nnames:\n")
    for i, n in enumerate(NAMES): fh.write(f"  {i}: {n}\n")
json.dump({"mapping": {src_names[i]: (NAMES[v] if v is not None else "DROP") for i, v in src2tgt.items()},
           "dropped": dict(drop_hist), "classes": NAMES}, open(os.path.join(DST,"mapping.json"),"w"), indent=1, ensure_ascii=False)
print("\n=== RESULT ===")
for sp in ("train","val"):
    print(f"{sp}: images={nimg[sp]} boxes={nbox[sp]}")
    print("   ", dict(stats[sp]))
print("wrote:", DST)
