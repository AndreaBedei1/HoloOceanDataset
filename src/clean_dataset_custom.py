from pathlib import Path
from typing import Iterable, Tuple, Dict, List
import os

# ===================== CONFIG =====================

# Mine
    # 0, 4),
    # (8, 13),
    # (18, 23),
    # (27, 31),

    # (0, 5),
    # (9, 14),
    # (17, 23),
    # (27, 31),

    # (0, 6),
    # (12, 18),
    # (24, 32),
    # (37, 44),

# Anchor
    # (0, 4),
    # (10, 14),
    # (19, 24),
    # (28, 33),


    # (0, 6),
    # (14, 19),
    # (26, 31),
    # (39, 44),

#Torpedo

    # (2, 5),
    # (11, 13),
    # (21, 22),
    # (30, 32),

    # (0, 5),
    # (8, 16),
    # (18, 25),
    # (27, 33),

    # (3, 5),
    # (15, 18),
    # (28, 30),
    # (40, 44),


DATASET_ROOT = Path("dataset/runs")
RUN_ID = "run_0201"

# 4 range inclusivi (start, end)
KEEP_RANGES: list[Tuple[int, int]] = [
    (1, 4),
    (11, 12),
    (20, 22),
    (29, 31),

]

TARGET_DIRS = [
    Path("images/front"),
    Path("images/bottom"),
    Path("sonar_raw"),
]

ALLOWED_EXT = {".png", ".npz"}  

STRICT_ALIGNMENT = True  

# ===================== HELPERS =====================

def normalize_ranges(ranges: Iterable[Tuple[int, int]]) -> list[Tuple[int, int]]:
    rs = sorted((min(a, b), max(a, b)) for a, b in ranges)
    merged: list[Tuple[int, int]] = []
    for a, b in rs:
        if not merged:
            merged.append((a, b))
            continue
        la, lb = merged[-1]
        if a <= lb + 1:
            merged[-1] = (la, max(lb, b))
        else:
            merged.append((a, b))
    return merged

def in_ranges(x: int, ranges: list[Tuple[int, int]]) -> bool:
    for a, b in ranges:
        if a <= x <= b:
            return True
    return False

def list_numeric_files(dir_path: Path) -> Dict[int, Path]:
    out: Dict[int, Path] = {}
    for p in dir_path.iterdir():
        if not p.is_file():
            continue
        if p.suffix not in ALLOWED_EXT:
            continue
        if not p.stem.isdigit():
            continue
        i = int(p.stem)
        
        if i in out:
            raise RuntimeError(f"ID duplicato {i} in {dir_path}: {out[i].name} e {p.name}")
        out[i] = p
    return out

def safe_rename_batch(rename_map: List[Tuple[Path, Path]]) -> None:
    tmp_map: List[Tuple[Path, Path]] = []
    for old, new in rename_map:
        tmp = old.with_name(f"__tmp__{new.name}")
        tmp_map.append((old, tmp))

    for old, tmp in tmp_map:
        os.replace(old, tmp)

    for (_, tmp), (_, new) in zip(tmp_map, rename_map):
        os.replace(tmp, new)

# ===================== MAIN =====================

KEEP_RANGES_NORM = normalize_ranges(KEEP_RANGES)
run_path = DATASET_ROOT / RUN_ID
if not run_path.exists():
    raise FileNotFoundError(f"Run not found: {run_path}")

dir_paths = [run_path / d for d in TARGET_DIRS]
for d in dir_paths:
    if not d.exists():
        raise FileNotFoundError(f"Missing folder: {d}")

print(f"Run: {run_path}")
print(f"Keep ranges (inclusive): {KEEP_RANGES_NORM}")

files_by_dir: Dict[Path, Dict[int, Path]] = {}
width_by_dir: Dict[Path, int] = {}

for d in dir_paths:
    m = list_numeric_files(d)
    files_by_dir[d] = m
    width_by_dir[d] = max((len(p.stem) for p in m.values()), default=6)

for d in dir_paths:
    m = files_by_dir[d]
    deleted = 0
    for i, p in list(m.items()):
        if not in_ranges(i, KEEP_RANGES_NORM):
            os.remove(p)
            deleted += 1
            del m[i]
    print(f"[FILTER] {d} -> deleted={deleted}, remaining={len(m)}")

id_sets = [set(files_by_dir[d].keys()) for d in dir_paths]

if STRICT_ALIGNMENT:
    base = id_sets[0]
    for s, d in zip(id_sets[1:], dir_paths[1:]):
        if s != base:
            missing_in_d = sorted(base - s)
            extra_in_d = sorted(s - base)
            raise RuntimeError(
                f"ID misalignment after filtering.\n"
                f"Directory: {d}\n"
                f"Missing IDs compared to the first one: {missing_in_d[:20]}{'...' if len(missing_in_d)>20 else ''}\n"
                f"Extra IDs found here: {extra_in_d[:20]}{'...' if len(extra_in_d)>20 else ''}\n"
                f"Suggestion: regenerate the run or set STRICT_ALIGNMENT=False (risky)."
            )
    kept_ids = sorted(base)
else:
    kept_ids = sorted(set.intersection(*id_sets))
    print(f"[WARN] STRICT_ALIGNMENT=False: {len(kept_ids)}")

if not kept_ids:
    raise RuntimeError("No IDs left after filtering. Check the ranges.")

old_to_new = {old: new for new, old in enumerate(kept_ids)}
print(f"[RENUMBER] frames kept: {len(kept_ids)} | new ids: 0..{len(kept_ids)-1}")

for d in dir_paths:
    m = files_by_dir[d]
    w = width_by_dir[d]

    rename_ops: List[Tuple[Path, Path]] = []
    for old_id in kept_ids:
        old_path = m.get(old_id)
        if old_path is None:
            continue
        new_id = old_to_new[old_id]
        new_name = f"{new_id:0{w}d}{old_path.suffix}"
        new_path = old_path.with_name(new_name)
        if old_path.name != new_name:
            rename_ops.append((old_path, new_path))

    safe_rename_batch(rename_ops)
    print(f"[RENAME] {d} -> renamed={len(rename_ops)} (width={w})")

print("Done.")
