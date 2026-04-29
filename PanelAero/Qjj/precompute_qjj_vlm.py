import json
from pathlib import Path
import numpy as np

def precompute_qjj_vlm(FSI_path, aerogrid, Ma_list, out_dir,
                       dtype=np.float32, verbose=True, resume=True, verify_existing=False):
    """
    Precompute steady (real) Qjj using VLM.calc_Qjj for a list of Mach numbers.
    Saves one real .npy per Ma: ma_{i:04d}.npy and an index.json with Ma_list and shape.
    """
    from panelaero import VLM

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    Ma_list = np.asarray(Ma_list, dtype=float)

    total = len(Ma_list)
    shape = None
    succeeded = 0
    failed = 0
    skipped = 0
    bad_cases = []

    for i_ma, Ma in enumerate(Ma_list):
        idx = i_ma + 1
        fname = out_dir / f"ma_{i_ma:04d}.npy"

        if fname.exists() and resume:
            if verify_existing:
                try:
                    data = np.load(fname, mmap_mode='r')
                    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                        if verbose:
                            print(f"[{idx}/{total}] ma[{i_ma}]={Ma:.6g} : corrupted, will recompute")
                    else:
                        if verbose and idx % 10 == 0:
                            print(f"[{idx}/{total}] Skipping existing ma[{i_ma}] (progress {idx/total*100:.1f}%)")
                        skipped += 1
                        if shape is None:
                            shape = data.shape
                        continue
                except Exception:
                    # recompute if cannot read
                    pass
            else:
                if verbose and idx % 10 == 0:
                    print(f"[{idx}/{total}] Skipping existing ma[{i_ma}] (progress {idx/total*100:.1f}%)")
                skipped += 1
                try:
                    if shape is None:
                        shape = np.load(fname, mmap_mode='r').shape
                except Exception:
                    pass
                continue

        if verbose:
            print(f"[{idx}/{total}] Computing ma[{i_ma}] = {Ma:.6g}")

        try:
            res = VLM.calc_Qjj(aerogrid, Ma)
            # VLM.calc_Qjj may return (Qjj, Bjj) or Qjj directly
            Qjj = res[0] if isinstance(res, tuple) else res
            Qjj = np.asarray(Qjj)

            nan_count = np.sum(np.isnan(Qjj))
            inf_count = np.sum(np.isinf(Qjj))
            if nan_count or inf_count:
                bad_cases.append({'i_ma': i_ma, 'Ma': Ma, 'nans': int(nan_count), 'infs': int(inf_count)})
                failed += 1
                if verbose:
                    print(f"  ⚠️  Found {nan_count} NaNs and {inf_count} Infs, skipping save")
                continue

            if shape is None:
                shape = Qjj.shape

            np.save(fname, Qjj.astype(dtype), allow_pickle=False)
            succeeded += 1
            if verbose:
                print(f"  ✓ Saved ma_{i_ma:04d}.npy")

        except Exception as e:
            failed += 1
            bad_cases.append({'i_ma': i_ma, 'Ma': Ma, 'error': str(e)})
            if verbose:
                print(f"  ❌ ERROR computing ma[{i_ma}]={Ma}: {e}")

    # write index
    index = {
        "Ma_list": Ma_list.tolist(),
        "shape": list(shape) if shape is not None else None,
        "dtype": np.dtype(dtype).name,
        "n_Ma": int(len(Ma_list)),
        "total": int(total),
    }
    (out_dir / "index.json").write_text(json.dumps(index, indent=2))

    if bad_cases:
        report = out_dir / "problematic_cases.json"
        report.write_text(json.dumps(bad_cases, indent=2))

    if verbose:
        print(f"\nSummary: total={total}, saved={succeeded}, skipped={skipped}, failed={failed}")
        print(f"Index saved to: {out_dir/'index.json'}")


def _load_index(root):
    root = Path(root)
    idx = json.loads((root / "index.json").read_text())
    return {
        "root": root,
        "Ma_list": np.array(idx["Ma_list"], dtype=float),
        "shape": tuple(idx["shape"]) if idx.get("shape") else None,
        "dtype": np.float32 if idx["dtype"] == "float32" else np.float64,
        "n_Ma": int(idx["n_Ma"]),
    }

def _load_slice(root, i_ma, mmap=True, dtype=np.float32):
    root = Path(root)
    path = root / f"ma_{i_ma:04d}.npy"
    return np.load(path, mmap_mode="r" if mmap else None).astype(dtype, copy=False)

def _bracket(arr, x):
    if x <= arr[0]: return 0, 0, 0.0
    if x >= arr[-1]: return len(arr)-1, len(arr)-1, 0.0
    i1 = int(np.searchsorted(arr, x, side="right"))
    i0 = i1 - 1
    t = (x - arr[i0]) / (arr[i1] - arr[i0])
    return i0, i1, float(t)

def interp_qjj_vlm_from_disk(root, Ma_query, mmap=True, dtype=np.float32):
    """
    Linearly interpolate Qjj in Ma (1D). Loads up to two neighboring ma_* files.
    Returns a real ndarray (shape from index).
    """
    meta = _load_index(root)
    Ma_list = meta["Ma_list"]

    i0, i1, t = _bracket(Ma_list, Ma_query)

    R0 = _load_slice(meta["root"], i0, mmap, dtype)
    if i1 == i0:
        return R0.astype(dtype, copy=False)
    R1 = _load_slice(meta["root"], i1, mmap, dtype)

    return ((1.0 - t) * R0 + t * R1).astype(dtype, copy=False)

def open_qjj_index_vlm(root):
    meta = _load_index(root)
    return meta["Ma_list"], meta["shape"], meta["dtype"]

def check_progress_vlm(out_dir):
    out_dir = Path(out_dir)
    meta = _load_index(out_dir)
    n_expected = meta["n_Ma"]
    existing = len(list(out_dir.glob("ma_*.npy")))
    print(f"Progress: {existing}/{n_expected} files ({existing/n_expected*100:.1f}%)")
    missing = [i for i in range(n_expected) if not (out_dir / f"ma_{i:04d}.npy").exists()]
    return existing, missing

if __name__ == "__main__":
    # quick CLI for manual use (edit paths as needed)
    out_dir = "/media/lorenzo/Seagate Por/recupero dati Linux/lorebasket/FSI/PanelAero/Qjj/qjj_precomputed_vlm"
    print("Run precompute_qjj_vlm.precompute_qjj_vlm(...) from your script with proper args")