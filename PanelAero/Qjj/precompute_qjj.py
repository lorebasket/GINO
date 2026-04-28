from pathlib import Path
import json
import numpy as np



def precompute_qjj_grid(FSI_path, aerogrid, k_list, Ma_list, out_dir,
                        dtype=np.float32, verbose=True, resume=True, verify_existing=False, dlm_method='parabolic'):
    """
    Old version: precomputes over k and Ma grid
    
    Args:
        resume: If True, skip cases where files already exist
        verify_existing: If True, check existing files for NaN/Inf and recompute if needed
        dlm_method: Integration method for DLM ('parabolic' or 'quartic')
    """
    import sys
    sys.path.append(FSI_path + '/PanelAero')
    from panelaero_utl import DLM

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    k_list = np.asarray(k_list, dtype=float)
    Ma_list = np.asarray(Ma_list, dtype=float)

    Q_shape = None
    nan_cases = []
    successful_cases = 0
    failed_cases = 0
    skipped_cases = 0

    total_cases = len(k_list) * len(Ma_list)
    
    print(f"\n{'='*60}")
    print(f"Starting Qjj precomputation")
    print(f"Total cases: {total_cases}")
    print(f"DLM method: {dlm_method}")
    print(f"Resume mode: {resume}")
    print(f"Verify existing: {verify_existing}")
    print(f"{'='*60}\n")

    for i_k, k in enumerate(k_list):
        for i_Ma, Ma in enumerate(Ma_list):
            
            case_idx = i_k * len(Ma_list) + i_Ma + 1
            
            fname_base = out_dir / f"k_{i_k:03d}__Ma_{i_Ma:04d}"
            file_real = f"{fname_base}_real.npy"
            file_imag = f"{fname_base}_imag.npy"
            
            # Check if files already exist
            files_exist = Path(file_real).exists() and Path(file_imag).exists()
            
            if files_exist and resume:
                # Optionally verify existing files
                should_skip = True
                
                if verify_existing:
                    try:
                        real_data = np.load(file_real, mmap_mode='r')
                        imag_data = np.load(file_imag, mmap_mode='r')
                        
                        if np.any(np.isnan(real_data)) or np.any(np.isnan(imag_data)) or \
                           np.any(np.isinf(real_data)) or np.any(np.isinf(imag_data)):
                            print(f"[{case_idx}/{total_cases}] k[{i_k}]={k:.6g}, Ma[{i_Ma}]={Ma:.6g}")
                            print(f"  ⚠️  Existing file is corrupted, will recompute")
                            should_skip = False
                    except Exception as e:
                        print(f"[{case_idx}/{total_cases}] k[{i_k}]={k:.6g}, Ma[{i_Ma}]={Ma:.6g}")
                        print(f"  ⚠️  Cannot read existing file: {e}, will recompute")
                        should_skip = False
                
                if should_skip:
                    if verbose and case_idx % 10 == 0:  # Print every 10 cases to avoid spam
                        print(f"[{case_idx}/{total_cases}] Skipping existing cases... "
                              f"(Progress: {case_idx/total_cases*100:.1f}%)")
                    skipped_cases += 1
                    
                    # Still need to get Q_shape from somewhere
                    if Q_shape is None:
                        try:
                            temp_data = np.load(file_real, mmap_mode='r')
                            Q_shape = temp_data.shape
                        except:
                            pass
                    
                    continue
            
            # Compute this case
            if verbose:
                print(f"[{case_idx}/{total_cases}] Computing k[{i_k}]={k:.6g}, Ma[{i_Ma}]={Ma:.6g}")

            try:
                # === calcolo Qjj complessa per questo (k,Ma) ===
                Qjj = DLM.calc_Qjj(aerogrid, Ma, k, method=dlm_method)

                # CHECK FOR NaN/Inf BEFORE SAVING
                nan_count = np.sum(np.isnan(Qjj))
                inf_count = np.sum(np.isinf(Qjj))
                
                if nan_count > 0 or inf_count > 0:
                    print(f"  ⚠️  WARNING: Found {nan_count} NaNs and {inf_count} Infs")
                    nan_cases.append({
                        'i_k': i_k, 'k': k,
                        'i_Ma': i_Ma, 'Ma': Ma,
                        'nan_count': nan_count,
                        'inf_count': inf_count,
                        'status': 'invalid_values'
                    })
                    failed_cases += 1
                    continue

                if Q_shape is None:
                    Q_shape = Qjj.shape

                # Salvataggio separato Re/Im
                np.save(file_real, Qjj.real.astype(dtype), allow_pickle=False)
                np.save(file_imag, Qjj.imag.astype(dtype), allow_pickle=False)
                
                successful_cases += 1
                if verbose:
                    print(f"  ✓ Saved (Progress: {case_idx/total_cases*100:.1f}%)")

            except Exception as e:
                print(f"  ❌ ERROR: {e}")
                nan_cases.append({
                    'i_k': i_k, 'k': k,
                    'i_Ma': i_Ma, 'Ma': Ma,
                    'error': str(e),
                    'status': 'exception'
                })
                failed_cases += 1

    # Salva report dei casi problematici
    if nan_cases:
        report_file = out_dir / "problematic_cases.txt"
        with open(report_file, 'w') as f:
            f.write(f"PROBLEMATIC CASES REPORT\n")
            f.write(f"========================\n\n")
            f.write(f"Total cases: {total_cases}\n")
            f.write(f"Successful: {successful_cases}\n")
            f.write(f"Failed: {failed_cases}\n")
            f.write(f"Skipped (already existed): {skipped_cases}\n\n")
            f.write(f"Details of problematic cases:\n")
            f.write("-" * 80 + "\n")
            
            for case in nan_cases:
                f.write(f"\nCase: i_k={case['i_k']:3d}, k={case['k']:8.4f}, "
                       f"i_Ma={case['i_Ma']:3d}, Ma={case['Ma']:8.4f}\n")
                if case['status'] == 'invalid_values':
                    f.write(f"  NaN count: {case['nan_count']}\n")
                    f.write(f"  Inf count: {case['inf_count']}\n")
                else:
                    f.write(f"  Error: {case['error']}\n")
        
        print(f"\n⚠️  Found {len(nan_cases)} problematic cases")
        print(f"    Report saved to: {report_file}")

    # indice
    index = {
        "k_list": k_list.tolist(),
        "Ma_list": Ma_list.tolist(),
        "shape": list(Q_shape) if Q_shape else None,
        "dtype": np.dtype(dtype).name,
        "n_k": len(k_list),
        "n_Ma": len(Ma_list),
        "total_cases": total_cases,
        "successful_cases": successful_cases,
        "failed_cases": failed_cases,
        "skipped_cases": skipped_cases,
        "pattern": "k_{i_k:03d}__Ma_{i_Ma:04d}_(real|imag).npy"
    }
    (out_dir / "index.json").write_text(json.dumps(index, indent=2))
    
    print(f"\n{'='*60}")
    print(f"✓ Precomputation complete")
    print(f"    Total cases: {total_cases}")
    print(f"    Computed now: {successful_cases}")
    print(f"    Failed: {failed_cases}")
    print(f"    Skipped (existed): {skipped_cases}")
    print(f"    Index: {out_dir/'index.json'}")
    print(f"{'='*60}\n")


from pathlib import Path
import numpy as np

def check_progress(out_dir, n_k, n_Ma):
    """Check how many Qjj files have been computed"""
    
    out_dir = Path(out_dir)
    
    total_cases = n_k * n_Ma
    existing_real = len(list(out_dir.glob("*_real.npy")))
    existing_imag = len(list(out_dir.glob("*_imag.npy")))
    
    # Both real and imag must exist
    complete_cases = min(existing_real, existing_imag)
    
    print(f"\nProgress Report:")
    print(f"{'='*50}")
    print(f"Total expected: {total_cases} cases")
    print(f"Complete cases: {complete_cases} ({complete_cases/total_cases*100:.1f}%)")
    print(f"Missing: {total_cases - complete_cases}")
    print(f"{'='*50}\n")
    
    # Find which specific cases are missing
    missing = []
    for i_k in range(n_k):
        for i_Ma in range(n_Ma):
            fname_base = out_dir / f"k_{i_k:03d}__Ma_{i_Ma:04d}"
            if not (Path(f"{fname_base}_real.npy").exists() and 
                    Path(f"{fname_base}_imag.npy").exists()):
                missing.append((i_k, i_Ma))
    
    if missing:
        print(f"First 10 missing cases (i_k, i_Ma):")
        for i_k, i_Ma in missing[:10]:
            print(f"  k[{i_k}], Ma[{i_Ma}]")
        if len(missing) > 10:
            print(f"  ... and {len(missing)-10} more")
    
    return complete_cases, missing




def _load_index_old(root):
    """loads k and Ma lists"""
    root = Path(root)
    idx = json.loads((root / "index.json").read_text())
    return {
        "root": root,
        "k_list": np.array(idx["k_list"], dtype=float),
        "Ma_list": np.array(idx["Ma_list"], dtype=float),
        "shape": tuple(idx["shape"]),
        "dtype": np.float32 if idx["dtype"] == "float32" else np.float64,
        "n_k": int(idx["n_k"]),
        "n_Ma": int(idx["n_Ma"]),
    }


def _load_slice_old(root, i_k, i_Ma, part, mmap=True, dtype=np.float32):
    """loads files named k_{i_k:03d}__Ma_{i_Ma:04d}"""
    root = Path(root)
    suffix = "_real.npy" if part == "real" else "_imag.npy"
    path = root / f"k_{i_k:03d}__Ma_{i_Ma:04d}{suffix}"
    return np.load(path, mmap_mode="r" if mmap else None).astype(dtype, copy=False)

def _bracket(arr, x):
    """
    Restituisce (i0, i1, t) tali che arr[i0] <= x <= arr[i1] e
    t è il peso lineare in [0,1] (t=0 -> i0, t=1 -> i1).
    Se x è fuori range, clamp ai bordi e t=0 o t=1.
    """
    n = len(arr)
    if n < 2:
        return 0, 0, 0.0  # Single element array
    if x <= arr[0]: 
        return 0, 0, 0.0
    if x >= arr[-1]: 
        return n-2, n-1, 1.0  # Clamp to last valid pair, t=1
    i1 = int(np.searchsorted(arr, x, side="right"))
    i0 = i1 - 1
    # Ensure indices are valid
    if i1 >= n:
        i1 = n - 1
        i0 = n - 2
    t = (x - arr[i0]) / (arr[i1] - arr[i0])
    return i0, i1, float(t)

def interp_qjj_from_disk_old(root, k_query, Ma_query, mmap=True):
    """
    Interpola bilinearmente Qjj in (k, Ma)
    """
    meta = _load_index_old(root)
    k_list, Ma_list = meta["k_list"], meta["Ma_list"]
    dtype = meta["dtype"]

    j0, j1, tk = _bracket(k_list, k_query)
    i0, i1, tm = _bracket(Ma_list, Ma_query)

    # Carica solo i vicini necessari
    R00 = _load_slice_old(meta["root"], j0, i0, "real", mmap, dtype)
    R10 = _load_slice_old(meta["root"], j1, i0, "real", mmap, dtype) if j1 != j0 else R00
    R01 = _load_slice_old(meta["root"], j0, i1, "real", mmap, dtype) if i1 != i0 else R00
    R11 = _load_slice_old(meta["root"], j1, i1, "real", mmap, dtype) if (i1 != i0 or j1 != j0) else R00

    I00 = _load_slice_old(meta["root"], j0, i0, "imag", mmap, dtype)
    I10 = _load_slice_old(meta["root"], j1, i0, "imag", mmap, dtype) if j1 != j0 else I00
    I01 = _load_slice_old(meta["root"], j0, i1, "imag", mmap, dtype) if i1 != i0 else I00
    I11 = _load_slice_old(meta["root"], j1, i1, "imag", mmap, dtype) if (i1 != i0 or j1 != j0) else I00

    # Interpolazione bilineare
    def bilinear(A00, A10, A01, A11):
        return (1-tm)*((1-tk)*A00 + tk*A10) + tm*((1-tk)*A01 + tk*A11)

    Qr = bilinear(R00, R10, R01, R11)
    Qi = bilinear(I00, I10, I01, I11)

    return Qr.astype(dtype, copy=False) + 1j * Qi.astype(dtype, copy=False)


def open_qjj_index_old(root):
    """Old version: returns k_list and Ma_list"""
    meta = _load_index_old(root)
    return meta["k_list"], meta["Ma_list"], meta["shape"], meta["dtype"]

def open_qjj_index(root):
    """Comodo se vuoi solo leggere k_list e Ma_list per sapere cosa c'è su disco."""
    meta = _load_index(root)
    return meta["omega_list"], meta["V_list"], meta["shape"], meta["dtype"]

def _load_index(root):
    root = Path(root)
    idx = json.loads((root / "index.json").read_text())
    return {
        "root": root,
        "omega_list": np.array(idx["omega_list"], dtype=float),
        "V_list": np.array(idx["V_list"], dtype=float),
        "shape": tuple(idx["shape"]),
        "dtype": np.float32 if idx["dtype"] == "float32" else np.float64,
        "n_omega": int(idx["n_omega"]),
        "dtype": dtype
    }

def _bracket(arr, x):
    """
    Restituisce (i0, i1, t) tali che arr[i0] <= x <= arr[i1] e
    t è il peso lineare in [0,1] (t=0 -> i0, t=1 -> i1).
    Se x è fuori range, clamp ai bordi e t=0 o t=1.
    """
    n = len(arr)
    if n < 2:
        return 0, 0, 0.0  # Single element array
    if x <= arr[0]: 
        return 0, 0, 0.0
    if x >= arr[-1]: 
        return n-2, n-1, 1.0  # Clamp to last valid pair, t=1
    i1 = int(np.searchsorted(arr, x, side="right"))
    i0 = i1 - 1
    # Ensure indices are valid
    if i1 >= n:
        i1 = n - 1
        i0 = n - 2
    t = (x - arr[i0]) / (arr[i1] - arr[i0])
    return i0, i1, float(t)

def _load_slice(root, i_omega, i_V, part, mmap=True, dtype=np.float32):
    """
    part: 'real' oppure 'imag'
    """
    root = Path(root)
    suffix = "_real.npy" if part == "real" else "_imag.npy"
    path = root / f"omega_{i_omega:03d}__V_{i_V:04d}{suffix}"
    # mmap_mode='r' evita di caricare in RAM tutto subito
    return np.load(path, mmap_mode="r" if mmap else None).astype(dtype, copy=False)

def interp_qjj_from_disk(root, omega_query, V_query, mmap=True):
    """
    Interpola bilinearmente Qjj in (omega, V) caricando solo
    i 2x2 vicini (fino a 4 matrici) per Re e Im, poi ricompone la parte complessa.
    """
    meta = _load_index(root)
    omega_list, V_list = meta["omega_list"], meta["V_list"]
    dtype = meta["dtype"]

    # FIX: These lines were still using k_list and Ma_list
    i0, i1, t_omega = _bracket(omega_list, omega_query)
    j0, j1, t_V = _bracket(V_list, V_query)

    # Carica solo i vicini necessari
    R00 = _load_slice(meta["root"], i0, j0, "real", mmap, dtype)
    R10 = _load_slice(meta["root"], i1, j0, "real", mmap, dtype) if i1 != i0 else R00
    R01 = _load_slice(meta["root"], i0, j1, "real", mmap, dtype) if j1 != j0 else R00
    R11 = _load_slice(meta["root"], i1, j1, "real", mmap, dtype) if (i1 != i0 or j1 != j0) else R00

    I00 = _load_slice(meta["root"], i0, j0, "imag", mmap, dtype)
    I10 = _load_slice(meta["root"], i1, j0, "imag", mmap, dtype) if i1 != i0 else I00
    I01 = _load_slice(meta["root"], i0, j1, "imag", mmap, dtype) if j1 != j0 else I00
    I11 = _load_slice(meta["root"], i1, j1, "imag", mmap, dtype) if (i1 != i0 or j1 != j0) else I00

    # Interpolazione bilineare separata su Re e Im:
    # (1-t_omega)*( (1-t_V)*00 + t_V*01 ) + t_omega*( (1-t_V)*10 + t_V*11 )
    def bilinear(A00, A10, A01, A11):
        return (1-t_omega)*((1-t_V)*A00 + t_V*A01) + t_omega*((1-t_V)*A10 + t_V*A11)

    Qr = bilinear(R00, R10, R01, R11)
    Qi = bilinear(I00, I10, I01, I11)

    # Ricomponi complesso
    return Qr.astype(dtype, copy=False) + 1j * Qi.astype(dtype, copy=False)

# ---------------------------------------------------

def check_existing_qjj_files(out_dir):
    """Verifica i file Qjj già salvati per NaN/Inf"""
    
    out_dir = Path(out_dir)
    
    print(f"Checking files in: {out_dir}\n")
    
    # Load index
    index_file = out_dir / "index.json"
    if index_file.exists():
        with open(index_file, 'r') as f:
            index = json.load(f)
        print(f"Index info:")
        print(f"  k range: {index['k_list'][0]:.4f} to {index['k_list'][-1]:.4f}")
        print(f"  Ma range: {index['Ma_list'][0]:.4f} to {index['Ma_list'][-1]:.4f}")
        print(f"  Expected files: {index['n_k'] * index['n_Ma'] * 2}\n")
    
    problematic_files = []
    total_files = 0
    
    for file in sorted(out_dir.glob("*.npy")):
        total_files += 1
        print(f"Checking file: {file}")
        print(f"")

        try:
            data = np.load(file, mmap_mode='r')
            nan_count = np.sum(np.isnan(data))
            inf_count = np.sum(np.isinf(data))
            
            if nan_count > 0 or inf_count > 0:
                problematic_files.append({
                    'file': file.name,
                    'nans': nan_count,
                    'infs': inf_count,
                    'size': data.size
                })
                print(f"❌ {file.name}: {nan_count} NaNs, {inf_count} Infs")

        except Exception as e:
            print(f"❌ Error reading {file.name}: {e}")
            problematic_files.append({
                'file': file.name,
                'error': str(e)
            })
    
    print(f"\n{'='*60}")
    print(f"SUMMARY:")
    print(f"  Total files checked: {total_files}")
    print(f"  Problematic files: {len(problematic_files)}")
    print(f"  Clean files: {total_files - len(problematic_files)}")
    
    if problematic_files:
        report_file = out_dir / "verification_report.txt"
        with open(report_file, 'w') as f:
            f.write("VERIFICATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            for prob in problematic_files:
                f.write(f"File: {prob['file']}\n")
                if 'error' in prob:
                    f.write(f"  Error: {prob['error']}\n")
                else:
                    f.write(f"  NaNs: {prob['nans']}, Infs: {prob['infs']}, Total size: {prob['size']}\n")
                f.write("\n")
        print(f"\nDetailed report saved to: {report_file}")

if __name__ == "__main__":
    out_dir = "/media/lorenzo/Seagate Por/recupero dati Linux/lorebasket/FSI/PanelAero/Qjj/qjj_precomputed/GOLAND_air_alpha2_nspan50_nchord30_klist_new"
    check_existing_qjj_files(out_dir)