#compute the following summary statistics, finding accuracy, precision, and F1 for each
# Activation
# Note Onset			
# Note Onset+Offset			
# Note Onset+Offset+ vel.			
# pedal activation			
# pedal onset			
# pedal onset+offset

import os
import sys
import glob
import warnings
import json
import argparse
import csv
import numpy as np
import pretty_midi
import mir_eval
from pathlib import Path
from multiprocessing import Pool
import tqdm
from collections import defaultdict
import sys
import os

sys.path.append("/scratch/gilbreth/li5042/transkun/transkun_fork")
from transkun import Data

# Suppress warnings for clean stdout
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
warnings.filterwarnings("ignore", message=".*not a valid type 0 or type 1 MIDI file.*")
sys.setrecursionlimit(10000)

# ==========================================
# 1. Extraction Helpers
# ==========================================
def extract_notes_and_pedal(midi_path, extend_sustain=False):
    """Extracts note and pedal intervals natively using TransKun's Data.py parser."""
    notes = Data.parseMIDIFile(str(midi_path), extendSustainPedal=extend_sustain)
    
    n_intervals, n_pitches, n_vels = [], [], []
    p_intervals = []
    
    for note in notes:
        # Data.py marks real notes as > 0 pitch, and CC controllers with negative pitch (e.g. -64)
        if note.pitch > 0:
            n_intervals.append([note.start, note.end])
            n_pitches.append(note.pitch)
            n_vels.append(note.velocity)
        elif note.pitch == -64:
            p_intervals.append([note.start, note.end])
            
    if n_intervals:
        n_intervals, n_pitches, n_vels = zip(*sorted(zip(n_intervals, n_pitches, n_vels), key=lambda x: x[0][0]))
        n_intervals = np.array(n_intervals)
        n_pitches = mir_eval.util.midi_to_hz(np.array(n_pitches))
        n_vels = np.array(n_vels)
    else:
        n_intervals, n_pitches, n_vels = np.empty((0, 2)), np.array([]), np.array([])
        
    if p_intervals:
        p_intervals = np.array(sorted(p_intervals, key=lambda x: x[0]))
        p_pitches = np.ones(len(p_intervals)) * 440.0 # dummy pitch
    else:
        p_intervals, p_pitches = np.empty((0, 2)), np.array([])
        
    return n_intervals, n_pitches, n_vels, p_intervals, p_pitches

def calc_frame_metrics(ref_bin, est_bin):
    """Calculates Precision, Recall, and F1 on flat boolean arrays."""
    tp = np.sum(ref_bin & est_bin)
    fp = np.sum(~ref_bin & est_bin)
    fn = np.sum(ref_bin & ~est_bin)
    
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1

# ==========================================
# 2. Core Evaluation Logic
# ==========================================
def evaluate_pair(args):
    ref_path, est_path = args
    base_name = Path(ref_path).stem

    try:
        ref_pm = pretty_midi.PrettyMIDI(str(ref_path))
        est_pm = pretty_midi.PrettyMIDI(str(est_path))

        results = {}
        results["base_name"] = base_name

        # --- A. Note Extraction ---
        # The TransKun paper naturally extends GT offsets with the sustain pedal (`extend_sustain=True`)
        # But prediction notes are NOT pedal extended unless `--applyPedalExtensionOnEstimated` is passed (default False).
        ref_inv, ref_p, ref_v, ref_pedal_inv, ref_pedal_p = extract_notes_and_pedal(ref_path, extend_sustain=True)
        est_inv, est_p, est_v, est_pedal_inv, est_pedal_p = extract_notes_and_pedal(est_path, extend_sustain=False)

        # 1. Note Onset
        p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
            ref_inv, ref_p, est_inv, est_p, onset_tolerance=0.05, offset_ratio=None
        )
        results['Note Onset'] = (p, r, f)

        # 2. Note Onset + Offset
        p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
            ref_inv, ref_p, est_inv, est_p, onset_tolerance=0.05, offset_ratio=0.2, offset_min_tolerance=0.05
        )
        results['Note Onset+Offset'] = (p, r, f)

        # 3. Note Onset + Offset + Velocity
        p, r, f, _ = mir_eval.transcription_velocity.precision_recall_f1_overlap(
            ref_inv, ref_p, ref_v, est_inv, est_p, est_v, onset_tolerance=0.05, offset_ratio=0.2, offset_min_tolerance=0.05
        )
        results['Note Onset+Offset+Velocity'] = (p, r, f)

        # 4. Activation (Frame-Level Pitch)
        fs = 100 # 10ms resolution
        ref_pr = (ref_pm.get_piano_roll(fs=fs) > 0)
        est_pr = (est_pm.get_piano_roll(fs=fs) > 0)
        
        max_len = max(ref_pr.shape[1], est_pr.shape[1])
        ref_pr = np.pad(ref_pr, ((0, 0), (0, max_len - ref_pr.shape[1])))
        est_pr = np.pad(est_pr, ((0, 0), (0, max_len - est_pr.shape[1])))
        results['Activation'] = calc_frame_metrics(ref_pr.flatten(), est_pr.flatten())

        # --- B. Pedal Extraction ---

        if len(ref_pedal_inv) > 0 or len(est_pedal_inv) > 0:
            # 5. Pedal Onset
            p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
                ref_pedal_inv, ref_pedal_p, est_pedal_inv, est_pedal_p, onset_tolerance=0.05, offset_ratio=None
            )
            results['Pedal Onset'] = (p, r, f)

            # 6. Pedal Onset + Offset
            p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
                ref_pedal_inv, ref_pedal_p, est_pedal_inv, est_pedal_p, onset_tolerance=0.05, offset_ratio=0.2, offset_min_tolerance=0.05
            )
            results['Pedal Onset+Offset'] = (p, r, f)
            
            # 7. Pedal Activation (Frame-Level Pedal)
            ref_ped_state = np.zeros(max_len, dtype=bool)
            est_ped_state = np.zeros(max_len, dtype=bool)
            
            for start, end in ref_pedal_inv:
                ref_ped_state[int(start * fs):int(end * fs)] = True
            for start, end in est_pedal_inv:
                est_ped_state[int(start * fs):int(end * fs)] = True
                
            results['Pedal Activation'] = calc_frame_metrics(ref_ped_state, est_ped_state)
        else:
            results['Pedal Onset'] = (0,0,0)
            results['Pedal Onset+Offset'] = (0,0,0)
            results['Pedal Activation'] = (0,0,0)

        return results
        
    except Exception as e:
        # Catch mir_eval ValueErrors, PrettyMIDI parse errors, etc.
        return {
            "error": str(e),
            "base_name": base_name, 
            "ref_path": str(ref_path),
            "est_path": str(est_path)
        }
    
    
# ==========================================
# 4. Modular Pipeline Steps
# ==========================================

def load_cache(cache_path):
    """Loads existing JSON cache safely."""
    if cache_path.exists():
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print("[WARN] Cache file corrupted. Initializing fresh cache.")
            return {}
    return {}

def build_task_queue(maestro_dir, est_dir, master_cache, split="all", maestro_csv=None):
    """Matches ground truth files to predictions and filters out cached files."""
    maestro_dir = Path(maestro_dir)

    if split == "all":
        gt_files = list(maestro_dir.rglob("*.mid")) + list(maestro_dir.rglob("*.midi"))
    else:
        csv_path = Path(maestro_csv) if maestro_csv is not None else (maestro_dir / "maestro-v3.0.0.csv")
        if not csv_path.exists():
            raise FileNotFoundError(f"MAESTRO metadata CSV not found: {csv_path}")

        gt_files = []
        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("split") != split:
                    continue
                midi_rel = row.get("midi_filename")
                if not midi_rel:
                    continue
                midi_path = (maestro_dir / midi_rel).resolve()
                if midi_path.exists():
                    gt_files.append(midi_path)

    pending_pairs = []
    total_pairs_found = 0

    for gt in gt_files:
        base_name = gt.stem
        est_file = est_dir / f"{base_name}_pred.mid"
        
        if est_file.exists():
            total_pairs_found += 1
            if base_name not in master_cache:
                pending_pairs.append((gt, est_file))
                
    return pending_pairs, total_pairs_found

def process_evaluations(pending_pairs, workers, master_cache, cache_path, error_log_path):
    """Runs the multiprocessing pool and dynamically updates the cache."""
    print(f"[INFO] Launching evaluation pool for {len(pending_pairs)} remaining files...")
    errors = 0

    with Pool(workers) as p:
        for res in tqdm.tqdm(p.imap_unordered(evaluate_pair, pending_pairs), total=len(pending_pairs)):
            base_name = res.pop("base_name")

            if "error" in res:
                errors += 1
                err_msg = (
                    f"=== METRIC FAILURE ===\n"
                    f"Error : {res['error']}\n"
                    f"Ref   : {res.get('ref_path')}\n"
                    f"Est   : {res.get('est_path')}\n\n"
                )
                sys.stderr.write(err_msg)
                with open(error_log_path, "a") as f:
                    f.write(err_msg)
                continue
            
            # Save success to cache on-the-fly
            master_cache[base_name] = res
            with open(cache_path, 'w') as f:
                json.dump(master_cache, f, indent=4)

    print(f"[INFO] Processing complete. {errors} files failed during evaluation.")
    return errors

def print_summary(master_cache):
    """Calculates dataset-wide averages and prints the formatted table."""
    if not master_cache:
        print("[WARN] Cache is empty. No statistics to summarize.")
        return

    agg_metrics = defaultdict(list)
    for file_name, file_metrics in master_cache.items():
        for metric_name, (prec, rec, f1) in file_metrics.items():
            agg_metrics[metric_name].append((prec, rec, f1))

    print("\n" + "="*50)
    print("HOLISTIC MAESTRO EVALUATION RESULTS (AVERAGES)")
    print("="*50)
    
    with open(Path(__file__).parent / "metric_legend.txt", 'w') as f:
        for metric_name in sorted(agg_metrics.keys()):
            arr = np.array(agg_metrics[metric_name])
            mean_p = np.mean(arr[:, 0])
            mean_r = np.mean(arr[:, 1])
            mean_f1 = np.mean(arr[:, 2])
            
            print(f"{metric_name:<26} | P: {mean_p:.4f} | R: {mean_r:.4f} | F1: {mean_f1:.4f}")
            f.write(f"{metric_name:<26} | P: {mean_p:.4f} | R: {mean_r:.4f} | F1: {mean_f1:.4f}\n")

    print("-" * 50)
    print(f"Total Successfully Evaluated : {len(master_cache)}")
    print("=" * 50)


# ==========================================
# 5. Main Execution
# ==========================================
def main():
    print("[INFO] Starting 'compute_comparison_metrics.py' ...")

    # 1. Argument Parsing
    parser = argparse.ArgumentParser(description="Holistic dataset evaluator using mir_eval.")
    parser.add_argument("--maestro_dir", required=True, help="Path to MAESTRO root")
    parser.add_argument("--est_dir", required=True, help="Path to predicted MIDIs root folder")
    parser.add_argument("--workers", type=int, default=16, help="Number of CPU cores")
    parser.add_argument("--split", choices=["all", "train", "validation", "test"], default="test")
    parser.add_argument("--maestro-csv", default=None, help="Optional path to MAESTRO metadata CSV")
    args = parser.parse_args()

    # 2. Path Setup
    #output_dir = os.environ.get("OUTPUT_DIR", "/scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/output")
    output_dir = args.est_dir  # Use the same directory for outputs and logs to keep things organized
    error_log_path = Path(output_dir) / "metric_failures.log"
    cache_path = Path(output_dir) / "metrics_cache.json"
    est_dir = Path(args.est_dir) / "predicted_midis" 
    
    print(f"[INFO] Output dir : {output_dir}")
    print(f"[INFO] Error log  : {error_log_path}")

    # 3. Load State
    master_cache = load_cache(cache_path)

    # 4. Build Queue
    pending_pairs, total_pairs_found = build_task_queue(
        args.maestro_dir,
        est_dir,
        master_cache,
        split=args.split,
        maestro_csv=args.maestro_csv,
    )

    print(f"[INFO] Dataset total: {total_pairs_found} files.")
    print(f"[INFO] Cached previously: {len(master_cache)} files.")

    if total_pairs_found == 0:
        print("[ERROR] No pairs matched. Check your paths and naming conventions.")
        sys.exit(1)

    # 5. Process Pending Tasks
    if pending_pairs:
        process_evaluations(pending_pairs, args.workers, master_cache, cache_path, error_log_path)
    else:
        print("[INFO] All files are already cached. Skipping evaluation pool.")

    # 6. Summarize
    print_summary(master_cache)

if __name__ == "__main__":
    #python /scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/compute_comparison_metrics.py --maestro_dir /scratch/gilbreth/li5042/datasets/maestro_dataset --est_dir /scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/output
    main()