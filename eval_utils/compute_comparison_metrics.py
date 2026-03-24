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
import argparse
import numpy as np
import pretty_midi
import mir_eval
from pathlib import Path
from multiprocessing import Pool
import tqdm
from collections import defaultdict

# Suppress warnings for clean stdout
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
warnings.filterwarnings("ignore", message=".*not a valid type 0 or type 1 MIDI file.*")
sys.setrecursionlimit(10000)

# ==========================================
# 1. Extraction Helpers
# ==========================================
def extract_notes(pm):
    """Extracts note intervals, pitches (in Hz), and velocities."""
    intervals, pitches, vels = [], [], []
    for inst in pm.instruments:
        if not inst.is_drum:
            for note in inst.notes:
                intervals.append([note.start, note.end])
                pitches.append(note.pitch)
                vels.append(note.velocity)
                
    if not intervals:
        return np.empty((0, 2)), np.array([]), np.array([])
        
    intervals, pitches, vels = zip(*sorted(zip(intervals, pitches, vels), key=lambda x: x[0][0]))
    return np.array(intervals), mir_eval.util.midi_to_hz(np.array(pitches)), np.array(vels)

def extract_pedal(pm):
    """Extracts sustain pedal (CC 64) intervals and assigns a dummy pitch for mir_eval."""
    intervals = []
    for inst in pm.instruments:
        if not inst.is_drum:
            pedal_on = False
            start_time = 0.0
            for cc in inst.control_changes:
                if cc.number == 64:
                    if cc.value >= 64 and not pedal_on:
                        pedal_on = True
                        start_time = cc.time
                    elif cc.value < 64 and pedal_on:
                        pedal_on = False
                        intervals.append([start_time, cc.time])
            if pedal_on:
                intervals.append([start_time, pm.get_end_time()])
                
    if not intervals:
        return np.empty((0, 2)), np.array([])
        
    intervals = np.array(intervals)
    # Give all pedal intervals the exact same dummy pitch (e.g., 440Hz)
    pitches = np.ones(len(intervals)) * 440.0 
    return intervals, pitches

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
    
    try:
        ref_pm = pretty_midi.PrettyMIDI(str(ref_path))
        est_pm = pretty_midi.PrettyMIDI(str(est_path))
    except Exception as e:
        return {"error": f"Parse error: {e}"}

    results = {}

    # --- A. Note Extraction ---
    ref_inv, ref_p, ref_v = extract_notes(ref_pm)
    est_inv, est_p, est_v = extract_notes(est_pm)

    # 1. Note Onset
    p, r, f, _ = mir_eval.transcription.evaluate(ref_inv, ref_p, est_inv, est_p, onset_tolerance=0.05, offset_ratio=None)
    results['Note Onset'] = (p, r, f)

    # 2. Note Onset + Offset
    p, r, f, _ = mir_eval.transcription.evaluate(ref_inv, ref_p, est_inv, est_p, onset_tolerance=0.05, offset_ratio=0.2, offset_min_tolerance=0.05)
    results['Note Onset+Offset'] = (p, r, f)

    # 3. Note Onset + Offset + Velocity
    p, r, f, _ = mir_eval.transcription_velocity.evaluate(ref_inv, ref_p, ref_v, est_inv, est_p, est_v, onset_tolerance=0.05, offset_ratio=0.2, offset_min_tolerance=0.05)
    results['Note Onset+Offset+Velocity'] = (p, r, f)

    # 4. Activation (Frame-Level Pitch)
    fs = 100 # 10ms resolution
    ref_pr = (ref_pm.get_piano_roll(fs=fs) > 0)
    est_pr = (est_pm.get_piano_roll(fs=fs) > 0)
    
    # Pad to equal length
    max_len = max(ref_pr.shape[1], est_pr.shape[1])
    ref_pr = np.pad(ref_pr, ((0, 0), (0, max_len - ref_pr.shape[1])))
    est_pr = np.pad(est_pr, ((0, 0), (0, max_len - est_pr.shape[1])))
    results['Activation'] = calc_frame_metrics(ref_pr.flatten(), est_pr.flatten())

    # --- B. Pedal Extraction ---
    ref_ped_inv, ref_ped_p = extract_pedal(ref_pm)
    est_ped_inv, est_ped_p = extract_pedal(est_pm)

    if len(ref_ped_inv) > 0 or len(est_ped_inv) > 0:
        # 5. Pedal Onset
        p, r, f, _ = mir_eval.transcription.evaluate(ref_ped_inv, ref_ped_p, est_ped_inv, est_ped_p, onset_tolerance=0.05, offset_ratio=None)
        results['Pedal Onset'] = (p, r, f)

        # 6. Pedal Onset + Offset
        p, r, f, _ = mir_eval.transcription.evaluate(ref_ped_inv, ref_ped_p, est_ped_inv, est_ped_p, onset_tolerance=0.05, offset_ratio=0.2, offset_min_tolerance=0.05)
        results['Pedal Onset+Offset'] = (p, r, f)
        
        # 7. Pedal Activation (Frame-Level Pedal)
        # Create 1D boolean arrays representing pedal state over time
        ref_ped_state = np.zeros(max_len, dtype=bool)
        est_ped_state = np.zeros(max_len, dtype=bool)
        
        for start, end in ref_ped_inv:
            ref_ped_state[int(start * fs):int(end * fs)] = True
        for start, end in est_ped_inv:
            est_ped_state[int(start * fs):int(end * fs)] = True
            
        results['Pedal Activation'] = calc_frame_metrics(ref_ped_state, est_ped_state)
    else:
        results['Pedal Onset'] = (0,0,0)
        results['Pedal Onset+Offset'] = (0,0,0)
        results['Pedal Activation'] = (0,0,0)

    return results

# ==========================================
# 3. Main Multiprocessing Loop
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Holistic dataset evaluator using mir_eval.")
    parser.add_argument("--maestro_dir", required=True, help="Path to MAESTRO root (Ground Truths)")
    parser.add_argument("--est_dir", required=True, help="Path to predicted MIDIs folder")
    parser.add_argument("--workers", type=int, default=16, help="Number of CPU cores to use")
    args = parser.parse_args()

    gt_files = list(Path(args.maestro_dir).rglob("*.mid")) + list(Path(args.maestro_dir).rglob("*.midi"))
    est_dir = Path(args.est_dir) / "predicted_midis"  # Assuming predictions are in a subfolder named 'predicted_midis'
    
    pairs = []
    for gt in gt_files:
        # Assumes predictions are flattened or replicate the folder structure
        # Matches by filename stem (e.g. 'Song1.mid' matches 'Song1_pred.mid')
        base_name = gt.stem
        est_file = est_dir / f"{base_name}_pred.mid"
        
        if est_file.exists():
            pairs.append((gt, est_file))

    print(f"[INFO] Found {len(pairs)} matching Ground Truth / Prediction pairs.")
    if not pairs:
        print("[ERROR] No pairs matched. Check your --est_dir path and naming conventions.")
        sys.exit(1)

    print(f"[INFO] Launching evaluation pool with {args.workers} workers...")
    
    agg_metrics = defaultdict(list)
    errors = 0

    with Pool(args.workers) as p:
        for res in tqdm.tqdm(p.imap_unordered(evaluate_pair, pairs), total=len(pairs)):
            if "error" in res:
                errors += 1
                continue
            
            for metric_name, (prec, rec, f1) in res.items():
                agg_metrics[metric_name].append((prec, rec, f1))

    print("\n" + "="*50)
    print("HOLISTIC MAESTRO EVALUATION RESULTS (AVERAGES)")
    print("="*50)
    
    # Calculate and print means across the dataset
    for metric_name in sorted(agg_metrics.keys()):
        arr = np.array(agg_metrics[metric_name])
        mean_p = np.mean(arr[:, 0])
        mean_r = np.mean(arr[:, 1])
        mean_f1 = np.mean(arr[:, 2])
        
        print(f"{metric_name:<26} | P: {mean_p:.4f} | R: {mean_r:.4f} | F1: {mean_f1:.4f}")

    print("-" * 50)
    print(f"Total Files Evaluated : {len(pairs) - errors}")
    print(f"Failed to Parse       : {errors}")
    print("=" * 50)

if __name__ == "__main__":
    main()