import os
import sys
import argparse
import subprocess
import csv
from pathlib import Path
import shutil

def main():
    parser = argparse.ArgumentParser(description="Wrapper to run Transkun's official evaluation script.")
    parser.add_argument("--maestro_dir", required=True, help="Path to MAESTRO root")
    parser.add_argument("--est_dir", required=True, help="Path to your flattened predicted_midis folder")
    parser.add_argument("--workers", type=int, default=16, help="Number of workers for evaluation")
    parser.add_argument("--output_json", default="official_transkun_metrics.json", help="Output JSON file")
    args = parser.parse_args()

    maestro_dir = Path(args.maestro_dir).resolve()
    est_dir = Path(args.est_dir).resolve()
    
    # Create the temporary shadow directory
    shadow_dir = Path("./shadow_eval_dir").resolve()
    if shadow_dir.exists():
        shutil.rmtree(shadow_dir)
    shadow_dir.mkdir(parents=True)

    print("[INFO] Building Symlink Shadow Directory...")
    
    # Build ground-truth list from MAESTRO metadata and keep only the test split.
    MAESTRO_DIR_CSV = maestro_dir / "maestro-v3.0.0.csv"
    if not MAESTRO_DIR_CSV.exists():
        print("[ERROR] Could not find MAESTRO metadata CSV in maestro_dir. Tried:")
        print(f"  - {MAESTRO_DIR_CSV}")
        sys.exit(1)

    gt_files = []
    with MAESTRO_DIR_CSV.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("split") == "test":
                midi_rel = row.get("midi_filename")
                if not midi_rel:
                    continue
                gt_path = (maestro_dir / midi_rel).resolve()
                if gt_path.exists():
                    gt_files.append(gt_path)
    print(f"[INFO] Loaded {len(gt_files)} test-split MIDI files from {MAESTRO_DIR_CSV.name}")
    
    # Create symlinks for each ground truth file
    links_created = 0
    missing_preds = 0

    for gt in gt_files:
        # Expected prediction filename from our batch transcribe script
        expected_pred_name = f"{gt.stem}_pred.mid"
        pred_file = est_dir / expected_pred_name
        
        if pred_file.exists():
            # Figure out the relative path (e.g., "2004/MIDI-Unprocessed_...midi")
            rel_path = gt.relative_to(maestro_dir)
            shadow_dest = shadow_dir / rel_path
            
            # Create the necessary subfolders inside the shadow directory
            shadow_dest.parent.mkdir(parents=True, exist_ok=True)
            
            # Create a symlink so the prediction mimics the exact GT filename and extension
            os.symlink(pred_file, shadow_dest)
            links_created += 1
        else:
            missing_preds += 1

    print(f"[INFO] Created {links_created} symlinks. Missing {missing_preds} predictions.")
    
    if links_created == 0:
        print("[ERROR] No matching predictions found. Exiting.")
        shutil.rmtree(shadow_dir)
        sys.exit(1)

    print("\n[INFO] Launching Transkun Official Evaluation...")
    
    # Construct the command to run your patched Transkun fork
    command = [
        sys.executable,
        "/scratch/gilbreth/li5042/transkun/transkun_fork/transkun/computeMetrics.py",  # Using your patched fork to avoid the KeyError
        str(shadow_dir),          # The perfectly mirrored predictions
        str(maestro_dir),         # The ground truth
        "--nProcess", str(args.workers),
        "--outputJSON", args.output_json
    ]

    try:
        # Run the evaluation
        subprocess.run(command, check=True)
        print(f"\n[SUCCESS] Evaluation complete! Detailed metrics saved to: {args.output_json}")
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Transkun evaluation crashed: {e}")
    finally:
        # Clean up the shadow directory to save clutter
        print("[INFO] Cleaning up shadow directory...")
        shutil.rmtree(shadow_dir)

if __name__ == "__main__":
    main()