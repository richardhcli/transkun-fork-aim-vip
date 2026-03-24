import os
import sys
import argparse
import subprocess
from pathlib import Path
import tqdm

def main():
    parser = argparse.ArgumentParser(
        description="Batch transcribe MAESTRO .wav files into MIDI using Transkun"
    )
    parser.add_argument(
        "--maestro_dir", 
        required=True, 
        help="Path to MAESTRO dataset root (e.g., /scratch/.../maestro_dataset)"
    )
    parser.add_argument(
        "--output_dir", 
        required=True, 
        help="Directory to save predicted MIDIs (e.g., ./output/predicted_midis)"
    )
    parser.add_argument(
        "--device", 
        default="cuda", 
        help="Device to use for inference (e.g., 'cuda' or 'cpu')"
    )

    args = parser.parse_args()

    maestro_path = Path(args.maestro_dir)
    out_path = Path(args.output_dir)
    
    # Create output directory if it doesn't exist
    out_path.mkdir(parents=True, exist_ok=True)
    error_log = out_path / "transcription_errors.log"

    print(f"[INFO] Scanning for .wav files in {maestro_path}...")
    # Recursively find all .wav files in the dataset folders (2004, 2006, etc.)
    wav_files = list(maestro_path.rglob("*.wav"))
    
    if not wav_files:
        print("[ERROR] No .wav files found. Check your --maestro_dir path.")
        sys.exit(1)
        
    print(f"[INFO] Found {len(wav_files)} audio files. Starting batch transcription...")

    success_count = 0
    skip_count = 0
    fail_count = 0

    # Initialize a progress bar to track the massive dataset
    for wav_file in tqdm.tqdm(wav_files, desc="Transcribing Dataset"):
        base_name = wav_file.stem
        
        # Name the output to perfectly match the requirement in evaluate_metrics.py
        pred_file = out_path / f"{base_name}_pred.mid"

        # Check if transcription already exists to prevent duplicate work
        if pred_file.exists():
            skip_count += 1
            continue

        # Build the command using the module syntax from your inspiration script
        command = [
            sys.executable,
            "-m",
            "transkun.transcribe",
            str(wav_file),
            str(pred_file),
            "--device",
            args.device,
        ]

        try:
            # Capture output so the terminal progress bar remains clean
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            success_count += 1
        except subprocess.CalledProcessError as e:
            fail_count += 1
            # Safely log errors to a file instead of breaking the loop
            with open(error_log, "a") as f:
                f.write(f"=== CRASH: {base_name} ===\n")
                f.write(e.stderr if e.stderr else str(e))
                f.write("\n\n")

    print("\n" + "="*40)
    print("BATCH TRANSCRIPTION COMPLETE")
    print("="*40)
    print(f"Successfully transcribed : {success_count}")
    print(f"Skipped (already exist)  : {skip_count}")
    print(f"Failed (check error log) : {fail_count}")

if __name__ == "__main__":
    main()