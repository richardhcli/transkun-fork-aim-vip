import os
import sys
import argparse
import subprocess
import json
import threading
import queue
from pathlib import Path
import tqdm
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed

class StateManager:
    """Thread-safe state tracker to prevent corruption and manage retries."""
    def __init__(self, state_file, max_attempts=2):
        self.state_file = Path(state_file)
        self.max_attempts = max_attempts
        self.lock = threading.Lock()
        self.state = {}
        self._load()

    def _load(self):
        """Loads existing state or initializes an empty dictionary."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    self.state = json.load(f)
            except json.JSONDecodeError:
                print("[WARN] State file corrupted. Creating a new one.")
                self.state = {}

    def _save(self):
        """Writes the current state to disk safely."""
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=4)

    def get_eligible_tasks(self, all_files):
        """
        Registers new files and filters out successes and max-failed files.
        Returns a list of files that need to be processed.
        """
        eligible = []
        with self.lock:
            for f in all_files:
                fs = str(f.resolve())
                
                # Register new files
                if fs not in self.state:
                    self.state[fs] = {"status": "PENDING", "attempts": 0}
                
                st = self.state[fs]
                
                # Skip if already successfully completed
                if st["status"] == "SUCCESS":
                    continue
                
                # Skip if it failed and has exhausted its retries
                if st["status"] == "FAILED" and st["attempts"] >= self.max_attempts:
                    continue
                
                # If PENDING, TRANSCRIBING (interrupted), or FAILED (with remaining attempts)
                eligible.append(f)
            
            self._save()
        return eligible

    def update_state(self, filepath, status, increment_attempt=False):
        """Thread-safe update of a file's state."""
        with self.lock:
            fs = str(filepath.resolve())
            self.state[fs]["status"] = status
            if increment_attempt:
                self.state[fs]["attempts"] += 1
            self._save()
    
    def get_summary(self):
        """Returns a summary of the current state counts."""
        summary = {"PENDING": 0, "TRANSCRIBING": 0, "SUCCESS": 0, "FAILED": 0}
        with self.lock:
            for st in self.state.values():
                summary[st["status"]] += 1
        return summary

def transcribe_file(wav_file, pred_file, gpu_queue, state_manager, error_log):
    """Worker function to transcribe a single file."""
    # 1. Update state to TRANSCRIBING and log the attempt
    state_manager.update_state(wav_file, "TRANSCRIBING", increment_attempt=True)
    
    # 2. Checkout an available GPU
    gpu_id = gpu_queue.get()
    device = f"cuda:{gpu_id}"
    
    command = [
        sys.executable,
        "-m",
        "transkun.transcribe",
        str(wav_file),
        str(pred_file),
        "--device",
        device,
    ]

    status = "SUCCESS"
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        # Explicitly update state to SUCCESS upon clean completion
        state_manager.update_state(wav_file, "SUCCESS")
    except subprocess.CalledProcessError as e:
        status = "FAILED"
        # Explicitly update state to FAILED upon crash
        state_manager.update_state(wav_file, "FAILED")
        
        # Log the exact error for debugging
        with open(error_log, "a") as f:
            f.write(f"=== CRASH: {wav_file.stem} on {device} ===\n")
            f.write(e.stderr if e.stderr else str(e))
            f.write("\n\n")
    finally:
        # 3. Always return the GPU to the queue, even if transcription crashed
        gpu_queue.put(gpu_id)
        
    return status

def main():
    parser = argparse.ArgumentParser(description="Stateful Multi-GPU Batch Transcribe")
    parser.add_argument("--maestro_dir", required=True, help="Path to MAESTRO dataset root")
    parser.add_argument("--output_dir", required=True, help="Directory to save predicted MIDIs")
    args = parser.parse_args()

    maestro_path = Path(args.maestro_dir)
    out_path = Path(args.output_dir)
    
    # WHAT: Create the root output directory for logs
    out_path.mkdir(parents=True, exist_ok=True)
    
    # WHAT: Create the dedicated subfolder for the actual MIDI files
    pred_midi_dir = out_path / "predicted_midis"
    pred_midi_dir.mkdir(parents=True, exist_ok=True)
    
    # Keep the logs securely in the root folder
    error_log = out_path / "transcription_errors.log"
    state_file = out_path / "transcription_state.json"

    # Auto-detect available GPUs assigned by Slurm
    num_gpus = torch.cuda.device_count()
    if num_gpus < 1:
        print("[ERROR] No GPUs detected by PyTorch. Check Slurm allocation.")
        sys.exit(1)

    print(f"[INFO] Detected {num_gpus} GPUs. Setting up worker queue...")
    gpu_queue = queue.Queue()
    for i in range(num_gpus):
        gpu_queue.put(i)

    # Initialize state manager allowing 1 initial attempt + 1 retry (max_attempts=2)
    state_manager = StateManager(state_file, max_attempts=2)

    print(f"[INFO] Scanning for .wav files in {maestro_path}...")
    wav_files = list(maestro_path.rglob("*.wav"))
    
    # Delegate task filtering to the State Manager
    tasks = []
    eligible_wavs = state_manager.get_eligible_tasks(wav_files)
    
    for wav_file in eligible_wavs:
        pred_file = pred_midi_dir / f"{wav_file.stem}_pred.mid"

        tasks.append((wav_file, pred_file))

    skip_count = len(wav_files) - len(tasks)
    print(f"[INFO] State Check: Skipping {skip_count} files (already SUCCESS or max FAILED).")
    
    if not tasks:
        print("[INFO] All files have been processed. ")
        summary = state_manager.get_summary()
        print(f"Summary: {summary['SUCCESS']} SUCCESS, {summary['FAILED']} FAILED, {summary['PENDING']} PENDING, {summary['TRANSCRIBING']} TRANSCRIBING")
        print("Exiting.")
        sys.exit(0)
        
    print(f"[INFO] Starting transcription of {len(tasks)} files across {num_gpus} GPUs...")

    success_count = 0
    fail_count = 0

    with ThreadPoolExecutor(max_workers=num_gpus) as executor:
        futures = {
            executor.submit(transcribe_file, wav, pred, gpu_queue, state_manager, error_log): wav 
            for wav, pred in tasks
        }
        
        for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Transcribing"):
            result = future.result()
            if result == "SUCCESS":
                success_count += 1
            else:
                fail_count += 1

    print("\n" + "="*40)
    print("STATEFUL MULTI-GPU TRANSCRIPTION COMPLETE")
    print("="*40)
    print(f"Successfully transcribed this run : {success_count}")
    print(f"Failed this run (check error log) : {fail_count}")
    print(f"State saved to : {state_file}")

if __name__ == "__main__":
    main()