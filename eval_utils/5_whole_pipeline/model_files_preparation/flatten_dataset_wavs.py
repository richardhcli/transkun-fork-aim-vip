import argparse
import json
import logging
from pathlib import Path
import traceback
import scipy.io.wavfile as wavfile
import numpy as np
import shutil

def flatten_dataset_wavs(dataset_dir, output_json, output_log):
    dataset_dir = Path(dataset_dir)
    output_json = Path(output_json)
    output_log = Path(output_log)
    
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(output_log),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"Scanning dataset directory: {dataset_dir}")
    wav_files = list(dataset_dir.rglob('*.wav'))
    logging.info(f"Found {len(wav_files)} .wav files.")
    
    results = []
    
    for wav_path in wav_files:
        try:
            fs, data = wavfile.read(wav_path, mmap=False)
            
            # Check if it's stereo (or more channels)
            if len(data.shape) > 1 and data.shape[1] > 1:
                logging.info(f"Flattening {wav_path.name} ({data.shape[1]} channels)...")
                
                # Backup
                backup_path = wav_path.with_suffix('.wav.backup')
                if not backup_path.exists():
                    shutil.copy2(wav_path, backup_path)
                
                # Ensure write permissions
                try:
                    wav_path.chmod(wav_path.stat().st_mode | 0o200)
                except Exception as e:
                    logging.warning(f"Could not chmod {wav_path.name}: {e}")
                
                # Downmix to mono by averaging channels
                mono_data = np.mean(data, axis=1).astype(data.dtype)
                
                # Write back to the same file
                wav_path.unlink() # Delete to safely overwrite read-only files if chmod couldn't
                wavfile.write(str(wav_path), fs, mono_data)
                
                # Verify
                _, verify_data = wavfile.read(wav_path, mmap=False)
                if len(verify_data.shape) > 1 and verify_data.shape[1] > 1:
                    raise RuntimeError(f"Verification failed: {wav_path.name} still has {verify_data.shape[1]} channels")
                
                logging.info(f"Successfully flattened {wav_path.name}.")
                results.append({
                    "path": str(wav_path),
                    "status": "modified"
                })
            else:
                results.append({
                    "path": str(wav_path),
                    "status": "skipped"
                })
        except Exception as e:
            logging.error(f"Failed during processing {wav_path.name}: {e}")
            logging.error(traceback.format_exc())
            results.append({
                "path": str(wav_path),
                "status": "error",
                "error": str(e)
            })
            
            # Try to restore backup if it was modified
            backup_path = wav_path.with_suffix('.wav.backup')
            if backup_path.exists():
                logging.info(f"Restoring backup {backup_path.name} -> {wav_path.name}")
                try:
                    shutil.copy2(backup_path, wav_path)
                except Exception as restore_e:
                    logging.error(f"Failed to restore backup for {wav_path.name}: {restore_e}")

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-log", required=True)
    args = parser.parse_args()
    
    flatten_dataset_wavs(args.dataset_dir, args.output_json, args.output_log)
