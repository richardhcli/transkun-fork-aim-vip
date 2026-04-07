#Before trying to transcribe anything, 

import sys
import argparse
from pathlib import Path
import torch
import moduleconf
import time

#python /scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/3_retrain_model_metrics/verify_training_artifacts.py

print(f"[{__file__}:{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting Transkun Training Artifact Validation Script...")

# Import Transkun's native I/O helpers directly
try:
    from transkun.transcribe import readAudio, writeMidi
except ImportError:
    print("[ERROR] Ensure the transkun environment is activated.")
    sys.exit(1)

# ==========================================
# Global Execution Control
# ==========================================
IS_MANUAL_CALL = True  # Set to False to use command-line arguments

# Hardcoded defaults for manual testing
DEFAULT_PT_FILE = "/scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/3_retrain_model_metrics/transkun_checkpoints/checkpoint_baseline.pt"
DEFAULT_CONF_FILE = "/scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/3_retrain_model_metrics/output/MAESTRO_METADATA/transkun_base.json"
DEFAULT_MAESTRO_DIR = "/scratch/gilbreth/li5042/datasets/MAESTRO/" # Adjust this if your path differs
DEFAULT_OUTPUT = "/scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/3_retrain_model_metrics/output/validation_test.mid"
DEFAULT_DEVICE = "gpu" if torch.cuda.is_available() else "cpu"

# ==========================================
# Core Validation Logic
# ==========================================
def run_validation(weight_path, conf_path, maestro_dir, output_path, device):
    """Executes the end-to-end validation protocol."""
    print("\n" + "="*55)
    print("TRANSKUN TRAINING VALIDATION PROTOCOL")
    print("="*55)

    # 1. Auto-Locate a Test File
    print(f"[INFO] Scanning {maestro_dir} for a test .wav file...")
    test_wav = next(Path(maestro_dir).rglob("*.wav"), None)
    if not test_wav:
        print(f"[ERROR] Could not find any .wav files in {maestro_dir}")
        sys.exit(1)
    print(f"[SUCCESS] Found test audio: {test_wav.name}")

    # 2. Structural & Config Check
    print(f"\n[INFO] Parsing config: {conf_path}")
    try:
        confManager = moduleconf.parseFromFile(conf_path)
        TransKun_Class = confManager["Model"].module.TransKun
        conf = confManager["Model"].config
        print("[SUCCESS] Config parsed via moduleconf.")
    except Exception as e:
        print(f"[ERROR] Failed to parse config: {e}")
        sys.exit(1)

    # 3. Checkpoint Loading & Architecture Mapping
    print(f"\n[INFO] Loading checkpoint: {weight_path}")
    try:
        checkpoint = torch.load(weight_path, map_location=device, weights_only=True)
        model = TransKun_Class(conf=conf).to(device)
        
        # Exact logic from transkun's transcribe.py
        if "best_state_dict" in checkpoint:
            print("[INFO] Found 'best_state_dict'. Loading optimal weights...")
            model.load_state_dict(checkpoint["best_state_dict"], strict=False)
        elif "state_dict" in checkpoint:
            print("[INFO] Found 'state_dict'. Loading current weights...")
            model.load_state_dict(checkpoint["state_dict"], strict=False)
        else:
            print("[INFO] Loading raw checkpoint as state dict...")
            model.load_state_dict(checkpoint, strict=False)
            
        model.eval()
        torch.set_grad_enabled(False)
        print("[SUCCESS] Weights successfully mapped to architecture.")
        
    except Exception as e:
        print(f"[ERROR] Model instantiation or weight mapping failed: {e}")
        sys.exit(1)

    # 4. Audio Processing & Inference
    print("\n[INFO] Commencing end-to-end inference test...")
    try:
        # Read audio using Transkun's native helper
        fs, audio = readAudio(str(test_wav))
        
        # Resample if dataset frequency doesn't match model config
        if fs != model.fs:
            print(f"[INFO] Resampling audio from {fs}Hz to {model.fs}Hz...")
            import soxr
            audio = soxr.resample(audio, fs, model.fs)
        
        x = torch.from_numpy(audio).to(device)
        
        # Transcribe
        print("[INFO] Transcribing... (this may take a minute on CPU)")
        notesEst = model.transcribe(x, discardSecondHalf=False)
        
        # Write output using Transkun's native helper
        outputMidi = writeMidi(notesEst)
        outputMidi.write(output_path)
        
        print(f"\n[SUCCESS] ✨ Valid MIDI generated at: {output_path} ✨")
        print("[SUCCESS] Your training artifacts are fully verified and ready for deployment!")
        
    except Exception as e:
        print(f"\n[ERROR] Inference crashed: {e}")
        sys.exit(1)

# ==========================================
# Execution Routing
# ==========================================
def main():
    if IS_MANUAL_CALL:
        print("[INFO] IS_MANUAL_CALL is True. Running with hardcoded defaults.")
        run_validation(
            weight_path=DEFAULT_PT_FILE,
            conf_path=DEFAULT_CONF_FILE,
            maestro_dir=DEFAULT_MAESTRO_DIR,
            output_path=DEFAULT_OUTPUT,
            device=DEFAULT_DEVICE
        )
    else:
        # Standard CLI routing for Slurm/Bash usage
        parser = argparse.ArgumentParser(description="End-to-End Transkun Artifact Validation")
        parser.add_argument("--weight", required=True, help="Path to your trained .pt checkpoint")
        parser.add_argument("--conf", required=True, help="Path to your model .json/.conf")
        parser.add_argument("--maestro_dir", required=True, help="Path to MAESTRO root to grab a test wav")
        parser.add_argument("--output", default="validation_test.mid", help="Output MIDI file name")
        parser.add_argument("--device", default="cpu", help="Device to run on (cpu or cuda)")
        args = parser.parse_args()

        run_validation(
            weight_path=args.weight,
            conf_path=args.conf,
            maestro_dir=args.maestro_dir,
            output_path=args.output,
            device=args.device
        )

if __name__ == "__main__":
    main()