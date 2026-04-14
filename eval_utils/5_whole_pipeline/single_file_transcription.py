import argparse
import subprocess
import sys

#optional: 

MODEL_WEIGHT_PATH = "/scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/2_recreate_metrics/V2checkpointMSimpler/checkpoint.pt"
MODEL_CONF_PATH = "/scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/2_recreate_metrics/V2checkpointMSimpler/model.conf"


def single_file_transcription(wav_file: str, prediction_path: str, device: str = "cuda", checkpoint_pt: str = MODEL_WEIGHT_PATH, model_conf: str = MODEL_CONF_PATH):
    """Transcribes a single audio file into MIDI using Transkun."""

    command = [
        sys.executable,
        "-m",
        "transkun.transcribe",
        wav_file,
        prediction_path,
        "--device",
        device,
        "--weight",
        checkpoint_pt,
        "--conf",
        model_conf,
    ]

    # Run the transcription command
    try:
        print(f"[single_file_transcription] Running: {' '.join(command)}")
        
        #Removing capture_output=True and text=True
        #WHY: In the Python subprocess module, a stream value of None means the child process inherits the standard output and standard error streams of the parent process. This means Transkun's progress bars, warnings, and logs will print directly to your HPC terminal live, exactly as if you had typed the bash command yourself.
        subprocess.run(command, check=True) 
        
        print(f"\n[single_file_transcription] Transcription completed! Output saved to: {command[4]}")
        return 0
        
    except subprocess.CalledProcessError as e:
        print(f"\n[single_file_transcription] Transcription failed with exit code: {e.returncode}")
        return e
        #sys.exit(1)



def main():
    parser = argparse.ArgumentParser(
        description="Transcribe piano audio into MIDI using Transkun AMT"
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Path to input audio file",
    )
    parser.add_argument(
        "-o", "--output", required=True, help="Path to output MIDI file"
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to use for inference (e.g., 'cuda' or 'cpu')",
    )

    parser.add_argument(
        "--weight",
        default=MODEL_WEIGHT_PATH,
        help="Path to the pretrained model weight (default: transkunV2 checkpoint)",
    )

    parser.add_argument(
        "--conf",
        default=MODEL_CONF_PATH,
        help="Path to the model configuration file (default: transkunV2 config)",
    )

    args = parser.parse_args()

    # Build the transkun command
    # command = [
    #     sys.executable,
    #     "-m",
    #     "transkun.transcribe",
    #     args.input,
    #     args.output,
    #     "--device",
    #     args.device,
    #     "--weight",
    #     args.weight,
    #     "--conf",
    #     args.conf,
    # ]

    single_file_transcription(args.input, args.output, args.device)

if __name__ == "__main__":
    import os
    print("[single_file_transcription] Starting single file transcription test...")
    dataset = os.path.abspath("/scratch/gilbreth/li5042/datasets/MAESTRO")
    #midi_file = os.path.join(dataset, "2009/MIDI-Unprocessed_11_R1_2009_06-09_ORIG_MID--AUDIO_11_R1_2009_11_R1_2009_07_WAV.midi")
    audio_file = os.path.join(dataset, "2009/MIDI-Unprocessed_11_R1_2009_06-09_ORIG_MID--AUDIO_11_R1_2009_11_R1_2009_07_WAV.wav")
    output_path = "/scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/2_recreate_metrics/output/test_output_single_file_transcription.mid"

    print(f"[single_file_transcription] Testing with audio file: {audio_file}\n * Saving to path: {output_path}")
    try: 
        single_file_transcription(audio_file, output_path)
    except Exception as e:
        print(f"[single_file_transcription] Error during transcription: {e}")