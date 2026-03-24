import argparse
import subprocess
import sys


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

    args = parser.parse_args()

    # Build the transkun command
    command = [
        sys.executable,
        "-m",
        "transkun.transcribe",
        args.input,
        args.output,
        "--device",
        args.device,
    ]

    # Run the transcription command
    try:
        print(f"Running: {' '.join(command)}")
        subprocess.run(command, check=True)
        print(f"\nTranscription completed! Output saved to: {args.output}")
    except subprocess.CalledProcessError as e:
        print(f"\nTranscription failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()