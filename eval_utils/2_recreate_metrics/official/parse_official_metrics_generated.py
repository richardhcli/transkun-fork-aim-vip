import json
import argparse

def main():
    parser = argparse.ArgumentParser(description="Parse Transkun JSON and print a clean summary table.")
    parser.add_argument("json_file", help="Path to the output JSON from Transkun evaluation")
    args = parser.parse_args()

    try:
        with open(args.json_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] Could not find {args.json_file}")
        return

    agg = data.get("aggregated", {})
    if not agg:
        print("[ERROR] No 'aggregated' data found in JSON. Did the evaluation complete successfully?")
        return

    # Map Transkun's internal code variables to the paper's official terminology
    key_map = {
        "frame": "Activation",
        "note": "Note Onset",
        "note+offset": "Note Onset+Offset",
        "note+velocity+offset": "Note Onset+Offset+ vel.",
        "pedal64frame": "Pedal Activation",
        "pedal64": "Pedal Onset",
        "pedal64+offset": "Pedal Onset+Offset"
    }

    output = ""

    print("\n" + "="*65)
    print("TRANSKUN AGGREGATED SUMMARY STATISTICS (From JSON)")
    print("="*65)
    print(f"{'Metric':<28} | {'Precision':<9} | {'Recall':<9} | {'F1-Score':<9}")
    print("-" * 65)

    output += "\n" + "="*65 + "\n"
    output += "TRANSKUN AGGREGATED SUMMARY STATISTICS (From JSON)"
    output += "\n" + "="*65 + "\n"
    output += f"{'Metric':<28} | {'Precision':<9} | {'Recall':<9} | {'F1-Score':<9}\n"
    output += "-" * 65 + "\n"


    # Order the keys exactly how they appear in the GitHub markdown table
    ordered_keys = [
        "frame", 
        "note", 
        "note+offset", 
        "note+velocity+offset", 
        "pedal64frame", 
        "pedal64", 
        "pedal64+offset"
    ]


    for k in ordered_keys:
        if k in agg:
            vals = agg[k]
            name = key_map.get(k, k)
            
            # Transkun arrays are [Precision, Recall, F1-Score, Overlap]
            p, r, f1 = vals[0], vals[1], vals[2]
            print(f"{name:<28} | {p:.4f}    | {r:.4f}    | {f1:.4f}")
            output += f"{name:<28} | {p:.4f}    | {r:.4f}    | {f1:.4f}\n"

    print("="*65 + "\n")



    #save to file
    with open("/scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/2_recreate_metrics/official/output/transkun_paper_metrics.txt", "w") as f:
        f.write(output)

if __name__ == "__main__":
    #python /scratch/gilbreth/li5042/transkun/transkun_fork/eval_utils/2_recreate_metrics/official/parse_official_metrics_generated.py /scratch/gilbreth/li5042/transkun/transkun_fork/transkun/transkun_paper_metrics.json
    main()