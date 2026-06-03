"""Analyze listening test results.

Input: rater CSV with columns:
  rater_id, trial_id, choice  (one of: A, B, same, or float for MOS)

Computes:
  - Per-trial: pairwise accuracy (correct vs random vs wrong)
  - Aggregate: % pairwise correct (concept-presence effect)
  - MOS quality means (steered vs baseline)
  - Fleiss' kappa for inter-rater agreement
"""
import os, sys, json, argparse
import numpy as np
import pandas as pd


def fleiss_kappa(ratings):
    """ratings: matrix [n_items, n_categories] of counts."""
    n_items, n_cat = ratings.shape
    n_raters = ratings.sum(axis=1).max()
    P_e = (ratings.sum(axis=0) / (n_items * n_raters)) ** 2
    P_e = P_e.sum()
    P_i = (ratings ** 2).sum(axis=1) - n_raters
    P_i = P_i / (n_raters * (n_raters - 1))
    P_bar = P_i.mean()
    return (P_bar - P_e) / (1 - P_e + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses_csv", required=True,
                    help="CSV: rater_id, trial_id, choice")
    ap.add_argument("--metadata_json", required=True,
                    help="stimuli_metadata.json from prepare_stimuli.py")
    ap.add_argument("--out_path", default="/scratch2/solbon1212/music-mechinterp/listening_test/results.json")
    args = ap.parse_args()

    df = pd.read_csv(args.responses_csv)
    meta = json.load(open(args.metadata_json))
    md = {t["trial_id"]: t for t in meta["trials"]}

    print(f"Total responses: {len(df)}")
    print(f"Unique raters:   {df.rater_id.nunique()}")
    print(f"Unique trials:   {df.trial_id.nunique()}")

    # --- Pairwise accuracy ---
    pairs = df[df.choice.isin(["A", "B", "same"])].copy()
    pairs["expected"] = pairs.trial_id.map(lambda t: md[t]["expected_correct"])
    pairs["concept"]  = pairs.trial_id.map(lambda t: md[t]["target_concept"])
    pairs["type"]     = pairs.trial_id.map(lambda t: md[t]["type"])

    pairs["correct"] = (pairs.choice == pairs.expected).astype(int)
    pairs["same"]    = (pairs.choice == "same").astype(int)

    per_concept = pairs.groupby(["type", "concept"]).agg(
        n=("correct", "size"),
        correct_rate=("correct", "mean"),
        same_rate=("same", "mean"),
    ).reset_index()
    print("\n=== Per (type, concept) pairwise accuracy ===")
    print(per_concept.to_string(index=False))

    # Overall pairwise summary by type
    type_agg = pairs.groupby("type").agg(
        n=("correct", "size"),
        correct_rate=("correct", "mean"),
        same_rate=("same", "mean"),
    ).reset_index()
    print("\n=== Pairwise by type ===")
    print(type_agg.to_string(index=False))

    # Overall pairwise
    overall_correct = pairs.correct.mean()
    print(f"\nOverall pairwise correct: {overall_correct*100:.1f}%  "
          f"(chance=50% w/o 'same'; or ~33% with 'same' as wrong)")

    # --- Inter-rater agreement (per trial) ---
    # Build counts matrix
    trial_ids = sorted(pairs.trial_id.unique())
    cats = ["A", "B", "same"]
    R = np.zeros((len(trial_ids), len(cats)), dtype=int)
    for i, tid in enumerate(trial_ids):
        sub = pairs[pairs.trial_id == tid]
        for j, c in enumerate(cats):
            R[i, j] = (sub.choice == c).sum()
    kappa = fleiss_kappa(R)
    print(f"\nFleiss κ (pairwise trials): {kappa:.3f}")

    # --- MOS quality ---
    mos = df[~df.choice.isin(["A", "B", "same"])].copy()
    if len(mos) > 0:
        mos["score"] = mos.choice.astype(float)
        mos["source"] = mos.trial_id.map(lambda t: md[t]["A_source"])
        mos_agg = mos.groupby("source").agg(
            n=("score", "size"),
            mean=("score", "mean"),
            std=("score", "std"),
        ).reset_index()
        print("\n=== MOS quality ===")
        print(mos_agg.to_string(index=False))

    # Save full results
    out = {
        "n_raters": int(df.rater_id.nunique()),
        "overall_pairwise_correct_rate": float(overall_correct),
        "fleiss_kappa": float(kappa),
        "per_concept": per_concept.to_dict(orient="records"),
        "per_type": type_agg.to_dict(orient="records"),
    }
    if len(mos) > 0:
        out["mos"] = mos_agg.to_dict(orient="records")
    with open(args.out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {args.out_path}")


if __name__ == "__main__":
    main()
