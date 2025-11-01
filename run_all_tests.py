#!/usr/bin/env python3
"""
Script to run many TEST_INPUT lists (from the original conversation) and print pairwise similarities.

Note: This script does not run the model here — run locally after installing requirements.
Usage:
    python run_all_tests.py --model modern --pooling mean
"""
import argparse
from itertools import combinations
from typing import List
from embedding_evaluator import EmbeddingEvaluator, build_model_from_args

TEST_SETS = [
    ["breast cancer in males", "cancer breast male", "Malignant neoplasm of breast of unspecified site, male", "Malignant neoplasm of unspecified site of unspecified female breast"],
    ["bone xray", "xray bone"],
    ["closed", "closed in"],
    ["red", "magenta", "blue", "blue silver", "grey", "charcoal"],
    ["#ff0000", "#ff00ff", "#0000ff", "#c4d4e0", "#777777", "#36454F"],
    ["relentless coughing", "coughing"],
    ["keratoderma", "skin disease"],
    ["Right heart failure due to left heart failure", "heart failure", "right heart failure", "congestive heart failure"],
    ["diabetes type 2 without complications", "E11.9", "ckd", "htn", "heart failure"],
    ["diabetes type 2 without complications", "dm2"],
    ["htn", "hypertension"],
    ["always", "all the time"],
    ["diarrhea, diarrheal epidemic", "infectious gastroenteritis and colitis, unspecified", "diarrhea, unspecified", "diarrhea, diarrheal infectious", "fever", "heart failure", "diabetes"],
    ["recrudescent typhus (fever)", "typhus fever", "epidemic louse-borne typhus fever due to rickettsia prowazekii", "recrudescent typhus [brill's disease]", "typhus fever due to rickettsia typhi", "typhus fever due to rickettsia tsutsugamushi", "typhus fever, unspecified"],
    ["leishmaniasis tegumentaria diffusa", "leishmaniasis", "visceral leishmaniasis", "cutaneous leishmaniasis", "mucocutaneous leishmaniasis", "leishmaniasis, unspecified"],
    ["benign neoplasm of hepatic flexure", "Benign neoplasm of transverse colon", "Malignant neoplasm of hepatic flexure"],
    ["Actamin Maximum Strength", "Simethicone Chew Tab 80 MG", "Simethicone Tab 125 MG", "Acetaminophen Tab 325 MG", "Acetaminophen Tab 500 MG"],
    ["Actamin Maximum Strength", "Actamin_Maximum_Strength", "Maximum", "Strength", "Actamin"],
    ["male", "female"],
    ["fname", "f_name", "first_name", "first name"],
    ["fever", "cough", "chest pain", "burn of conjuctival sac", "Toxic effect of corrosive alkalis and alkali-like substances, accidental (unintentional)", "Lye causing toxic effect", "Alkaline chemical burn of cornea and conjunctival sac", "Chemical burn injury to conjunctiva", "Burning caused by caustic alkali", "Alkali burn of skin", "Alkaline chemical burn of cornea", "Chemical injury to cornea", "Accidental poisoning by caustic alkalis", "Toxic effect of potassium hydroxide", "Toxic effect of sodium hydroxide", "Toxic effect of corrosive alkalis and alk-like substnc, acc", "Accidental poisoning by sodium hydroxide", "Alkaline chemical burn of conjunctival sac", "Toxic effect of caustic alkali", "Burning caused by cement", "Alkaline chemical burn of cornea AND/OR conjunctival sac", "Toxic effect of corrosive alkalis and alkali-like substances, accidental (unintentional)", "Burning caused by lime", "Accidental poisoning by lye", "Burning caused by ammonia", "Burning caused by caustic oven cleaner", "Accidental burning caused by lye"],
]

def pairwise_list(strings: List[str]):
    for a, b in combinations(strings, 2):
        yield a, b

def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    ap.add_argument("--model", required=False, help="Registry model name (see embedding_evaluator.DEFAULT_MODEL_REGISTRY)")
    ap.add_argument("--model-path", required=False, help="Path or checkpoint for a SentenceTransformer model")
    ap.add_argument("--pooling", choices=["mean", "idf", "sif", "max", "sent"], default="mean")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lowercase", action="store_true")
    ap.add_argument("--no-normalize", action="store_true")
    ap.add_argument("--sif-a", type=float, default=1e-3)
    ap.add_argument("--remove-pc", type=int, default=1)
    args = ap.parse_args()

    model = build_model_from_args(args.model, args.model_path)
    ev = EmbeddingEvaluator(model=model, lowercase=args.lowercase, normalize=not args.no_normalize, batch_size=args.batch_size)
    print("Running test sets...")
    for idx, sset in enumerate(TEST_SETS):
        print(f"\n== Test set #{idx+1} (n={len(sset)}) ==")
        # Pre-encode all items for speed
        _ = ev.encode_sentences(sset)
        if args.pooling == "sif":
            # compute token probs for the set
            token_probs = ev.compute_token_probs(sset)
        else:
            token_probs = None
        for a, b in pairwise_list(sset):
            sim = ev.pairwise_similarity(a, b, pooling=args.pooling, token_probs=token_probs, sif_a=args.sif_a)
            print(f"{a!r} <-> {b!r} : {sim:.4f}")

if __name__ == "__main__":
    main()
