#!/usr/bin/env bash
# Creates the embedding-evaluator project files and packages them into embedding-evaluator.zip
# Usage:
#   chmod +x create_project_and_zip.sh
#   ./create_project_and_zip.sh
#
# The script will:
#  - create files and directories
#  - run `zip -r embedding-evaluator.zip embedding-evaluator` to produce the archive
#  - also write a base64 version to embedding-evaluator.zip.b64
#
# Requires: zip, base64 (commonly available on macOS/Linux). On Windows use WSL or Git Bash.

set -e
ROOT_DIR="embedding-evaluator"
rm -rf "$ROOT_DIR"
mkdir -p "$ROOT_DIR"

# embedding_evaluator.py
cat > "$ROOT_DIR/embedding_evaluator.py" <<'EOF'
#!/usr/bin/env python3
"""
Embedding evaluator with multiple pooling strategies (mean, idf, sif, max, sent),
caching, batch encoding, and a model registry.

Usage (CLI):
    python embedding_evaluator.py --model modern --pooling idf --sts-file example_sts.tsv

Library usage example:
    from sentence_transformers import SentenceTransformer
    from embedding_evaluator import EmbeddingEvaluator
    model = SentenceTransformer("all-MiniLM-L6-v2")
    ev = EmbeddingEvaluator(model)
    pairs = [("a","b",1.0)]
    res = ev.evaluate(pairs, pooling="mean")
"""
from __future__ import annotations
import argparse
import math
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None  # type: ignore

_TOKEN_RE = re.compile(r"[A-Za-z0-9#@_\-]+", flags=re.UNICODE)


def simple_tokenize(text: str, lowercase: bool = True) -> List[str]:
    t = text.lower() if lowercase else text
    return _TOKEN_RE.findall(t)


@dataclass
class EmbeddingEvaluator:
    model: object
    lowercase: bool = True
    normalize: bool = True
    batch_size: int = 64
    cache: Dict[str, np.ndarray] = field(default_factory=dict)

    def _key_for_token(self, token: str) -> str:
        return f"tok:{token}"

    def _key_for_sentence(self, sentence: str) -> str:
        return f"sent:{sentence}"

    def _prepare_text(self, text: str) -> str:
        return text.strip().lower() if self.lowercase else text.strip()

    def _encode_batch(self, texts: Sequence[str]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("No model set for EmbeddingEvaluator.")
        try:
            emb = self.model.encode(list(texts), convert_to_numpy=True, show_progress_bar=False)
        except TypeError:
            emb = np.asarray(self.model.encode(list(texts), show_progress_bar=False))
        emb = np.asarray(emb)
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)
        return emb

    def _safe_normalize(self, v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        if math.isfinite(n) and n > 0:
            return v / (n + 1e-12)
        return v

    def _batch_encode_and_cache(self, keys: Sequence[str], getter: callable, key_prefix: str):
        prepared = [getter(k) for k in keys]
        unseen = [p for p in prepared if (key_prefix + p) not in self.cache]
        if not unseen:
            return
        encoded = []
        for i in range(0, len(unseen), self.batch_size):
            batch = unseen[i : i + self.batch_size]
            emb = self._encode_batch(batch)
            encoded.extend(emb)
        for txt, e in zip(unseen, encoded):
            if self.normalize:
                e = self._safe_normalize(e)
            self.cache[key_prefix + txt] = e

    def encode_sentences(self, sentences: Iterable[str]) -> List[np.ndarray]:
        prepared = [self._prepare_text(s) for s in sentences]
        self._batch_encode_and_cache(prepared, lambda x: x, "sent:")
        return [self.cache["sent:" + p] for p in prepared]

    def encode_tokens(self, tokens: Iterable[str]) -> List[np.ndarray]:
        prepared = [t.lower() if self.lowercase else t for t in tokens]
        self._batch_encode_and_cache(prepared, lambda x: x, "tok:")
        return [self.cache["tok:" + p] for p in prepared]

    def get_sentence_embedding(self, sentence: str) -> np.ndarray:
        return self.encode_sentences([sentence])[0]

    def get_token_embedding(self, token: str) -> np.ndarray:
        return self.encode_tokens([token])[0]

    def compute_idf(self, sentences: Sequence[str]) -> Dict[str, float]:
        N = 0
        df: Dict[str, int] = {}
        for s in sentences:
            N += 1
            toks = set(simple_tokenize(s, lowercase=self.lowercase))
            for t in toks:
                df[t] = df.get(t, 0) + 1
        idf: Dict[str, float] = {}
        for t, cnt in df.items():
            idf[t] = max(0.0, math.log((N + 1) / (cnt + 1)))
        return idf

    def compute_token_probs(self, sentences: Sequence[str]) -> Dict[str, float]:
        freq: Dict[str, int] = {}
        total = 0
        for s in sentences:
            toks = simple_tokenize(s, lowercase=self.lowercase)
            for t in toks:
                freq[t] = freq.get(t, 0) + 1
                total += 1
        probs: Dict[str, float] = {}
        for t, c in freq.items():
            probs[t] = c / total if total > 0 else 0.0
        return probs

    def sentence_vector(
        self,
        sentence: str,
        pooling: str = "mean",
        idf: Optional[Dict[str, float]] = None,
        token_probs: Optional[Dict[str, float]] = None,
        sif_a: float = 1e-3,
    ) -> Optional[np.ndarray]:
        toks = simple_tokenize(sentence, lowercase=self.lowercase)
        if not toks:
            return None
        uniq = list(dict.fromkeys(toks))
        self._batch_encode_and_cache(uniq, lambda x: x, "tok:")
        vecs = []
        weights = []
        for t in toks:
            key = "tok:" + (t.lower() if self.lowercase else t)
            if key not in self.cache:
                continue
            vec = self.cache[key]
            vecs.append(vec)
            if pooling == "idf":
                w = idf.get(t, 1.0) if idf is not None else 1.0
            elif pooling == "sif":
                p = token_probs.get(t, 1e-8) if token_probs is not None else 1e-8
                w = sif_a / (sif_a + p)
            else:
                w = 1.0
            weights.append(w)
        if not vecs:
            return None
        mat = np.vstack(vecs)
        w = np.array(weights, dtype=float).reshape(-1, 1)
        if pooling == "max":
            return np.max(mat, axis=0)
        sent = (mat * w).sum(axis=0) / (w.sum() + 1e-12)
        return sent

    def remove_pc(self, X: np.ndarray, npc: int = 1) -> np.ndarray:
        if X.ndim != 2:
            raise ValueError("X must be 2D array")
        try:
            u, s, vh = np.linalg.svd(X, full_matrices=False)
        except np.linalg.LinAlgError:
            return X
        pc = vh[:npc]
        X_processed = X.copy()
        for i in range(npc):
            comp = pc[i].reshape(1, -1)
            proj = X_processed.dot(comp.T) * comp
            X_processed = X_processed - proj
        return X_processed

    def pairwise_similarity(self, s1: str, s2: str, pooling: str = "mean", **pool_kwargs) -> float:
        if pooling == "sent":
            e1 = self.get_sentence_embedding(s1)
            e2 = self.get_sentence_embedding(s2)
        else:
            e1 = self.sentence_vector(s1, pooling=pooling, **pool_kwargs)
            e2 = self.sentence_vector(s2, pooling=pooling, **pool_kwargs)

        if e1 is None or e2 is None:
            return 0.0
        if self.normalize:
            e1 = self._safe_normalize(e1)
            e2 = self._safe_normalize(e2)
        denom = np.linalg.norm(e1) * np.linalg.norm(e2)
        if denom == 0:
            return 0.0
        return float(np.dot(e1, e2) / (denom + 1e-12))

    def evaluate(
        self,
        pairs: Sequence[Tuple[str, str, float]],
        pooling: str = "mean",
        progress: bool = True,
        sif_a: float = 1e-3,
        remove_pc_n: int = 1,
    ) -> Dict[str, float]:
        sentences = []
        for s1, s2, _ in pairs:
            sentences.append(s1)
            sentences.append(s2)

        idf = None
        token_probs = None
        if pooling == "idf":
            idf = self.compute_idf(sentences)
        if pooling == "sif":
            token_probs = self.compute_token_probs(sentences)

        y_true = []
        y_pred = []

        if pooling == "sif":
            svecs = []
            for s in tqdm(sentences, desc="Computing SIF sentence vectors") if progress else sentences:
                sv = self.sentence_vector(s, pooling="sif", token_probs=token_probs, sif_a=sif_a)
                if sv is None:
                    # Zero vector fallback
                    dim = self.model.get_sentence_embedding_dimension() if hasattr(self.model, "get_sentence_embedding_dimension") else None
                    if dim is None:
                        # try to infer dim from a cached token or from model.encode of a tiny sample
                        try:
                            sample = self._encode_batch([""])
                            dim = sample.shape[-1]
                        except Exception:
                            dim = 768
                    sv = np.zeros(dim, dtype=float)
                svecs.append(sv)
            svecs = np.vstack(svecs)
            svecs = self.remove_pc(svecs, npc=remove_pc_n)
            idx = 0
            for i in range(0, len(svecs), 2):
                v1 = svecs[i]
                v2 = svecs[i + 1]
                if self.normalize:
                    v1 = self._safe_normalize(v1)
                    v2 = self._safe_normalize(v2)
                sim = float(np.dot(v1, v2))
                y_pred.append(sim)
                y_true.append(float(pairs[idx][2]))
                idx += 1
        else:
            iterator = tqdm(pairs, desc="Computing similarities") if progress else pairs
            for s1, s2, score in iterator:
                sim = self.pairwise_similarity(s1, s2, pooling=pooling, idf=(idf if pooling=="idf" else None), token_probs=(token_probs if pooling=="sif" else None), sif_a=sif_a)
                y_true.append(float(score))
                y_pred.append(float(sim))

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        if len(y_true) == 0:
            raise ValueError("No valid pairs to evaluate")
        pearson_r, pearson_p = pearsonr(y_true, y_pred)
        spearman_rho, spearman_p = spearmanr(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        return {
            "pearson_r": float(pearson_r),
            "pearson_p": float(pearson_p),
            "spearman_rho": float(spearman_rho),
            "spearman_p": float(spearman_p),
            "mse": float(mse),
            "n_pairs": int(len(y_true)),
        }


DEFAULT_MODEL_REGISTRY: Dict[str, Dict[str, object]] = {
    # registry items collected from your earlier list and some common checkpoints:
    "modern": {"checkpoint": "nomic-ai/modernbert-embed-base"},
    "stella_v1": {"checkpoint": "dunzhang/stella_en_400M_v5"},
    "stella_jina": {"checkpoint": "jinaai/jina-embeddings-v3"},
    "biomodel": {"checkpoint": "bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12"},
    "uae": {"checkpoint": "WhereIsAI/UAE-Large-V1"},
    "ada2": {"checkpoint": "Xenova/text-embedding-ada-002"},
    "biobert": {"checkpoint": "dmis-lab/biobert-base-cased-v1.2"},
    "bioclinicalbert": {"checkpoint": "emilyalsentzer/Bio_ClinicalBERT"},
    "biomedbert": {"checkpoint": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"},
    "mistral_e5": {"checkpoint": "intfloat/e5-mistral-7b-instruct"},
    "gary": {"checkpoint": "garyw/clinical-embeddings-300d-ft-cr"},
    "gatortron": {"checkpoint": "UFNLP/gatortron-base"},
    "medembed": {"checkpoint": "abhinand/MedEmbed-large-v0.1"},
    "jina_v3": {"checkpoint": "jinaai/jina-embeddings-v3"},
    # user-provided ones (aliases)
    "modernbert": {"checkpoint": "nomic-ai/modernbert-embed-base"},
    "stella": {"checkpoint": "dunzhang/stella_en_400M_v5"},
    "bge_large": {"checkpoint": "BAAI/bge-large-en-v1.5"},
    "bge_base": {"checkpoint": "BAAI/bge-base-en-v1.5"},
    "e5_small": {"checkpoint": "intfloat/e5-small"},
    # Add any other checkpoints you prefer here. Loading many models may use significant RAM.
}


def read_sts_file(path: str, delimiter: str = "\t", sentence_cols=(0, 1), score_col: int = 2, header: bool = False):
    pairs = []
    with open(path, "r", encoding="utf-8") as fh:
        if header:
            next(fh)
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split(delimiter)
            if len(parts) <= max(sentence_cols[0], sentence_cols[1], score_col):
                continue
            s1 = parts[sentence_cols[0]].strip()
            s2 = parts[sentence_cols[1]].strip()
            try:
                score = float(parts[score_col])
            except ValueError:
                continue
            pairs.append((s1, s2, score))
    return pairs


def build_model_from_args(model_name: Optional[str], model_path: Optional[str], trust_remote_code: bool = False):
    if model_path:
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers not available; please install requirements.")
        return SentenceTransformer(model_path, trust_remote_code=trust_remote_code)
    if model_name:
        entry = DEFAULT_MODEL_REGISTRY.get(model_name)
        if entry is None:
            raise ValueError(f"Unknown registry model '{model_name}'. Choices: {list(DEFAULT_MODEL_REGISTRY.keys())}")
        ckpt = entry["checkpoint"]
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers not available; please install requirements.")
        return SentenceTransformer(ckpt)
    raise ValueError("Either model_name or model_path must be provided")


def _cli_main(argv=None):
    ap = argparse.ArgumentParser(description="Embedding evaluator with IDF and SIF pooling + model registry")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--model", choices=list(DEFAULT_MODEL_REGISTRY.keys()), help="Model name from registry")
    g.add_argument("--model-path", help="Local or remote checkpoint string")
    ap.add_argument("--pooling", choices=["sent", "mean", "idf", "sif", "max"], default="mean")
    ap.add_argument("--sts-file", help="TSV/CSV file with sentence1,sentence2,score (if omitted, a demo runs)")
    ap.add_argument("--delimiter", default="\t")
    ap.add_argument("--sentence-cols", default="0,1")
    ap.add_argument("--score-col", type=int, default=2)
    ap.add_argument("--header", action="store_true")
    ap.add_argument("--lowercase", action="store_true", help="Lowercase input for tokenization/pooling")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--no-normalize", action="store_true", help="Do not normalize vectors before cosine")
    ap.add_argument("--sif-a", type=float, default=1e-3)
    ap.add_argument("--remove-pc", type=int, default=1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    model = build_model_from_args(args.model, args.model_path)
    evaluator = EmbeddingEvaluator(model=model, lowercase=args.lowercase, normalize=not args.no_normalize, batch_size=args.batch_size)

    if args.sts_file:
        scols = tuple(int(x) for x in args.sentence_cols.split(","))
        pairs = read_sts_file(args.sts_file, delimiter=args.delimiter, sentence_cols=scols, score_col=args.score_col, header=args.header)
    else:
        TEST_INPUT = [
            "breast cancer in males",
            "cancer breast male",
            "Malignant neoplasm of breast of unspecified site, male",
            "Malignant neoplasm of unspecified site of unspecified female breast"
        ]
        pairs = []
        for i in range(len(TEST_INPUT)):
            for j in range(i + 1, len(TEST_INPUT)):
                pairs.append((TEST_INPUT[i], TEST_INPUT[j], 0.0))

    results = evaluator.evaluate(pairs, pooling=args.pooling, sif_a=args.sif_a, remove_pc_n=args.remove_pc, progress=True)

    print("Results:")
    print(f"  model: {args.model or args.model_path}")
    print(f"  pooling: {args.pooling}")
    print(f"  Pearson r:    {results['pearson_r']:.4f} (p={results['pearson_p']:.3g})")
    print(f"  Spearman rho: {results['spearman_rho']:.4f} (p={results['spearman_p']:.3g})")
    print(f"  MSE:          {results['mse']:.6f}")
    print(f"  Pairs eval:   {results['n_pairs']}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write("s1\ts2\tgold\tpred\n")
            for s1, s2, gold in pairs:
                pred = evaluator.pairwise_similarity(s1, s2, pooling=args.pooling, idf=(evaluator.compute_idf([s1, s2]) if args.pooling == "idf" else None))
                fh.write(f"{s1}\t{s2}\t{gold}\t{pred}\n")
        print("Wrote per-pair results to", args.out)


if __name__ == "__main__":
    _cli_main()
EOF

# run_all_tests.py
cat > "$ROOT_DIR/run_all_tests.py" <<'EOF'
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
EOF

# example_sts.tsv
cat > "$ROOT_DIR/example_sts.tsv" <<'EOF'
breast cancer in males	cancer breast male	5.0
red	magenta	3.0
bone xray	xray bone	4.5
relentless coughing	coughing	4.0
keratoderma	skin disease	3.5
EOF

# requirements.txt
cat > "$ROOT_DIR/requirements.txt" <<'EOF'
sentence-transformers>=2.2.2
numpy
scipy
scikit-learn
tqdm
nltk
EOF

# requirements-dev.txt
cat > "$ROOT_DIR/requirements-dev.txt" <<'EOF'
# Base runtime deps
-r requirements.txt

# Development / CI
pytest
flake8
mypy
pre-commit
black
isort
EOF

# README.md
cat > "$ROOT_DIR/README.md" <<'EOF'
# Embedding Evaluator — Complete Project

This project provides a clean, tested, and extensible toolkit to compute and evaluate sentence embeddings using SentenceTransformer-style models.

What you get
- embedding_evaluator.py — main library with caching, multiple pooling methods (mean, idf, sif, max, sent), SIF postprocessing, model registry, and utilities.
- run_all_tests.py — script that runs many small TEST_INPUT lists you provided and prints pairwise similarities.
- example_sts.tsv — small STS-style example file.
- demo_notebook.ipynb — small runnable Colab / Jupyter notebook demonstrating installation and evaluation.
- requirements.txt — Python dependencies.
- examples.md — short usage examples showing how to call mean/idf/sif/max/sent pooling.

Important notes
- I cannot run model code in this environment. The notebook and scripts are ready to run on your machine or Colab.
- Some registry models may require `trust_remote_code=True` or extra dependencies — see notes in the README and in the registry constants inside embedding_evaluator.py.
- For SIF pooling we compute token probabilities from the STS corpus or provided sentences; SIF will subtract principal component(s), and this is optional.

Quick start (local)
1. Create a venv and install deps:
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   python -m nltk.downloader punkt

2. Run the demo script with a registry model:
   python run_all_tests.py --model biomodel --pooling mean

3. Run the CLI evaluation over example_sts.tsv:
   python embedding_evaluator.py --model modern --sts-file example_sts.tsv --pooling idf

Files
- embedding_evaluator.py: core library + CLI
- run_all_tests.py: runs pairwise tests on many lists (prints outputs)
- example_sts.tsv: example STS (tsv)
- demo_notebook.ipynb: Jupyter/Colab demo that loads a model and runs each pooling method.
- requirements.txt: dependencies
- examples.md: usage examples per pooling method
EOF

# examples.md
cat > "$ROOT_DIR/examples.md" <<'EOF'
# Examples: Using EmbeddingEvaluator

1) Mean pooling (token mean)