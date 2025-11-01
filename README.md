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
