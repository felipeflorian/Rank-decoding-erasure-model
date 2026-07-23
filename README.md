# Erasure-Aware Rank Decoding & Side-Channel Recovery Model

A Python and SageMath framework for generating, simulating, and evaluating **Rank-Metric Linear Codes** under erasure scenarios and noisy side-channel leakage models (specifically targeting **Cold-Boot Attacks**). 

This repository implements the **Algebraic Profile (AP) Erasure-Aware Decoder** for rank-metric codes, along with a bit-level **Optimal Key Enumeration Algorithm (OKEA)** pipeline to reconstruct secret seeds/keys from noisy observations.

---

## 📁 Repository Structure

```text
├── generated_data/              # Dataset of error vector and matrix instances
│   ├── instances_n7_k3_r2_m4.json
│   ├── instances_n15_k7_r5_m15.json
│   └── instances_n33_k15_r10_m31.json
│
├── data_for_seeds/              # Multi-beta simulated leakage datasets for seed recovery
│   └── simulation_data_*_beta_*.json
│
├── evaluation_results/          # Evaluation logs of decoder success and timings
│   ├── CBA/                     # Leakage-guided erasure decoding outputs
│   └── true_columns/            # Random/oracle baseline erasure decoding outputs
│
├── utils.py                     # SageMath core algebraic decoders and leakage simulators
├── candidate.py                 # ChunkCandidate class for key enumeration scoring
├── enumeration_utils.py         # Utilities to merge search candidates
├── okeanode.py                  # Heap priority queue search nodes for OKEA
├── seed_recovery_algorithms.py  # Algorithms for posterior generation and OKEA search taken from https://github.com/AmOchoat/Paper-Recovery-Seeds
│
├── generate_data_simulation.py  # Generates code matrices, error vectors, and syndromes
├── generate_data_CBA.py         # Generates simulated leakage data under various betas
├── CBA_simulation.py            # Evaluates the AP decoder with leakage-guided erasures
├── simulation_true_columns.py   # Evaluates the AP decoder with random true erasures
│
├── seed_results.csv             # Collected logs of seed recovery experiments
├── generation_seeds.ipynb       # Jupyter notebook analyzing seed-recovery statistics
├── evaluation_analysis.ipynb    # Jupyter notebook visualizing decoding performance
├── test_algorithms.ipynb        # SageMath playground testing
```

---

## ⚙️ Core Modules

### 1. Algebraic Foundations (`utils.py`)
Provides SageMath implementations for code manipulation and decoding:
- `full_entry_ap_decoder_module(...)`: Implements the complete **Full-Entry Erasure-Aware AP Decoder** (Algorithm 5).
- `ap_erasure_decoder_module(...)`: Solver wrapper with local dimensional checks and coordinate-reduction hybridization loops.
- `simulate_and_analyze_leakage(...)`: Simulates cold-boot leakage channel bit-flips on $E$ and ranks column reliability.
- `grs_scalar_erasures(...)`: Decoder variant tailored for Generalized Reed-Solomon-like scalar rank codes.

### 2. Key Enumeration & Seed Recovery (`seed_recovery_algorithms.py`, `okeanode.py`, `candidate.py`, `enumeration_utils.py`)
Reconstructs the original binary representations from noisy leakages:
- `ChunkCandidate`: Stores the candidate bit array, score (negative log-likelihood), and base-field integer weight.
- `generate_candidates(...)`: Implements candidate generation per chunk using local posterior log-probabilities.
- `OKeaNode`: Tree search nodes representing paths in the Cartesian search space.
- `getSeed(...)`: Evaluates candidates sequentially to match target syndrome-verified keys.

---

