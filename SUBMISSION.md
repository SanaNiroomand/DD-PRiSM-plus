# DD-PRiSM — Reproduction Report

**Reproducing:** Jin I, Lee S, Schmuhalek M, Nam H. *DD-PRiSM: a deep learning
framework for decomposition and prediction of synergistic drug combinations.*
Briefings in Bioinformatics 2025;26(1):bbae717.
Original code: <https://github.com/GIST-CSBL/DD-PRiSM>

| | |
|---|---|
| ثنا نیرومند — Sana Niroomand | 402171104 |
| ریحانه خیاط زاده ماهانی — Reyhaneh Khayyatzadeh Mahani | 402105965 |

**Video:** <https://drive.google.com/drive/folders/1412g7UOoaB3qsiqZMGRoz9ZBv3i3QVvu?usp=sharing>

**Repository:** <https://github.com/SanaNiroomand/DD-PRiSM-plus>

---

## What DD-PRiSM does

Two models in sequence. The **Monotherapy model** takes a cell line's
pathway-grouped gene expression, a drug's 512-bit Morgan fingerprint and a dose,
and predicts cell viability by reconstructing a four-parameter Hill curve. The
**Combination therapy model** then takes both drugs' predicted viability and
pathway attention and splits the pair's effect into three parts:

    E(C, D1, D2, d1, d2) = α·E(C,D1,d1) + β·E(C,D2,d2) + γ

α and β weight each drug's own contribution; **γ is the synergy** — what the
pair achieves beyond the sum of its parts. That decomposition is the paper's
contribution: earlier work predicts a synergy score, this predicts actual
viability *and* attributes it.

## Reproduction results

Preprocessing was validated against the paper's published counts.

**NCI-ALMANAC — exact, zero difference:**

| | ours | paper |
|---|---|---|
| monotherapy rows | 35,041 | 35,041 |
| combination rows | 1,981,135 | 1,981,135 |
| distinct drug pairs | 5,032 | 5,032 |

**NCI60 — within 1%:**

| | ours | paper | Δ |
|---|---|---|---|
| rows in DOSERESP | 23,636,946 | 23,636,946 | **exact** |
| filtered total | 10,204,028 | 10,105,780 | +0.97% |
| train + validation | 8,993,256 | 8,905,388 | +0.99% |

The residual comes from one extra cell line (67 vs 66) and RDKit version
differences in which structures parse — tool drift, not pipeline error.

**Model equations verified without training.** Supplementary Data 3 publishes
the trained model's output for 2,556 combinations, including both coefficients
and the synergy term. Our decomposition reproduces the authors' own viabilities
to **1.5e-07**, and α + β = 1 to 1e-07 (`scripts/check_against_paper.py`).

## What we found in the published code

Three issues, all documented in code and reproducible:

1. **Predicted viability is unbounded.** `CombinationTherapyModel` constructs
   `efficacy_relu` and `viability_relu` and never calls them. This is not
   hypothetical: **573 of the 2,556 rows (22.4%) in the authors' own
   Supplementary Data 3 have negative viability**, down to −0.78. Viability is a
   surviving-cell fraction, so those values are physically impossible. Exposed as
   `bound_output`, defaulting to off so the paper is reproduced as written.

2. **Test data leaks into BatchNorm.** `test_mono`, `test_comb` and both
   `predict_*` helpers call `model.train()` and run the *evaluation* set before
   switching to `eval()`, updating BatchNorm running statistics on test data.

3. **A rank-2 bottleneck.** All four Hill-curve parameters are linear functions
   of the same 2-dimensional non-negative vector, and nothing constrains
   `y_max > y_min`.

## Engineering contributions

**A vectorised Monotherapy model, 5.2× faster on a Tesla T4.** The published
model loops over 186 KEGG pathways in Python. Batching them by size bucket cuts
an NCI60 epoch from 31.5 to 6.0 minutes — the difference between 57 and 300
epochs on a 30-hour GPU quota. Pinned to the published model by tests: identical
outputs **and gradients** to 1e-10 in float64, measured at 1.3e-15 on a T4.

*Naive padding made it 2× slower* — the gene-attention layer costs O(n²) in
pathway size and KEGG sizes are skewed (median 60, max 390). Bucketing by size
was what made it work.

**Device-resident data pipeline.** The published dataset rebuilds 186 tensors
per batch from pandas, ~50 ms — invisible on CPU, but 53% of a T4 step. There
are only 66 cell lines, so their expression is a fixed 31 MB table that lives on
the GPU. End to end: **~36 min → ~6.5 min per epoch.**

**Checkpoint/resume.** Kaggle ends sessions at ~12 hours; pretraining runs
longer. Every epoch saves weights, optimiser moments, learning rate, patience
counters and RNG state.

**44 tests.**

## Layout

```
original/ddprism_original.py   the authors' code, extracted verbatim from their
                               notebooks by scripts/extract_original.py.
                               3 lines differ, all IPython %run magics, each logged
ddprism/                       our package
  monotherapy.py               vectorised model, pinned to the original by tests
  combination.py               combination model, same
  losses.py                    density-weighted MSE + correlation (α=1, β=0.5, γ=0.75)
  data.py                      device-resident batching
  train.py                     three-stage training with checkpoint/resume
scripts/
  get_data.py                  fetch + verify all 8 sources (size, archive, MD5)
  preprocess.py                raw files -> training tables
  paper_sets.py                the 102 drugs / 44 cell lines from Supplementary Data 1
  check_against_paper.py       verify the decomposition against Supplementary Data 3
  extract_original.py          regenerate original/ from the notebooks
kaggle/                        01_setup_and_data, 02_preprocess, 03_train
tests/                         44 tests
00_*.ipynb, 01-04_*.ipynb      the authors' original notebooks
```

## Running it

Three Kaggle notebooks in order. Steps 1 and 2 need CPU only; step 3 needs a GPU.

```bash
python -m pytest tests -q                              # 44 tests
python scripts/get_data.py --dest data                 # ~1 GB, 8 sources
python scripts/preprocess.py --data data --out processed
python -m ddprism.train --data data --processed processed --out runs --model fast
```

**Obstacles worth recording**, since a reproducer will hit them:

- `DOSERESP.zip` is **Deflate64**. Python's `zipfile` reads its directory but
  cannot decompress it; `pd.read_csv` fails. Needs `zipfile-deflate64`.
- The NCI wiki returns **403 to HEAD but serves GET fine**. Checking those links
  with `curl -I` makes three live URLs look dead.
- **figshare was down site-wide** during this work, answering `202 Accepted` with
  an empty body. The two DepMap files must be fetched by hand from the portal.
- Pin **DOSERESP version 10** (January 2024). Later releases carry newer
  experiments and the published row counts stop matching.
- A naive `pd.read_csv` of DOSERESP costs **11.1 GB**. Six columns with tight
  dtypes, read in chunks, is 26× smaller.

## Licence

The authors' code in `original/` is CC-BY-NC-SA-4.0 by Jin et al. Data belongs
to NCI DTP, DepMap/Broad, MSigDB and AACR under their respective terms.
