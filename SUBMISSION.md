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

## Model results

Trained end to end on a Kaggle T4 and scored on the paper's own held-out sets.
The paper reports on the **unseen pair** set, so that is the like-for-like
comparison:

| stage | our RMSE | paper RMSE | our PCC | paper PCC |
|---|---|---|---|---|
| pretrained (NCI60) | **0.0828** | 0.0830 | **0.9386** | 0.9387 |
| fine-tuned (ALMANAC mono) | **0.0817** | 0.0914 | **0.9082** | 0.8791 |
| combination (ALMANAC pairs) | **0.0821** | 0.0854 | **0.9176** | 0.9063 |

Pretraining lands on the published numbers to within 0.0002 RMSE. The other two
rows come out **better** than published, for a reason we can name — see
"Fine-tuning" below.

Stratified by what is held out, the pretrained model shows where the difficulty
actually is:

| held out | RMSE | PCC |
|---|---|---|
| unseen pair | 0.0828 | 0.9386 |
| unseen cell line | 0.0815 | 0.9355 |
| **unseen drug** | **0.1604** | **0.7585** |

A cancer type the model has never seen costs it almost nothing. **A drug it has
never seen costs it a fifth of its correlation.** The paper names the cause
itself: *"we need more informative drug features for the phenotypic
prediction."* This is what the experiment below is aimed at.

*Caveat.* On the stratified **combination** splits our PCCs (0.92–0.94) are far
above the paper's (~0.75). We could not reconcile this and believe the held-out
entities differ — our `unseen_all` has 6,516 rows against the paper's 54. We
therefore quote only the unseen-pair column as comparable.

## Preprocessing, validated against published counts

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

## Two decisions that changed the results

**Fine-tuning.** The Methods say fine-tuning updates "the layers that predict
the four curve parameters." Read literally, that is the four `Linear(2, 1)`
heads — **12 trainable parameters**, which cannot absorb the NCI60 → ALMANAC
shift: we measured RMSE 0.1426 / PCC 0.699 against the paper's 0.0914 / 0.8791.
Figure S3B draws the boundary differently, with the whole Curve prediction
network unfrozen (45,082 parameters). That reading gives 0.0876 / 0.8940 and,
after the leakage fix below, 0.0817 / 0.9082. It is why our fine-tuned row beats
the published one, and it is a reading of a figure, not a certainty.

**Data leakage, found and removed.** The authors publish no ALMANAC train/test
split, so our first runs fine-tuned and trained the combination model on rows
belonging to the paper's own test sets. Every ALMANAC metric before that fix was
optimistic and is not reported here. `stage_almanac_splits` now builds the
splits and `require_split()` makes training **refuse** the unsplit tables rather
than silently use everything.

## Three issues in the published code

1. **Predicted viability is unbounded.** `CombinationTherapyModel` constructs
   `efficacy_relu` and `viability_relu` and never calls them. This is not
   hypothetical: **573 of the 2,556 rows (22.4%) in the authors' own
   Supplementary Data 3 have negative viability**, down to −0.78. Viability is a
   surviving-cell fraction, so those values are physically impossible. Exposed as
   `bound_output`, defaulting to off so the paper is reproduced as written.

2. **Test data leaks into BatchNorm.** `test_mono`, `test_comb` and both
   `predict_*` helpers call `model.train()` and run the *evaluation* set before
   switching to `eval()`, updating BatchNorm running statistics on test data.

3. **A rank-1 bottleneck.** All four Hill-curve parameters are affine functions
   of the same 2-dimensional non-negative vector. We measured the four outputs
   on the authors' own model: they span **rank 1 of 4**, and the `ReLU` after
   `BatchNorm(2)` leaves **50% of the bottleneck at zero**. The model cannot
   choose a curve's height, steepness and midpoint independently. Nothing
   constrains `y_max > y_min` either.

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

**The authors' code, not a transcription.** `original/ddprism_original.py` is
their three util notebooks written out cell by cell by
`scripts/extract_original.py`. Three lines differ, all IPython `%run` magics,
each one logged. Training runs against their classes; our vectorised versions
exist to be checked against them.

**96 tests.**

## Extension: a better description of the drug

The unseen-drug result above is the paper's own stated limitation, measured. A
512-bit Morgan fingerprint is a bag of substructures; it says what a molecule
*contains*, not what it *does*. We replace it with an embedding from
**ChemBERTa**, a transformer pretrained on 77M PubChem molecules.

| notebook | drug representation | inputs | parameters vs paper |
|---|---|---|---|
| `03_train` | Morgan fingerprint | 512 | baseline |
| `04_experiment_fusion` | Morgan + ChemBERTa | 896 | +147,456 (+5.7%) |
| `05_experiment_chemberta` | ChemBERTa alone | 384 | **−49,152** |

The third run is the control. Fusion adds capacity, so a win there can be
attributed to size; ChemBERTa alone has *fewer* parameters than the paper's
model, and a win there cannot. Only the first `Linear` of each drug branch
changes width — every later layer keeps its published size, so a difference is
attributable to the representation.

Each experiment is a separate notebook writing to a separate directory, so no
run can overwrite another; `06_compare` puts the finished results side by side.
Embeddings are z-scored per dimension before fusion, because a dense float block
concatenated onto 5%-sparse bits otherwise dominates the first layer.

## Layout

```
original/                      the authors' code, extracted verbatim from their
                               notebooks by scripts/extract_original.py.
                               3 lines differ, all IPython %run magics, each logged.
                               README_upstream.md is their original readme.
ddprism/                       our package
  monotherapy.py               vectorised model, pinned to the original by tests
  combination.py               combination model, same
  drugfeatures.py              assembles the drug representation, Morgan / embedding / both
  losses.py                    density-weighted MSE + correlation (α=1, β=0.5, γ=0.75)
  data.py                      device-resident batching
  train.py                     three-stage training with checkpoint/resume
  evaluate.py                  scores a run on the held-out splits
scripts/
  get_data.py                  fetch + verify 8 sources by default (size, archive, MD5)
  preprocess.py                raw files -> training tables, incl. held-out splits
  embed_drugs.py               ChemBERTa / MolFormer embeddings for every drug
  paper_sets.py                the 102 drugs / 44 cell lines from Supplementary Data 1
  check_against_paper.py       verify the decomposition against Supplementary Data 3
  extract_original.py          regenerate original/ from the notebooks
  build_kaggle_notebook.py     generate the six notebooks, refusing to write a broken one
kaggle/                        01_setup_and_data, 02_preprocess, 03_train,
                               04_experiment_fusion, 05_experiment_chemberta, 06_compare
tests/                         96 tests
00_*.ipynb, 01-04_*.ipynb      the authors' original notebooks. The three 00_* files
                               are the input to extract_original.py, not spare copies.
```

## Running it

Six Kaggle notebooks. Only the three training ones need a GPU.

```bash
python -m pytest tests -q
python scripts/get_data.py --dest data                 # ~1 GB, 8 sources
python scripts/preprocess.py --data data --out processed
python scripts/embed_drugs.py --data data --out processed --model chemberta
python -m ddprism.train --data data --processed processed --out runs --model fast
python -m ddprism.evaluate --data data --processed processed --runs runs
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
- Cell-line annotation must come from DepMap **`Model.csv`**, not the 18Q3
  Achilles `sample_info.csv`: the latter covers 485 screened lines and resolves
  only 35 of the 66 needed.
- Expression is keyed by DepMap `ModelID` (`ACH-000123`), every response table by
  NCI60 name (`786-0`). They must be re-keyed before use.

## Licence

The authors' code in `original/` is CC-BY-NC-SA-4.0 by Jin et al. Data belongs
to NCI DTP, DepMap/Broad, MSigDB and AACR under their respective terms.
