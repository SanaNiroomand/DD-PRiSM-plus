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
| pretrained (NCI60) | 0.0828 | 0.0830 | 0.9386 | 0.9387 |
| fine-tuned (ALMANAC mono) | 0.0898 | 0.0914 | 0.8854 | 0.8791 |
| combination (ALMANAC pairs) | 0.0849 | 0.0854 | 0.9100 | 0.9063 |

All three stages land within 0.0016 RMSE of published.

Stratified by what is held out, the pretrained model shows where the difficulty
actually is:

| held out | RMSE | PCC |
|---|---|---|
| unseen pair | 0.0828 | 0.9386 |
| unseen cell line | 0.0815 | 0.9355 |
| **unseen drug** | **0.1604** | **0.7585** |
| unseen both | 0.1502 | 0.7673 |

A cancer type the model has never seen costs it almost nothing. **A drug it has
never seen costs it a fifth of its correlation.** The paper names the cause
itself: *"we need more informative drug features for the phenotypic
prediction."* This is what the experiment below is aimed at.

The combination model shows the same shape, degrading monotonically with how
much is withheld:

| held out | RMSE | PCC |
|---|---|---|
| unseen pair | 0.0849 | 0.9100 |
| unseen cell line | 0.1203 | 0.7882 |
| unseen one drug | 0.1680 | 0.7810 |
| unseen two drugs | 0.2186 | 0.7604 |
| unseen all | 0.2024 | 0.6522 |

### A leak of our own, found by the ordering

An earlier version of this report quoted 0.0817 / 0.9082 and 0.0821 / 0.9176 for
the last two rows and claimed they beat the paper. They did not; they were
contaminated, and the giveaway was that the *ordering* above was inverted. The
fine-tuned model scored **PCC 0.9526 on unseen drugs against 0.9082 on unseen
pairs**, and the combination model 0.9436 on `unseen_all` against 0.9176 on
`unseen_pair`. A drug the model has never seen cannot be easier than a new
pairing of drugs it knows. Those rows were in its training set.

The cause was the resume path, not the split builder. Both stages were first
trained before the leakage fix below; re-running afterwards did not correct them,
because each had already early-stopped, so it resumed, ran one epoch, stopped
again and kept the contaminated weights. Retraining the two stages from the
intact pretrained weights moved `unseen_all` by **0.29 PCC**.

This also settles a caveat the earlier report could not explain — that our
stratified combination PCCs (0.92–0.94) sat far above the paper's ~0.75. Clean,
they are 0.65–0.91. The gap was our leak, not a difference in held-out entities.

**No metric is quoted here that a monotonic difficulty ordering does not
support.** That check costs nothing and would have caught this weeks earlier.

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
network unfrozen (45,082 parameters, 0.92% of the model). That reading lands at
0.0898 / 0.8854, essentially on the published figure, which is the evidence for
it. It remains a reading of a figure rather than a certainty.

An earlier draft credited this choice with *beating* the paper. That was wrong:
the margin came from the leak described above, not from the reading.

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

| notebook | drug representation | inputs | parameters |
|---|---|---|---|
| `03_train` | Morgan fingerprint | 512 | 4,882,576 |
| `04_experiment_fusion` | Morgan + ChemBERTa | 896 | 5,030,032 (+147,456) |
| `05_experiment_chemberta` | ChemBERTa alone | 384 | **4,833,424 (−49,152)** |

The third run is the control. Fusion adds capacity, so a win there can be
attributed to size; ChemBERTa alone has *fewer* parameters than the baseline,
and a win there cannot. Only the first `Linear` of each drug branch changes
width — every later layer keeps its published size, so a difference is
attributable to the representation.

Each experiment is a separate notebook writing to a separate directory, so no
run can overwrite another; `06_compare` puts the finished results side by side.
Embeddings are z-scored per dimension before fusion, because a dense float block
concatenated onto 5%-sparse bits otherwise dominates the first layer.

### Result: it helps on new drugs, and only there

Monotherapy model on NCI60, all three runs trained to early stop. RMSE / PCC:

| held out | Morgan | Morgan + ChemBERTa | ChemBERTa alone |
|---|---|---|---|
| **unseen drug** | 0.1604 / 0.7585 | **0.1502 / 0.7912** | 0.1545 / 0.7802 |
| **unseen both** | 0.1502 / 0.7673 | 0.1506 / **0.7910** | **0.1491** / 0.7806 |
| unseen pair | 0.0828 / 0.9386 | 0.0825 / 0.9390 | 0.0854 / 0.9346 |
| unseen cell line | 0.0815 / 0.9355 | 0.0940 / 0.9292 | 0.0846 / 0.9323 |

Three things follow.

**The gain is confined to drugs the model has never seen** — exactly where the
paper says its representation is weak. All three models sit within 0.006 PCC of
each other on unseen pairs and unseen cell lines. Those splits were never
limited by how the drug is described.

**It is not a capacity effect.** ChemBERTa alone has **49,152 fewer parameters**
than the Morgan baseline and still improves unseen-drug PCC by 0.022 and RMSE by
3.7%. A gain cannot be bought with capacity by a model that has less of it.

**The two representations are complementary.** On unseen drugs, fusion (0.7912)
beats ChemBERTa alone (0.7802), which beats Morgan (0.7585); RMSE agrees. The
fingerprint is not redundant once an embedding is present — the substructure
bag and the pretrained representation each carry something the other does not.

### What does not follow

**The gain does not survive fine-tuning.** On ALMANAC both variants are level or
behind at every split. We read this as a property of the data rather than of the
embedding: fine-tuning trains 0.93% of the model with the drug branches frozen,
and ALMANAC contains **102 drugs**. A richer representation earns its keep
generalising across 51,416 compounds; across 102 there is nothing to buy.
Testing that would need a fine-tuning set with far more drugs, which
NCI-ALMANAC does not have.

**The combination-stage differences are not interpretable.** Training that stage
at lr 1e-2 is unstable: validation correlation swings between +0.93 and −0.59
across neighbouring epochs before the rate drops. ChemBERTa alone posts the best
combination `unseen_pair` of the three (0.0784 / 0.9246, better than the paper's
0.0854 / 0.9063) — but the same model is the *worst* of the three at pretraining
`unseen_pair`. A representation effect does not reverse sign between stages, so
we read that spread as noise and do not claim it. The instability is a property
of the published schedule, which is the schedule we ran.

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
