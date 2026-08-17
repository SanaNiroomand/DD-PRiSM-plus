# DD-PRiSM-plus

A from-scratch reimplementation of **DD-PRiSM**, plus an experiment the original
paper suggests but does not run.

> Jin I, Lee S, Schmuhalek M, Nam H. *DD-PRiSM: a deep learning framework for
> decomposition and prediction of synergistic drug combinations.*
> Briefings in Bioinformatics 2025;26(1):bbae717.
> Authors' code: <https://github.com/GIST-CSBL/DD-PRiSM>

---

## The problem, in plain words

Doctors often treat cancer with **two drugs at once**, because two drugs can do
more together than either does alone. That extra benefit is called **synergy**.

The trouble is arithmetic. There are thousands of cancer drugs and hundreds of
cancer types, so there are millions of possible combinations. Nobody can test
them all in a lab. So the question is: **can a computer predict which pairs will
work well together, without testing them?**

## What the model does

Two models, one after the other.

**Model 1 — one drug at a time.** You give it three things:

- which genes are switched on in the cancer cell
- a description of the drug's chemical structure
- how much of the drug you used

It predicts **what fraction of the cells survive**. Give it more of a drug and
fewer cells survive — so what it really learns is the shape of that curve.

**Model 2 — two drugs at once.** It takes what Model 1 said about each drug
separately, and splits the pair's total effect into three pieces:

```
total effect  =  α × (what drug 1 does alone)
              +  β × (what drug 2 does alone)
              +  γ
```

That last piece, **γ, is the synergy** — the part you only get by using both.

This split is the paper's real contribution. Earlier methods give you one score
saying "these two work well together." This tells you *how much* of the effect
came from each drug, and how much came from the pairing itself.

## What's in this repository

**1. A faithful reproduction.** The models are the authors' own code, copied
byte for byte into `original/`, not rewritten from the paper. Everything around
them — downloading, preprocessing, training, evaluation — is ours.

**2. A rebuilt data pipeline** that reproduces the paper's published row counts
exactly for NCI-ALMANAC and to within 1% for NCI60.

**3. A faster model.** Same maths, 5.2× faster on a Tesla T4, which is what made
this runnable on a free GPU at all.

**4. An experiment the paper points to but doesn't run** — see below.

## Does it reproduce?

Yes. Measured on the paper's own held-out sets:

| stage | our RMSE | paper | our PCC | paper |
|---|---|---|---|---|
| pretrained | 0.0828 | 0.0830 | 0.9386 | 0.9387 |
| fine-tuned | 0.0817 | 0.0914 | 0.9082 | 0.8791 |
| combination | 0.0821 | 0.0854 | 0.9176 | 0.9063 |

RMSE is error, so **lower is better**. PCC is agreement, so **higher is better**.

These are the "unseen pair" numbers, which is what the paper reports. Our last
two rows come out slightly *better* than published — we read the fine-tuning
step as unfreezing the whole curve prediction network, which is how Figure S3B
draws it. Reading it as only the four output neurons gives 12 trainable
parameters and much worse results, so we think this reading is right, but it is
a reading.

We also checked the model's equations *without training anything*. The authors
published their trained model's output for 2,556 real drug pairs, so we fed
those same numbers through our decomposition. It matches theirs to **0.00000015**.

## The experiment: better drug descriptions

The paper describes each drug as a **512-bit Morgan fingerprint** — 512 yes/no
answers to "does this molecule contain this fragment?" It is a list of parts,
and it says nothing about what the molecule actually *does*.

Our held-out numbers show exactly where that hurts:

| test | what it means | PCC |
|---|---|---|
| unseen pair | new combination of familiar drugs | 0.9386 |
| unseen cell line | a cancer type never trained on | 0.9355 |
| **unseen drug** | **a drug never trained on** | **0.7585** |

New cancer types are easy. **New drugs are not.** The paper names the reason
itself: *"we need more informative drug features for the phenotypic prediction."*

So we replace the fingerprint with an embedding from **ChemBERTa**, a model
pretrained on 77 million molecules — a model that has *seen chemistry*, rather
than a checklist of fragments. Three experiments, each in its own notebook:

| notebook | drug description | why |
|---|---|---|
| `03_train` | Morgan fingerprint | the paper — the baseline |
| `04_experiment_fusion` | Morgan **+** ChemBERTa | does richer chemistry help? |
| `05_experiment_chemberta` | ChemBERTa alone | the control (see below) |

The third one matters. Fusion adds 5.7% more parameters than the paper's model,
so if it wins, someone can fairly say "you just used a bigger network."
ChemBERTa alone has **fewer** parameters than the paper's model. If that wins
too, size is not the explanation — the chemistry is.

Only the **first layer** of the network changes width. Every other layer keeps
the published size, so any difference in the results comes from the drug
description and not from a bigger model.

## How to run it

Six notebooks in `kaggle/`, designed for a free Kaggle account. Only the
training ones need a GPU.

| # | notebook | needs | roughly |
|---|---|---|---|
| 1 | `01_setup_and_data` | CPU | 30 min |
| 2 | `02_preprocess` | CPU, internet on | 40 min |
| 3 | `03_train` | GPU | 7 h |
| 4 | `04_experiment_fusion` | GPU | 7 h |
| 5 | `05_experiment_chemberta` | GPU | 7 h |
| 6 | `06_compare` | CPU | seconds |

Each experiment is a **separate notebook writing to a separate folder**, so they
cannot overwrite each other, and `06_compare` puts the finished results side by
side.

Or locally:

```bash
python -m pytest tests -q
python scripts/get_data.py --dest data
python scripts/preprocess.py --data data --out processed
python scripts/embed_drugs.py --data data --out processed --model chemberta
python -m ddprism.train --data data --processed processed --out runs --model fast --drug-features morgan+chemberta
python -m ddprism.evaluate --data data --processed processed --runs runs
```

Training on a CPU is not realistic — one pass over the data takes about half an
hour on a GPU.

## Three things we found in the published code

**1. Predicted survival can go negative.** The combination model builds two
`ReLU` layers to prevent this and then never calls them. This is not
theoretical: **573 of the 2,556 rows in the authors' own published results are
negative**, down to −0.78. A survival fraction below zero means more than all of
the cells died. Available as `bound_output`, off by default so the paper
reproduces as written.

**2. Test data leaks into the model during testing.** Their `test_` and
`predict_` helpers switch the model to training mode and run the *test* set
through it, which quietly updates the model's internal statistics using data it
is supposed to be judged on.

**3. A very narrow bottleneck.** All four numbers describing the dose-response
curve are computed from the **same 2 numbers**, and half of those 2 are switched
off by a ReLU. We measured the four outputs: they have **rank 1** — effectively
one number wearing four hats. The model cannot describe a curve's height and its
steepness independently.

## Layout

```
original/            the authors' code, extracted verbatim from their notebooks.
                     3 lines differ, all IPython %run magics, each one logged.
                     README_upstream.md is their original readme.
ddprism/             our package
  monotherapy.py       the faster model, pinned to the original by tests
  combination.py       the combination model, same
  drugfeatures.py      assembles the drug description (Morgan / ChemBERTa / both)
  losses.py            density-weighted MSE + correlation penalty
  data.py              keeps the data on the GPU instead of rebuilding it
  train.py             three-stage training, resumable
  evaluate.py          scores a run on the held-out sets
scripts/
  get_data.py          fetch and verify all 8 data sources
  preprocess.py        raw downloads -> training tables
  embed_drugs.py       ChemBERTa / MolFormer embeddings for every drug
  check_against_paper.py   verify the decomposition against the paper's own output
  extract_original.py  regenerate original/ from the authors' notebooks
kaggle/              the six notebooks above
tests/               the test suite
```

## Why "plus"

The reproduction is the floor, not the ceiling. `original/` is untouched and the
Morgan baseline reproduces the paper, so every experiment is measured against a
version of the paper that actually works — not against a description of it.

## Credit and licence

The model code in `original/` is by Jin et al. and is CC-BY-NC-SA-4.0. Their
original readme is preserved at `original/README_upstream.md`. The data belongs
to NCI DTP, DepMap/Broad, MSigDB and AACR under their own terms.
