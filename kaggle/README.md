# Running DD-PRiSM-plus on Kaggle

**The six notebooks in this directory are the actual procedure.** Import them
into Kaggle and run them in order; each one carries its own instructions.

| # | notebook | session | roughly | what it does |
|---|---|---|---|---|
| 1 | `01_setup_and_data` | CPU | 30 min | fetch and verify the 8 required data sources |
| 2 | `02_preprocess` | CPU, internet **on** | 40 min | build the training tables and the drug embeddings |
| 3 | `03_train` | GPU | ~7 h | the paper's model — Morgan fingerprint |
| 4 | `04_experiment_fusion` | GPU | ~7 h | Morgan + ChemBERTa |
| 5 | `05_experiment_chemberta` | GPU | ~7 h | ChemBERTa alone (the control) |
| 6 | `06_compare` | CPU | seconds | all finished runs side by side |

Each experiment writes to its own directory, so no run can overwrite another.
Attach the *previous* notebook's saved output as an input to the next one.

Do not hand-edit the notebooks. They are generated:

```bash
python scripts/build_kaggle_notebook.py
```

The generator compiles every code cell before writing and refuses to emit a
notebook that does not parse — one shipped with a `SyntaxError` once and cost a
session. `tests/test_notebooks.py` checks the committed files still match.

This document is the **background**: the account setup, and the things that went
wrong during development that a reproducer will hit too.

---

## Before anything

- Sign in at <https://www.kaggle.com>
- **Settings -> Phone Verification** and verify your phone number. Without this
  you cannot use a GPU or turn on internet access.
- Open a notebook's right-hand panel (the `<` arrow, top right) to reach
  **Session options -> Accelerator** and **-> Internet**.

Two Jupyter rules worth knowing, because breaking either gives you a confusing
`SyntaxError`: shell commands need a leading `!`, and `%cd` is a notebook magic
that must sit alone on its own line. You cannot chain them with `&&`.

---

## Optional: prove the fast model on your GPU first (~10 min)

Needs no data, so it runs the moment a GPU session starts:

```python
!python kaggle/gpu_check.py --batch 1024 --iters 10
```

### What to look for

The script prints three blocks:

- **ENVIRONMENT** -- confirms the GPU was actually allocated. If it says
  `NO GPU FOUND`, the accelerator is not enabled; fix that and rerun.
- **CORRECTNESS** -- must say `verdict: MATCHES`. This proves the vectorised
  model gives the same numbers as the published one *on CUDA*, which the test
  suite has only ever checked on CPU. If it says MISMATCH, stop and report it.
- **SPEED** -- a table across bucket counts, ending in a verdict line.

Whichever bucket count wins is the one to train with.

**Already measured on a Tesla T4** (batch 1024, forward+backward): 8 buckets at
**5.2x**, 16 buckets at 4.9x, against 232 ms for the published loop. Rerun this
on your own session to confirm, since a different GPU may prefer a different
bucket count.

The `1 bucket` row is worth a look. It pads every pathway to the widest, doing
~17x more arithmetic than necessary, and still beats the loop on a GPU (1.4x)
while *losing* badly on CPU (0.2x). That is the whole argument in one line:
on a GPU the cost is how many operations you issue, not how much maths each
one does.

---

## The data (entirely on Kaggle)

Every source is scriptable, so nothing needs to be downloaded locally and
re-uploaded. This is what `01_setup_and_data` runs:

```python
!python scripts/get_data.py --dest /kaggle/working/data
```

That pulls eight files, about 1.0 GB, in a few minutes:

| File | Size | Source |
|---|---|---|
| `DOSERESP.zip` (NCI60, **version 10**) | 333 MB | NCI wiki |
| `ComboDrugGrowth_Nov2017.zip` (ALMANAC) | 86 MB | NCI wiki |
| `Chem2D_Jun2016.zip` | 81 MB | NCI wiki |
| `nsc_smiles.csv` | 17 MB | NCI wiki |
| `OmicsExpressionProteinCodingGenesTPMLogp1.csv` | 450 MB | DepMap 23Q4 via figshare |
| `Model.csv` | 0.4 MB | DepMap 23Q4 via figshare |
| `sample_info_18q3.csv` | 0.06 MB | DepMap 18Q3 via figshare |
| `c2.cp.kegg_legacy...gmt` | 0.1 MB | Broad data host |

A ninth source, O'Neil et al. (39 MB, AACR), is the paper's external validation
set. Nothing in this repository reads it yet, so it is **opt-in**:
`--include-optional`.

**Use `Model.csv`, not `sample_info_18q3.csv`, for cell-line names.** The 18Q3
Achilles file covers only the 485 lines that were screened and resolves just 35
of the 66 needed. That mistake looks like a preprocessing bug for a long time.

**Fingerprints come from `Chem2D_Jun2016.zip`**, which is what the authors
parsed. `nsc_smiles.csv` is simpler and *more* complete, which sounds better and
is not: it covers all 57,041 NSCs and inflates every downstream count by ~11%.

Then verify:

```python
!python scripts/get_data.py --dest /kaggle/working/data --check
```

Every required row must read `ok`. A `SUSPECT SIZE` usually means an HTML error
page got saved instead of the file.

### If figshare answers 202

The two DepMap files come from figshare, which starts returning **202 Accepted
with an empty body** to every request from an IP after a handful of downloads.
It is a 2xx, so naive code reads an empty response and saves a 0-byte file
believing it succeeded. `get_data.py` detects this and backs off, but no amount
of backoff helps while the throttle holds -- it outlasted 15 minutes of retries
during development.

It is per-client, not per-file: a small file that had downloaded seconds earlier
began returning 202 as well. There is no mirror, and `depmap.org/portal/api`
serves a bot-verification page rather than data.

If you hit it, wait and rerun just that file:

```bash
python scripts/get_data.py --dest data --only depmap_expression --attempts 9
```

A fresh Kaggle session usually is not affected, since the throttle follows the
client address.

### Why the URLs look dead when they are not

The NCI wiki answers **403 to a HEAD request** but serves a **GET** perfectly
well, provided a browser User-Agent is set. Checking those links with
`curl -I` reports 403 and makes them look blocked. They are not. `get_data.py`
sets the User-Agent and uses GET.

MSigDB is similar: the web portal wants an account, but the Broad's data host
serves the same GMT with no login.

### Pin DOSERESP to version 10

NCI keeps revising this file; the current one is version 20 (July 2026) and
carries experiments run since the paper. Different rows means different counts,
and you lose the only checkpoint that tells you whether preprocessing worked.
Version 10 (January 2024) is what the authors used and is still served. The URL
in `get_data.py` is already pinned to it.

(The wiki's note about switching from aggregated values to per-experiment
`EXPID` rows predates version 10 -- that column is already present there -- so
the two versions differ in content, not layout.)

### Preprocessing, on Kaggle

Measured, not guessed. `DOSERESP.csv` is **23,636,946 rows / 2.44 GB
uncompressed**, covering 163 cell lines and 57,495 drugs before any filtering.

**DOSERESP is Deflate64.** The standard library reads its directory but cannot
decompress it -- `pd.read_csv("DOSERESP.zip")` fails with
`NotImplementedError: That compression method is not supported`, and so does
`zipfile.extractall`. One package fixes it:

```python
!pip install zipfile-deflate64
```
```python
import zipfile_deflate64, zipfile   # the import patches zipfile in place
import pandas as pd
z = zipfile.ZipFile("data/Raw/DOSERESP.zip")
with z.open("DOSERESP.csv") as f:
    ...
```

**Read it lean, and never `pd.read_csv` the whole thing.** Loading all 18
columns at default dtypes costs **11.1 GB**. Taking only the six columns the
paper uses, with tight dtypes, costs a fraction of that:

```python
cols = ["NSC", "CONCENTRATION", "CELL_NAME",
        "AVERAGE_GIPRCNT", "STDDEV_GIPRCNT", "CONCENTRATION_UNIT"]
dtypes = {"NSC": "int32", "CONCENTRATION": "float32",
          "AVERAGE_GIPRCNT": "float32", "STDDEV_GIPRCNT": "float32",
          "CELL_NAME": "category", "CONCENTRATION_UNIT": "category"}

for chunk in pd.read_csv(f, usecols=cols, dtype=dtypes, chunksize=2_000_000):
    ...  # filter and aggregate here, do not accumulate raw chunks
```

That is a **26x** reduction per row. Filter inside the loop -- keep
`CONCENTRATION_UNIT == "M"` and the 66 cell lines you actually need -- and peak
memory stays in the hundreds of MB rather than tens of GB. A full lean pass
takes about a minute.

Two smaller things:

- `sample_info_18q3.csv` arrives with columns `Broad_ID`, `CCLE_name`,
  `aliases`. The notebook expects `CCLE_Name` and `Aliases`; rename first.
- RDKit is not preinstalled: `!pip install rdkit`. You can skip it entirely by
  using `nsc_smiles.csv` instead of parsing Chem2D.

**Checkpoint.** Table S2 says NCI60 must yield **7,915,900** training
responses; Table S4 says the combination training set must yield **1,387,317**.
Hit those and preprocessing is right.

### Keep the result

`/kaggle/working` is wiped when the session ends. Click **Save Version** once
preprocessing finishes, then add that output as an input dataset to the next
session so you never redo this.

---

## Phase 3 -- Training

### Things Kaggle will do to you

- **GPU quota is ~30 hours per week.** It resets weekly. Watch it.
- **A session is killed after ~12 hours**, and sooner if idle.
- **`/kaggle/working` is wiped** unless you click **Save Version**.

Training NCI60 will not finish in one session. So:

- Save a checkpoint every epoch into `/kaggle/working/`
- **Save Version** at the end of a session
- Add that output as an input dataset to the next session and resume

### Before the first real run

Use `ddprism.data` instead of the published `MonotherapyDataset`, and skip
`DataLoader` entirely.

The published dataset rebuilds 186 tensors per batch from pandas object
columns: about 50 ms per batch of 1024. That was invisible when the model took
560 ms on CPU, but it is 53% of every step once the model runs at 44 ms on a
T4. Raising `num_workers` would help, but the work is avoidable rather than
parallelisable -- there are only 66 cell lines, so their pathway expression is
a small fixed table. Packed once it is **31 MB**, fits on any GPU, and each
batch becomes an index gather costing ~6 ms.

```python
from ddprism.data import MonotherapyBatches, MonotherapyTensorData

data = MonotherapyTensorData(model.spec, expression_by_cellline, fingerprints,
                             device="cuda")
batches = MonotherapyBatches(data, cell_ids, drug_ids, dose, viability,
                             batch_size=1024, shuffle=True)

for (genes, fp, dose), y in batches:
    prediction = model(genes, fp, dose)
```

Measured end to end on a T4, per NCI60 epoch:

| setup | epoch |
|---|---|
| published loop + published dataset | ~36 min |
| bucketed model + published dataset | ~12 min |
| bucketed model + `ddprism.data` | **~6.5 min** |

---

### Held-out splits are not optional

The authors publish no ALMANAC train/test split. Without building one,
fine-tuning and the combination model train on rows that belong to the paper's
own test sets, and every ALMANAC metric you get back is optimistic. Our first
runs did exactly this.

`scripts/preprocess.py --stage almanac_splits` builds them, and `require_split()`
makes training **refuse** the unsplit tables rather than quietly use everything.
If training stops with a message naming `almanac_mono_splits`, that is why.

---

## Quick reference

```bash
python -m pytest tests -q                              # the test suite
python kaggle/gpu_check.py --batch 1024 --iters 10     # correctness + speed on GPU
python bench/bench_monotherapy.py --device cuda --backward --batch 1024
python scripts/get_data.py --dest data                 # fetch the 8 required sources (~1 GB)
python scripts/get_data.py --dest data --check         # verify them
python scripts/embed_drugs.py --data data --out processed --model chemberta
python scripts/build_kaggle_notebook.py                # regenerate the six notebooks
```
