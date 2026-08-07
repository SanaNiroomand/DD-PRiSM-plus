# Running DD-PRiSM-plus on Kaggle GPU

Three phases. Phase 1 needs no data and takes about ten minutes -- do it first,
because it tells us whether the vectorised model is worth using on a GPU before
we invest any effort in the data.

---

## Phase 1 -- GPU check (no data needed, ~10 min)

### 1. Prepare your Kaggle account

- Sign in at <https://www.kaggle.com>
- Go to **Settings -> Phone Verification** and verify your phone number.
  Without this you cannot use a GPU or turn on internet access.

### 2. Create the notebook

- **Create -> Notebook**
- Open the right-hand panel (the `<` arrow, top right)
- **Session options -> Accelerator -> GPU T4 x2**
- **Session options -> Internet -> On** (needed for `git clone`)

### 3. Paste and run these cells

Two Jupyter rules worth knowing, because breaking either gives you a confusing
`SyntaxError`: shell commands need a leading `!`, and `%cd` is a notebook magic
that must sit alone on its own line. You cannot chain them with `&&`.

```python
!git clone https://github.com/SanaNiroomand/DD-PRiSM-plus.git /kaggle/working/ddprism-plus
```

```python
%cd /kaggle/working/ddprism-plus
```

```python
!python -m pytest tests -q
```

```python
!python kaggle/gpu_check.py --batch 1024 --iters 10
```

### 4. What to look for

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

## Phase 2 -- Data (entirely on Kaggle)

Every source is scriptable, so nothing needs to be downloaded locally and
re-uploaded. Run this in the same notebook, with **Internet -> On**:

```python
!python scripts/get_data.py --dest /kaggle/working/data
```

That pulls all seven files, about 1 GB, in a few minutes:

| File | Size | Source |
|---|---|---|
| `DOSERESP.zip` (NCI60, **version 10**) | 333 MB | NCI wiki |
| `ComboDrugGrowth_Nov2017.zip` (ALMANAC) | 86 MB | NCI wiki |
| `nsc_smiles.csv` | 17 MB | NCI wiki |
| `OmicsExpressionProteinCodingGenesTPMLogp1.csv` | 450 MB | DepMap 23Q4 via figshare |
| `sample_info_18q3.csv` | 0.06 MB | DepMap 18Q3 via figshare |
| `c2.cp.kegg_legacy...gmt` | 0.1 MB | Broad data host |
| `oneil_combination_response.xls` | 39 MB | AACR |

Then verify:

```python
!python scripts/get_data.py --dest /kaggle/working/data --check
```

Every required row must read `ok`. A `SUSPECT SIZE` usually means an HTML error
page got saved instead of the file.

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

## Quick reference

```bash
python -m pytest tests -q                              # 23 correctness checks
python kaggle/gpu_check.py --batch 1024 --iters 10     # correctness + speed on GPU
python bench/bench_monotherapy.py --device cuda --backward --batch 1024
python scripts/get_data.py --dest data                 # fetch all seven sources (~1 GB)
python scripts/get_data.py --dest data --check         # verify them
```
