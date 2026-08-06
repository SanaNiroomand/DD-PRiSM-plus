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

## Phase 2 -- Data

**Preprocess on Kaggle, not on your laptop.** DOSERESP expands to 2.37 GB and
the notebook writes roughly another 1 GB of intermediates. A machine with 6 GB
free cannot do this comfortably, and Windows degrades badly below ~10% free.
Kaggle gives you 20 GB of working disk and ~30 GB of RAM, `/kaggle/input` does
not count against the quota, and the data ends up where training happens.

So only three files ever touch your laptop, and only long enough to upload.

### 1. Fetch what can be automated

```bash
python scripts/get_data.py --dest data
```

That pulls the KEGG pathways, the DepMap 23Q4 expression matrix (450 MB), the
DepMap 18Q3 cell-line annotation, and the O'Neil validation set. All four URLs
were verified on 2026-08-06.

### 2. Download three files by hand

The NCI wiki serves these fine to a browser but returns **403 to any script**,
so they cannot be automated. Save them into `data/Raw/`.

| File | Size | Link |
|---|---|---|
| `DOSERESP.zip` | 333 MB | [NCI-60 Growth Inhibition Data](https://wiki.nci.nih.gov/display/NCIDTPdata/NCI-60+Growth+Inhibition+Data) |
| `ComboDrugGrowth_Nov2017.zip` | 86 MB | [NCI-ALMANAC](https://wiki.nci.nih.gov/display/NCIDTPdata/NCI-ALMANAC) |
| `nsc_smiles.csv` | 17 MB | [NCI Chemical Data](https://wiki.nci.nih.gov/display/NCIDTPdata/Chemical+Data) |

**Take DOSERESP version 10, not the current one.** The July 2026 release
changed format: it now reports one row per experiment (`EXPID`) instead of
aggregating across experiments, so your row counts will not match the paper.
Version 10 (January 2024) is what the authors used and is still available under
"Previous Releases" or by appending `?version=10` to the attachment URL. The
exact URL is printed by `get_data.py`.

`nsc_smiles.csv` is a shortcut: it maps NSC numbers to SMILES directly. The
paper instead parses `Chem2D_Jun2016.zip` (80 MB) with RDKit. Use Chem2D if you
want to match the paper exactly; use `nsc_smiles.csv` if you want it simpler.

### 3. Check before uploading

```bash
python scripts/get_data.py --dest data --check
```

Every row must say `ok`. A `SUSPECT SIZE` usually means you saved an HTML error
page instead of the file.

### 4. Upload to Kaggle and preprocess there

- **Create -> Dataset**, drag in `data/Raw/`, set it **Private** (~1 GB)
- In your notebook: **Add Input -> Datasets ->** your dataset
- Then delete the local copies to get your disk space back

Run the preprocessing on Kaggle. Two notes that will save you time:

- Read DOSERESP straight out of the zip -- `pd.read_csv("DOSERESP.zip")` works
  and avoids writing 2.37 GB to disk.
- The 18Q3 annotation arrives with columns `Broad_ID`, `CCLE_name`, `aliases`.
  The notebook expects `CCLE_Name` and `Aliases`, so rename them first.

**Checkpoint.** The supplement's Table S2 says NCI60 must yield **7,915,900**
training responses; Table S4 says the combination training set must yield
**1,387,317**. Hit those and your preprocessing is right. This is why the
version-10 pin matters -- with the July 2026 file these numbers are meaningless.

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
python scripts/get_data.py --dest data                 # fetch the automatable sources
python scripts/get_data.py --dest data --check         # verify all seven files
```
