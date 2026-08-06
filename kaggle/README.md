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

**Copy the whole SPEED table back.** Whichever bucket count wins there is the
one we use for real training. On CPU the answer is 8-16 buckets at roughly
2x; a GPU should do considerably better, because batching small operations
mainly saves kernel-launch overhead, and that is a GPU cost.

---

## Phase 2 -- Data (the slow part, done on your laptop)

None of this needs a GPU. Preprocessing is ordinary CPU work.

### 1. Download the raw files

| File | Where | Note |
|---|---|---|
| `DOSERESP.csv` (NCI60) | [NCI-60 Growth Inhibition Data](https://wiki.nci.nih.gov/display/NCIDTPdata/NCI-60+Growth+Inhibition+Data) | **Browser only** -- the NCI wiki returns 403 to scripts |
| `ComboDrugGrowth_Nov2017.csv` | [NCI-ALMANAC](https://wiki.nci.nih.gov/display/NCIDTPdata/NCI-ALMANAC) | Browser only |
| `Chem2D_Jun2016.sdf` | [NCI Chemical Data](https://wiki.nci.nih.gov/display/NCIDTPdata/Chemical+Data) | Browser only |
| `OmicsExpressionProteinCodingGenesTPMLogp1.csv` | [DepMap 23Q4](https://depmap.org/portal/download/all/) | Large; pin release 23Q4 |
| `DepMap-2018q3-celllines.csv` | DepMap 18Q3 | Needed for cell-line name matching |
| `c2.cp.kegg_legacy.v2023.2.Hs.symbols.gmt` | [MSigDB](https://www.gsea-msigdb.org/gsea/msigdb/) | Needs a free account now |
| O'Neil supplementary `.xls` | [AACR MCT 15(6):1155](https://aacrjournals.org/mct/article/15/6/1155/92159/) | External validation only -- can wait |

Put them all in one folder and point `base_directory` at it.

### 2. Preprocess

Run `01_Preprocessing.ipynb`. You will need RDKit, which is not installed yet:

```bash
pip install rdkit
```

**Checkpoint:** the supplement's Table S2 says NCI60 must yield **7,915,900**
training responses, and Table S4 says the combination training set must yield
**1,387,317**. If you hit those numbers, preprocessing is correct. Do not skip
this -- everything downstream depends on it.

### 3. Upload to Kaggle

Upload the *preprocessed* inputs, not the raw downloads -- they are far
smaller.

- **Create -> Dataset**, drag in the processed folder, set it **Private**
- In your notebook: **Add Input -> Datasets ->** your dataset
- It appears at `/kaggle/input/<your-dataset-name>/`

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

Set `num_workers=4` in the DataLoader. The published notebooks use
`num_workers=0`, which prepares data on the main thread while the GPU waits. I
measured data preparation at ~50 ms per batch: invisible on CPU (8% of a step),
but roughly 70% of a step once the GPU makes the model part fast. This is a
one-line change and on a GPU it will matter more than anything else here.

---

## Quick reference

```bash
python -m pytest tests -q                              # 17 correctness checks
python kaggle/gpu_check.py --batch 1024 --iters 10     # correctness + speed on GPU
python bench/bench_monotherapy.py --device cuda --backward --batch 1024
```
