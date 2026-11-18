"""Generate kaggle/01_setup_and_data.ipynb, and refuse to write a broken one.

Hand-editing notebook JSON is how a stray escape once turned into a real
newline mid-string and shipped a notebook that could not be parsed. Every code
cell is compiled here before anything is written.

    python scripts/build_kaggle_notebook.py
"""

import ast
import json
import sys
from pathlib import Path

KAGGLE = Path(__file__).resolve().parents[1] / "kaggle"


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": "\n".join(lines)}


def code(*lines):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": "\n".join(lines)}


CELLS = [
md("# DD-PRiSM-plus — Step 1: set up and fetch all data",
   "",
   "Run once in a **CPU** session. Click **Save Version** at the end or everything is lost.",
   "",
   "**Session options (right-hand `<` panel):** Accelerator → **None**, Internet → **On**",
   "",
   "> **figshare is down site-wide** — it answers `202 Accepted` with an empty body,",
   "> `figshare.com` included. Three DepMap files come from there, so supply them",
   "> yourself from the [DepMap 23Q4 downloads page](https://depmap.org/portal/data_page/?tab=allData):",
   ">",
   "> | file | size |",
   "> |---|---|",
   "> | `OmicsExpressionProteinCodingGenesTPMLogp1.csv` | 449.8 MB |",
   "> | `Model.csv` | 0.5 MB |",
   ">",
   "> Upload them as a Kaggle Dataset and attach with **Add Input → Datasets**.",
   "> Step 4 finds them wherever they land."),

code("# 0. Session check",
     "import subprocess, os, glob, shutil",
     "",
     "ok = subprocess.run(['curl', '-sI', '--max-time', '15', 'https://github.com'],",
     "                    capture_output=True).returncode == 0",
     "print('internet:', 'ON' if ok else 'OFF  <-- enable it, then rerun')",
     "print('free disk:', subprocess.run(['df', '-h', '/kaggle/working'],",
     "      capture_output=True, text=True).stdout.splitlines()[-1])"),

md("## 1. Get the code"),

code("REPO = '/kaggle/working/ddprism-plus'",
     "DATA = '/kaggle/working/data'",
     "",
     "if os.path.exists(REPO):",
     "    !cd {REPO} && git pull --quiet",
     "else:",
     "    !git clone --quiet https://github.com/SanaNiroomand/DD-PRiSM-plus.git {REPO}",
     "",
     "os.chdir(REPO)",
     "print('working in', os.getcwd())"),

md("## 2. Install what Kaggle lacks",
   "",
   "`zipfile-deflate64` is **mandatory** — DOSERESP.zip is Deflate64 and the",
   "standard library cannot decompress it."),

code("!pip install --quiet zipfile-deflate64 rdkit openpyxl",
     "print('installed')"),

md("## 3. Check the model code"),

code("!python -m pytest tests -q"),

md("## 4. Copy in the DepMap files you supplied",
   "",
   "Accepts either DepMap naming. Harmless if you have not attached a dataset."),

code("os.makedirs(DATA, exist_ok=True)",
     "",
     "# what you might have uploaded  ->  what the pipeline expects",
     "ALIASES = {",
     "    'OmicsExpressionProteinCodingGenesTPMLogp1.csv':"
     "      'OmicsExpressionProteinCodingGenesTPMLogp1.csv',",
     "    'OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv':"
     "  'OmicsExpressionProteinCodingGenesTPMLogp1.csv',",
     "    'Model.csv':            'Model.csv',",
     "    'sample_info.csv':      'sample_info_18q3.csv',",
     "    'sample_info_18q3.csv': 'sample_info_18q3.csv',",
     "}",
     "",
     "for found, wanted in ALIASES.items():",
     "    target = os.path.join(DATA, wanted)",
     "    if os.path.exists(target):",
     "        continue",
     "    hits = glob.glob('/kaggle/input/**/' + found, recursive=True)",
     "    if hits:",
     "        shutil.copy(hits[0], target)",
     "        print('copied', wanted, ' <- ', hits[0])",
     "        if found != wanted:",
     "            print('  NOTE:', found, 'is a post-23Q4 release. It works, but the')",
     "            print('  paper row counts and metrics were produced with 23Q4.')",
     "",
     "print()",
     "print('contents of', DATA)",
     "!ls -la {DATA} 2>/dev/null || echo '  (empty)'"),

md("## 5. Download everything else (~1 GB)",
   "",
   "Files land **directly in `DATA`**. Already-present files are skipped, so this",
   "is safe to rerun."),

code("!python scripts/get_data.py --dest {DATA} --attempts 3"),

md("## 6. Retry stragglers",
   "",
   "Only useful if figshare has recovered. Skip it if step 4 already supplied both."),

code("!python scripts/get_data.py --dest {DATA} --only depmap_expression depmap_samples --attempts 3"),

md("## 7. Verify — the cell that matters",
   "",
   "Every required row must read `ok`. Size, archive integrity and (for the DepMap",
   "files) the official MD5 are all checked."),

code("!python scripts/get_data.py --dest {DATA} --check",
     "print()",
     "!du -sh {DATA}"),

md("## 8. Save it",
   "",
   "**Save Version → Save & Run All (Commit).** Without this everything here is",
   "deleted when the session ends.",
   "",
   "The next notebook attaches this via **Add Input → Your Work → Notebook Output**.",
   "",
   "---",
   "",
   "**Next:** preprocessing. Success is exactly **7,915,900** NCI60 training rows",
   "and **1,387,317** combination rows."),
]



PREPROCESS_CELLS = [
md("# DD-PRiSM-plus — Step 2: preprocessing",
   "",
   "Turns the raw downloads into training-ready tables. **CPU session** — no GPU",
   "needed, and none should be spent here.",
   "",
   "**Attach step 1's output first:** right panel → **Add Input → Your Work →**",
   "**Notebook Output**, pick your `01_setup_and_data` version.",
   "",
   "Every stage prints the paper's expected count next to the actual one. The two",
   "that decide whether this worked:",
   "",
   "| table | expected |",
   "|---|---|",
   "| NCI60 training rows | **7,915,900** |",
   "| combination training rows | **1,387,317** |"),

code("import os, glob, subprocess",
     "",
     "REPO = '/kaggle/working/ddprism-plus'",
     "OUT  = '/kaggle/working/processed'",
     "",
     "if os.path.exists(REPO):",
     "    !cd {REPO} && git pull --quiet",
     "else:",
     "    !git clone --quiet https://github.com/SanaNiroomand/DD-PRiSM-plus.git {REPO}",
     "os.chdir(REPO)",
     "",
     "# Find step 1's data wherever Kaggle mounted it.",
     "hits = glob.glob('/kaggle/input/**/DOSERESP.zip', recursive=True)",
     "if not hits:",
     "    raise SystemExit('DOSERESP.zip not found under /kaggle/input -- attach '",
     "                     'the output of 01_setup_and_data via Add Input.')",
     "DATA = os.path.dirname(hits[0])",
     "print('data :', DATA)",
     "print('files:', len(os.listdir(DATA)))"),

md("## Install"),

code("!pip install --quiet zipfile-deflate64 rdkit pyarrow",
     "print('installed')"),

md("## Verify the inputs before spending time on them"),

code("!python scripts/get_data.py --dest {DATA} --check"),

md("## Run preprocessing",
   "",
   "Roughly 10-20 minutes. DOSERESP is read in chunks, six columns at tight",
   "dtypes, so peak memory stays in the hundreds of MB rather than the 11.1 GB a",
   "naive full read would cost."),

code("!python scripts/preprocess.py --data {DATA} --out {OUT}"),

md("The `almanac_splits` stage above builds the held-out sets Tables S3 and S4",
   "describe. Training refuses to run without them, because the unsplit tables",
   "contain the very rows the paper scores on."),

md("## Pretrained drug embeddings",
   "",
   "The paper describes a drug as a 512-bit Morgan fingerprint and names the",
   "consequence itself: *\"we need more informative drug features for the",
   "phenotypic prediction.\"* Our held-out numbers agree — the unseen-drug split",
   "scores PCC 0.7585 against 0.9386 on unseen pairs.",
   "",
   "This embeds every drug with ChemBERTa, a transformer pretrained on 77M",
   "PubChem molecules, so training can use `morgan+chemberta`.",
   "",
   "**Needs Internet → On** (right panel) to download the model, ~15 MB. It runs",
   "on CPU in roughly 20 minutes — no GPU quota spent.",
   "",
   "**Order matters:** run this *after* preprocessing. It reads the response",
   "tables to embed only the ~51k drugs the study touches, instead of all",
   "281,264 compounds in Chem2D."),

code("!pip install --quiet transformers",
     "!python scripts/embed_drugs.py --data {DATA} --out {OUT} --model chemberta"),

md("## What came out"),

code("!ls -la {OUT} && du -sh {OUT}"),

md("## Save it",
   "",
   "**Save Version → Save & Run All (Commit)**, then attach this output to the",
   "training notebook.",
   "",
   "If a count is off, send me the numbers — every stage prints what the paper",
   "expected, so a mismatch says which step drifted."),
]



# --------------------------------------------------------------------------
# training notebooks -- one per experiment
# --------------------------------------------------------------------------
#
# Each experiment is its own notebook, so each is its own Kaggle notebook with
# its own version history, its own saved output and its own checkpoints. That
# keeps the runs genuinely independent: nothing one experiment does can reach
# into another's directory, and a rerun of one never disturbs another. The
# comparison happens afterwards, in 06_compare, from the saved outputs.

CHEMBERTA_NEEDS = {
    "file": "drug_embeddings_chemberta.parquet",
    "source": "02_preprocess",
    "build": "python scripts/embed_drugs.py --model chemberta",
    "external": False,
}
L1000_NEEDS = {
    "file": "drug_signatures_l1000.parquet",
    "source": "02b_build_l1000",
    "build": "python scripts/fetch_l1000.py",
    "external": True,
}


EXPERIMENTS = [
    {
        "file": "03_train.ipynb",
        "features": "morgan",
        "curve_width": 2,
        "runs": "/kaggle/working/runs",
        "title": "Step 3: train the paper's model",
        "what": "The published representation: a **512-bit Morgan fingerprint**, "
                "radius 2. This is the baseline every other experiment is "
                "measured against, and it is the run that reproduces the paper.",
        "expect": "Reproduces the published numbers. On the unseen pair set we "
                  "measured RMSE 0.0828 / PCC 0.9386 against the paper's "
                  "0.0830 / 0.9387.",
        "needs": None,
        "repair": ["finetune", "combination"],
    },
    {
        "file": "04_experiment_fusion.ipynb",
        "features": "morgan+chemberta",
        "curve_width": 2,
        "runs": "/kaggle/working/runs-morgan-chemberta",
        "title": "Experiment 1: Morgan + ChemBERTa",
        "what": "The fingerprint **concatenated with a 384-dim ChemBERTa "
                "embedding** -- 896 inputs in total. ChemBERTa is a transformer "
                "pretrained on 77M PubChem molecules, so it brings chemistry a "
                "bag of substructures cannot express.",
        "expect": "The number to watch is **unseen_drug**. The baseline scores "
                  "RMSE 0.1604 / PCC 0.7585 there, against 0.0828 / 0.9386 on "
                  "unseen pairs -- new drugs are where the fingerprint fails, and "
                  "the only place a richer representation has obvious room to "
                  "help. Expect unseen_cellline and unseen_pair to barely move; "
                  "they are already above 0.93 and are not limited by this.",
        "needs": CHEMBERTA_NEEDS,
    },
    {
        "file": "05_experiment_chemberta.ipynb",
        "features": "chemberta",
        "curve_width": 2,
        "runs": "/kaggle/working/runs-chemberta",
        "title": "Experiment 2: ChemBERTa alone",
        "what": "The embedding **without** the fingerprint -- 384 inputs. This is "
                "the control, and it is the one that makes the result "
                "defensible.",
        "expect": "Fusion adds 147,456 parameters (+5.7%) over the paper's model, "
                  "so a sceptic can always say a win came from the extra "
                  "capacity. This model has **49,152 fewer** parameters than the "
                  "paper's. If it also beats the baseline, capacity is not the "
                  "explanation and the representation is. Run it after the fusion "
                  "experiment.",
        "needs": CHEMBERTA_NEEDS,
    },
    {
        "file": "07_experiment_curve_head.ipynb",
        "features": "morgan",
        "curve_width": 8,
        "runs": "/kaggle/working/runs-curve8",
        "title": "Experiment 3: widen the curve-prediction head",
        "what": "The paper's drug representation, unchanged, with **one layer "
                "widened**. The published model ends `Linear(8, 2) -> "
                "BatchNorm1d(2) -> ReLU` and reads all four Hill-curve "
                "parameters off that 2-dimensional non-negative vector. This "
                "makes it 8 wide, matching the layer directly above it.",
        "expect": "Measured on the trained baseline, the four curve parameters "
                  "span **rank 1 of 4** and half the bottleneck is zero -- the "
                  "model cannot set a curve's height, steepness and midpoint "
                  "independently. The check that matters is not RMSE but "
                  "`scripts/measure_curve_rank.py`: if the rank does not move, "
                  "the change did nothing and must not be reported as if it "
                  "did. Compare against `03_train`, which differs only in this "
                  "one width.",
        "needs": None,
    },
    {
        "file": "08_experiment_l1000.ipynb",
        "features": "morgan+l1000",
        "curve_width": 2,
        "runs": "/kaggle/working/runs-morgan-l1000",
        "title": "Experiment 4: Morgan + L1000",
        "what": "The fingerprint **concatenated with a 978-gene L1000 "
                "signature** -- 1,490 inputs in total. L1000 describes a drug "
                "by the transcriptional change it induces in cells, so it "
                "carries what a molecule *does*, not only what it is built from.",
        "expect": "Watch **unseen_drug**, as with the ChemBERTa experiment: the "
                  "baseline scores PCC 0.7585 there against 0.9386 on unseen "
                  "pairs. A measured cell-effect signature is the most direct "
                  "description of 'what this drug does', so this is where it has "
                  "room to help. The catch is coverage -- only drugs that have "
                  "an L1000 signature are scored, so read the coverage line that "
                  "02b_build_l1000 printed first.",
        "needs": L1000_NEEDS,
    },
    {
        "file": "09_experiment_l1000_only.ipynb",
        "features": "l1000",
        "curve_width": 2,
        "runs": "/kaggle/working/runs-l1000",
        "title": "Experiment 5: L1000 alone",
        "what": "The L1000 signature **without** the fingerprint -- 978 inputs. "
                "The control for the fusion experiment: if the signature alone "
                "carries the drug, the structural fingerprint was not what "
                "helped.",
        "expect": "Compare against Morgan on **unseen_drug**, over the same "
                  "covered drugs. A signature is a measured response rather than "
                  "a pretrained guess, so it can help where a fingerprint cannot "
                  "-- or prove too noisy, which is itself a finding. Run it after "
                  "the fusion experiment.",
        "needs": L1000_NEEDS,
    },
]


def train_cells(experiment):
    """Cells for one experiment's notebook.

    Every experiment gets the same machinery -- the differences are the feature
    spec, the run directory, whether it depends on a prebuilt feature file, and
    what the markdown says to look at.
    """
    features = experiment["features"]
    runs = experiment["runs"]
    folder = runs.rsplit("/", 1)[-1]

    needs = experiment.get("needs")
    # A fusion run puts Morgan first, so a Morgan checkpoint can warm-start it.
    is_fusion = bool(needs) and features.startswith("morgan+")
    # An external feature file is built by its own notebook, not 02_preprocess,
    # so it arrives as a separate attachment and has to be linked in.
    external = bool(needs) and needs.get("external")

    needs_note = ([
        "",
        f"**Requires `{needs['file']}` from `{needs['source']}`.** If it is",
        "missing, training stops at once and tells you how to build it -- it",
        "does not quietly fall back to Morgan.",
    ] if needs else [])

    assemble_cells = ([
md("## Put the feature file beside the preprocessed tables",
   "",
   f"`{needs['source']}` wrote `{needs['file']}` as its own output, and training",
   "reads every feature source from one `--processed` directory. This links the",
   "preprocessed tables and that file into one writable folder, so attach",
   f"**both** the `02_preprocess` output **and** the `{needs['source']}` output",
   "via **Add Input**."),
code("import glob, os",
     "",
     "MERGED = '/kaggle/working/processed'",
     "os.makedirs(MERGED, exist_ok=True)",
     "for name in os.listdir(PROCESSED):",
     "    link = os.path.join(MERGED, name)",
     "    if not os.path.exists(link):",
     "        os.symlink(os.path.join(PROCESSED, name), link)",
     "",
     f"FEATURE_FILE = {needs['file']!r}",
     "hits = glob.glob('/kaggle/input/**/' + FEATURE_FILE, recursive=True)",
     "if not hits:",
     f"    raise SystemExit(FEATURE_FILE + ' not found -- attach the {needs['source']} '",
     "                     'output via Add Input.')",
     "destination = os.path.join(MERGED, os.path.basename(hits[0]))",
     "if not os.path.exists(destination):",
     "    os.symlink(hits[0], destination)",
     "PROCESSED = MERGED",
     "print('merged processed dir:', PROCESSED)",
     "print('feature file        :', hits[0])"),
    ] if external else [])

    check_cells_block = ([
md("## Check the feature file is present before spending GPU time"),
code("import pandas as pd",
     f"path = os.path.join(PROCESSED, {needs['file']!r})",
     "if not os.path.exists(path):",
     "    raise SystemExit(",
     f"        {needs['file']!r} + ' is missing from ' + PROCESSED +",
     f"        '\\nBuild it in {needs['source']} ({needs['build']}) and attach that '",
     "        'output here.')",
     "feature = pd.read_parquet(path)",
     "print('feature file:', feature.shape)",
     "print('drugs       :', f'{len(feature):,}')"),
    ] if needs else [])

    warm_start_md = ([
md("### Warm start — only if the budget forces it",
   "",
   "Widening the drug input means the Morgan checkpoint cannot simply be",
   "loaded: `drug_block` gains input columns. `--init-from` copies the Morgan",
   "columns it can and starts the new ones at **zero**, so at step zero the",
   "network computes exactly what the Morgan model computed and the new feature",
   "has to earn its weight from there. Two tests pin that equivalence.",
   "",
   "It turns ~6 GPU-hours of pretraining into a fraction of that — but a",
   "warm-started run is not an independent one, and has to be reported as a",
   "warm start. **Leave `INIT_FROM = None` unless you are out of quota.**")]
   if is_fusion else [])

    init_from_lines = (["",
         "# None = train the drug branches from scratch, which is the clean",
         "# comparison. To warm-start instead, attach a Morgan run and uncomment:",
         "INIT_FROM = None",
         "# INIT_FROM = glob.glob('/kaggle/input/**/runs/pretrain/best.pt', recursive=True)[0]"]
        if is_fusion else ["", "INIT_FROM = None"])

    return [
md(f"# DD-PRiSM-plus — {experiment['title']}",
   "",
   f"**Drug representation: `{features}`**",
   "",
   experiment["what"],
   *needs_note,
   "",
   "**GPU session.** Accelerator → **GPU T4 x2**, Internet → **On**.",
   "",
   "**Attach step 2's output:** right panel → **Add Input → Your Work →**",
   "**Notebook Output**, pick your `02_preprocess` version.",
   "",
   "### What to look for",
   "",
   experiment["expect"],
   "",
   "---",
   "",
   "Three stages run in order, all on the authors' own model classes from",
   "`original/ddprism_original.py`:",
   "",
   "| stage | data | learning rate |",
   "|---|---|---|",
   "| pretrain | NCI60, ~8.9M rows | 1e-2 |",
   "| finetune | ALMANAC monotherapy, 35k rows, curve prediction network unfrozen | 1e-3 |",
   "| combination | ALMANAC pairs, 1.98M rows | 1e-2 |",
   "",
   "> **This may not finish in one session.** Kaggle stops you at ~12 hours.",
   "> Every epoch is checkpointed, so just Save Version, attach *this* notebook's",
   "> output next time, and rerun — it resumes at the epoch it reached."),

md("## Check the GPU is actually attached",
   "",
   "A freshly imported notebook defaults to **no accelerator**, and an exhausted",
   "GPU quota falls back to CPU silently. Either way training still *works* --",
   "it just takes about 18x longer, which is how an experiment quietly spends",
   "eleven hours reaching six epochs. This refuses to start instead."),

code("import torch",
     "",
     "if not torch.cuda.is_available():",
     "    print('No GPU attached, refusing to start.')",
     "    print('  Right panel -> Session options -> Accelerator -> GPU T4 x2.')",
     "    print('  If that is already set, your weekly GPU quota may be spent;')",
     "    print('  Kaggle then falls back to CPU without saying so.')",
     "    print('  Check kaggle.com/settings.')",
     "    print('  On CPU an NCI60 epoch takes ~107 min instead of ~6.')",
     "    raise SystemExit(1)",
     "",
     "print('gpu    :', torch.cuda.get_device_name(0))",
     "print('memory :', f'{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')"),

code("import os, glob",
     "",
     "REPO = '/kaggle/working/ddprism-plus'",
     "",
     f"FEATURES = {features!r}",
     f"RUNS     = {runs!r}",
     "",
     "# This notebook owns this directory and no other. Two representations",
     "# produce models of different shapes, so sharing one would try to resume a",
     "# 512-wide model from a wider checkpoint -- or silently overwrite the",
     "# baseline this experiment is meant to be measured against.",
     "print('features:', FEATURES)",
     "print('runs    :', RUNS)",
     "",
     "if os.path.exists(REPO):",
     "    !cd {REPO} && git pull --quiet",
     "else:",
     "    !git clone --quiet https://github.com/SanaNiroomand/DD-PRiSM-plus.git {REPO}",
     "os.chdir(REPO)",
     "",
     "hits = glob.glob('/kaggle/input/**/nci60_filtered.parquet', recursive=True)",
     "if not hits:",
     "    raise SystemExit('preprocessed data not found -- attach the output of '",
     "                     '02_preprocess via Add Input.')",
     "PROCESSED = os.path.dirname(hits[0])",
     "",
     "raw = glob.glob('/kaggle/input/**/DOSERESP.zip', recursive=True)",
     "if not raw:",
     "    raise SystemExit('raw data not found -- attach 01_setup_and_data too, '",
     "                     'for the KEGG .gmt.')",
     "DATA = os.path.dirname(raw[0])",
     "",
     "print('processed:', PROCESSED)",
     "print('raw      :', DATA)"),

*assemble_cells,

*check_cells_block,

md("## Resume from a previous session",
   "",
   "**Save & Run All always starts a fresh container**, so a previous run's",
   "results only survive if you feed them back in: **Add Input → Your Work →**",
   "**Notebook Output**, and pick this notebook's own earlier version. This cell",
   "then copies its checkpoints into place. Does nothing on a first run.",
   "",
   "`REDO` deletes a stage's checkpoint so it trains from scratch, while keeping",
   "the stages before it. Without that, a restored checkpoint is resumed -- a",
   "stage that already early-stopped would run one epoch, stop again, and change",
   "nothing. Its `best.pt` is kept, since the next stage loads weights from it",
   "until the retrain overwrites them."),

*( [md("### One-time repair: " + ", ".join(experiment["repair"]),
       "",
       "**`REPAIR = True` below, and it should stay that way until you have run",
       "this once.**",
       "",
       "These stages were first trained before the ALMANAC leakage fix, when",
       "training still used the unsplit tables. Re-running afterwards did not",
       "correct them: both stages had already early-stopped, so they resumed,",
       "ran a single epoch, stopped again, and kept the contaminated weights.",
       "",
       "It shows up as an impossible ordering in the held-out scores -- the",
       "fine-tuned model scored **PCC 0.9526 on unseen drugs against 0.9082 on",
       "unseen pairs**. A drug the model has never seen cannot be easier than a",
       "new pairing of drugs it knows. Those rows were in its training set.",
       "",
       "Pretraining is unaffected -- the NCI60 splits existed all along, and its",
       "ordering is correct -- so this reuses `pretrain/best.pt` and retrains",
       "only the two damaged stages. About 15 minutes, not 7 hours.",
       "",
       "**Set `REPAIR = False` afterwards**, or every later run throws these two",
       "stages away and retrains them for nothing.")]
   if experiment.get("repair") else [] ),

code("import shutil",
     "",
     *( [f"# One-time repair of {', '.join(experiment['repair'])} -- see above.",
         "# Set to False once it has run.",
         "REPAIR = True",
         "",
         f"REDO = {experiment['repair']!r} if REPAIR else []"]
        if experiment.get("repair") else
        ["REDO = []          # e.g. ['finetune', 'combination'] to retrain those two"] ),
     "",
     "os.makedirs(RUNS, exist_ok=True)",
     "restored = 0",
     f"# Only this experiment's own checkpoints, from {folder}/ and nowhere else.",
     f"pattern = '/kaggle/input/**/{folder}/*/checkpoint.pt'",
     "for found in glob.glob(pattern, recursive=True):",
     "    stage = os.path.basename(os.path.dirname(found))",
     "    os.makedirs(os.path.join(RUNS, stage), exist_ok=True)",
     "    for name in ('checkpoint.pt', 'best.pt', 'history.json'):",
     "        source = os.path.join(os.path.dirname(found), name)",
     "        if os.path.exists(source):",
     "            shutil.copy(source, os.path.join(RUNS, stage, name))",
     "            restored += 1",
     f"for config in glob.glob('/kaggle/input/**/{folder}/config.json', recursive=True):",
     "    shutil.copy(config, os.path.join(RUNS, 'config.json'))",
     "print('restored', restored, 'checkpoint files')",
     "",
     "for stage in REDO:",
     "    stale = os.path.join(RUNS, stage, 'checkpoint.pt')",
     "    if os.path.exists(stale):",
     "        os.remove(stale)",
     "        print('cleared', stage, '-- it will train from scratch')",
     "",
     "!ls -R {RUNS} 2>/dev/null | head -20"),

md("## Install and check"),

code("!pip install --quiet zipfile-deflate64 pyarrow",
     "!python -m pytest tests -q -x --ignore=tests/test_train.py"),

md("## Train",
   "",
   "`original` is the authors' loop over 186 pathways; `fast` batches those 186",
   "steps and is pinned to the published model at 1e-10 by the test suite, and",
   "measured at 1.3e-15 on a T4. They compute the same thing.",
   "",
   "The difference is only speed, but at this scale speed decides whether you",
   "finish. Measured on a T4, per NCI60 epoch:",
   "",
   "| model | min/epoch | epochs in 30 GPU-hours |",
   "|---|---|---|",
   "| `original` | 31.5 | **57** |",
   "| `fast` | 6.0 | **300** |",
   "",
   "`--max-hours` stops cleanly and checkpoints before Kaggle pulls the plug."),

*warm_start_md,

code("MODEL = 'fast'      # 'original' for the authors' loop, 'fast' for the same maths batched",
     *( [f"STAGE = {' '.join(experiment['repair'])!r} if REPAIR else 'all'"]
        if experiment.get("repair") else
        ["STAGE = 'all'       # or e.g. 'finetune combination' to redo just those two"] ),
     *init_from_lines,
     "",
     f"CURVE_WIDTH = {experiment.get('curve_width', 2)}"
     f"{'      # published' if experiment.get('curve_width', 2) == 2 else '      # published value is 2'}",
     "",
     "ARGS = (f'--data {DATA} --processed {PROCESSED} --out {RUNS} '",
     "        f'--model {MODEL} --stage {STAGE} --drug-features {FEATURES} '",
     "        f'--curve-width {CURVE_WIDTH}')",
     "if INIT_FROM:",
     "    ARGS += f' --init-from {INIT_FROM}'",
     "",
     "!python -m ddprism.train {ARGS} --batch-size 1024 --max-hours 10.5"),

md("## Training curve"),

code("import json",
     "for stage in ('pretrain', 'finetune', 'combination'):",
     "    path = os.path.join(RUNS, stage, 'history.json')",
     "    if not os.path.exists(path):",
     "        continue",
     "    history = json.load(open(path))",
     "    best = min(history, key=lambda h: h['val_loss'])",
     "    print(f\"{stage:<12} {len(history):>3} epochs   best val {best['val_loss']:.5f}  \"",
     "          f\"RMSE {best['val_rmse']:.4f}  PCC {best['val_pcc']:.4f}\")",
     "print()",
     "print('paper, unseen pair set:')",
     "print('  pretrained  RMSE 0.0830  PCC 0.9387')",
     "print('  fine-tuned  RMSE 0.0914  PCC 0.8791')",
     "print('  combination RMSE 0.0854  PCC 0.9063')"),

md("## Score on the held-out sets — the numbers to quote",
   "",
   "Training reports on a random slice of its own pool, which splits individual",
   "measurements: the same drug on the same cell line can be in training at one",
   "dose and validation at another. The paper's figures hold out whole entities.",
   "This scores the trained model on those, so the comparison is like for like.",
   "",
   "No training, one forward pass per split."),

code("!python -m ddprism.evaluate --data {DATA} --processed {PROCESSED} "
     "--runs {RUNS} --drug-features {FEATURES}"),

md("## The curve-prediction bottleneck, measured",
   "",
   "All four Hill-curve parameters are read off one vector at the end of",
   "`final_block`. The published width is 2, and on the trained baseline the",
   "four outputs span **rank 1 of 4** -- one number wearing four hats -- with",
   "half the vector dead from the `ReLU`.",
   "",
   "This measures it on whatever this run produced. For a run that widened the",
   "head, **the rank is the result**, not the RMSE: if it did not move, the",
   "change did nothing and must not be reported as though it had."),

code("!python scripts/measure_curve_rank.py --data {DATA} "
     "--processed {PROCESSED} --runs {RUNS} --stage pretrain"),

md("## Save",
   "",
   "**Save Version → Save & Run All (Commit).**",
   "",
   "This notebook's output holds `evaluation.json` for this experiment alone.",
   "`06_compare` reads it, together with the other experiments', and puts them",
   "side by side.",
   "",
   "If training stopped on the time budget rather than finishing, attach this",
   "notebook's own output next session and rerun. It picks up mid-stage."),
]


COMPARE_CELLS = [
md("# DD-PRiSM-plus — Compare the experiments",
   "",
   "**CPU session** — this only reads saved results, and should cost no GPU.",
   "",
   "**Attach every experiment's output:** right panel → **Add Input → Your Work",
   "→ Notebook Output**, once per notebook you want in the table:",
   "",
   "| notebook | representation |",
   "|---|---|",
   "| `03_train` | `morgan` — the paper |",
   "| `04_experiment_fusion` | `morgan+chemberta` |",
   "| `05_experiment_chemberta` | `chemberta` |",
   "| `07_experiment_curve_head` | `morgan`, curve head widened to 8 |",
   "| `08_experiment_l1000` | `morgan+l1000` |",
   "| `09_experiment_l1000_only` | `l1000` |",
   "",
   "Whichever are attached will appear. Missing ones are simply left out."),

code("import glob, json, os",
     "",
     "found = sorted(glob.glob('/kaggle/input/**/runs*/evaluation.json',",
     "                         recursive=True))",
     "results, configs = {}, {}",
     "for path in found:",
     "    directory = os.path.dirname(path)",
     "    label = os.path.basename(directory)",
     "    label = 'morgan' if label == 'runs' else label.replace('runs-', '')",
     "    results[label] = json.load(open(path))",
     "    config = os.path.join(directory, 'config.json')",
     "    if os.path.exists(config):",
     "        configs[label] = json.load(open(config))",
     "",
     "if not results:",
     "    raise SystemExit('no evaluation.json found -- attach the experiment '",
     "                     'notebook outputs via Add Input.')",
     "",
     "# Baseline first; everything else is read as a change from it.",
     "results = dict(sorted(results.items(), key=lambda kv: kv[0] != 'morgan'))",
     "",
     "print(f\"{'experiment':<20}{'drug input':>12}{'warm start':>12}\")",
     "for label in results:",
     "    config = configs.get(label, {})",
     "    started = 'yes' if config.get('init_from') else 'no'",
     "    print(f\"{label:<20}{config.get('drug_dim', '?'):>12}{started:>12}\")"),

md("## Held-out results, side by side",
   "",
   "**`unseen_drug` is the column this whole line of work is about.** The paper",
   "names the limitation itself — *\"we need more informative drug features for",
   "the phenotypic prediction\"* — and the baseline scores PCC 0.7585 there",
   "against 0.9386 on unseen pairs.",
   "",
   "`unseen_cellline` and `unseen_pair` are not expected to move much. They are",
   "already above 0.93 and are not limited by how the drug is described."),

code("STAGES = ('pretrain', 'finetune', 'combination')",
     "labels = list(results)",
     "",
     "for stage in STAGES:",
     "    rows = {k: v[stage] for k, v in results.items() if stage in v}",
     "    if not rows:",
     "        continue",
     "    splits = sorted({s for r in rows.values() for s in r})",
     "    print(f'\\n{stage}')",
     "    print(f\"  {'split':<18}\" + ''.join(f'{k:>24}' for k in rows))",
     "    print(f\"  {'':<18}\" + ''.join(f\"{'RMSE':>12}{'PCC':>12}\" for _ in rows))",
     "    for split in splits:",
     "        line = f'  {split:<18}'",
     "        for label in rows:",
     "            got = rows[label].get(split)",
     "            line += (f\"{got['rmse']:>12.4f}{got['pcc']:>12.4f}\" if got",
     "                     else ' ' * 24)",
     "        print(line)"),

md("## Did it actually help?",
   "",
   "Change relative to the Morgan baseline, on every split. A negative RMSE",
   "delta and a positive PCC delta are improvements."),

code("base = results.get('morgan')",
     "if base is None:",
     "    print('no morgan baseline attached -- nothing to compare against')",
     "else:",
     "    for label, values in results.items():",
     "        if label == 'morgan':",
     "            continue",
     "        print(f'\\n{label}  vs  morgan')",
     "        for stage in STAGES:",
     "            if stage not in values or stage not in base:",
     "                continue",
     "            for split in sorted(values[stage]):",
     "                if split not in base[stage]:",
     "                    continue",
     "                got, ref = values[stage][split], base[stage][split]",
     "                d_rmse = got['rmse'] - ref['rmse']",
     "                d_pcc = got['pcc'] - ref['pcc']",
     "                mark = '  <-- better' if (d_rmse < 0 and d_pcc > 0) else ''",
     "                print(f'  {stage:<12}{split:<18}'",
     "                      f'RMSE {d_rmse:+.4f}   PCC {d_pcc:+.4f}{mark}')"),

md("## A caution on reading this",
   "",
   "One run per representation is one sample. A difference of a few thousandths",
   "of RMSE is inside the noise of a different random seed and should not be",
   "reported as an improvement. What would be a real result is `unseen_drug`",
   "moving by a visible margin — its gap to `unseen_pair` is currently about",
   "**0.18 PCC**, which is far larger than seed noise.",
   "",
   "If the margin turns out to be small, the honest report is that a pretrained",
   "embedding did not help here, which is a finding rather than a failure."),
]


BUILD_L1000_CELLS = [
md("# DD-PRiSM-plus — Build the L1000 drug signatures",
   "",
   "Produces `drug_signatures_l1000.parquet`: one 978-landmark-gene transcriptomic",
   "signature per drug, from **CMap / LINCS 2020** (the newer, larger release),",
   "mapped to NSC by InChIKey. **CPU session, Internet On** — no GPU needed.",
   "",
   "Attach two inputs via **Add Input**:",
   "",
   "1. Your **01_setup_and_data** output — for `nsc_smiles.csv`.",
   "2. A **Kaggle Dataset holding the big matrix** `level5_beta_trt_cp_n720216x12328.gctx`",
   "   (for example the public *lincs-data* dataset). Attaching it as a Dataset keeps",
   "   it in read-only `/kaggle/input`, off your working-disk quota.",
   "",
   "The three small metadata tables are downloaded automatically from CLUE below —",
   "you do **not** need to upload those."),

code("import os, glob, subprocess",
     "",
     "REPO = '/kaggle/working/ddprism-plus'",
     "OUT  = '/kaggle/working/l1000'",
     "",
     "ok = subprocess.run(['curl', '-sI', '--max-time', '15', 'https://github.com'],",
     "                    capture_output=True).returncode == 0",
     "print('internet:', 'ON' if ok else 'OFF  <-- enable it, then rerun')",
     "",
     "if os.path.exists(REPO):",
     "    !cd {REPO} && git pull --quiet",
     "else:",
     "    !git clone --quiet https://github.com/SanaNiroomand/DD-PRiSM-plus.git {REPO}",
     "os.chdir(REPO)",
     "print('working in', os.getcwd())"),

md("## Install",
   "",
   "`cmapPy` reads the `.gctx`; `rdkit` canonicalises SMILES to InChIKeys for the",
   "NSC↔L1000 mapping."),

code("!pip install --quiet cmapPy rdkit pyarrow",
     "print('installed')"),

md("## Gather the files",
   "",
   "The three small CMap 2020 metadata tables are fetched from CLUE; the big",
   "`.gctx` is found in whatever dataset you attached. Everything is linked into",
   "one folder so the builder sees a single release together."),

code("import glob, os",
     "",
     "LINCS = '/kaggle/working/lincs'",
     "DATA  = '/kaggle/working/raw'",
     "os.makedirs(LINCS, exist_ok=True)",
     "os.makedirs(DATA, exist_ok=True)",
     "",
     "CLUE = 'https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020'",
     "for name in ('geneinfo_beta.txt', 'siginfo_beta.txt', 'compoundinfo_beta.txt'):",
     "    target = os.path.join(LINCS, name)",
     "    if not os.path.exists(target):",
     "        !wget -q -O {target} {CLUE}/{name}",
     "    print(name, os.path.getsize(target), 'bytes')",
     "",
     "# the big matrix comes from the attached Kaggle dataset",
     "gctx = glob.glob('/kaggle/input/**/level5_beta_trt_cp*.gctx', recursive=True)",
     "if not gctx:",
     "    raise SystemExit('level5_beta_trt_cp*.gctx not found -- attach a Kaggle '",
     "                     'dataset that holds it via Add Input.')",
     "link = os.path.join(LINCS, os.path.basename(gctx[0]))",
     "if not os.path.exists(link):",
     "    os.symlink(gctx[0], link)",
     "print('gctx:', gctx[0])",
     "",
     "# nsc_smiles.csv from the 01_setup_and_data output",
     "smiles = glob.glob('/kaggle/input/**/nsc_smiles.csv', recursive=True)",
     "if not smiles:",
     "    raise SystemExit('nsc_smiles.csv not found -- attach the 01_setup_and_data '",
     "                     'output via Add Input.')",
     "dst = os.path.join(DATA, 'nsc_smiles.csv')",
     "if not os.path.exists(dst):",
     "    os.symlink(smiles[0], dst)",
     "print('nsc_smiles:', smiles[0])"),

md("## Build the signatures",
   "",
   "Reads only the 978 landmark rows and the signatures of drugs it can map to an",
   "NSC, averages each drug's signatures into one consensus vector, and writes the",
   "parquet. `--all-compounds` maps every NSC with a SMILES.",
   "",
   "**Read the coverage line it prints** — it says how many of the drugs got an",
   "L1000 signature. That is the honest scope of the whole L1000 experiment."),

code("!python scripts/fetch_l1000.py --lincs {LINCS} --data {DATA} --out {OUT} --all-compounds"),

md("## What came out"),

code("import pandas as pd, os",
     "frame = pd.read_parquet(os.path.join(OUT, 'drug_signatures_l1000.parquet'))",
     "print('shape:', frame.shape)",
     "print('drugs:', f'{len(frame):,}')",
     "frame.iloc[:3, :6]"),

md("## Save it",
   "",
   "**Save Version → Save & Run All (Commit).** The `08_experiment_l1000` and",
   "`09_experiment_l1000_only` notebooks attach this output alongside the",
   "`02_preprocess` output, via **Add Input → Your Work → Notebook Output**."),
]


def _as_python(source):
    """Make a notebook cell parseable as plain Python.

    IPython magics (`!cmd`, `%cd`) are not Python. Blanking them breaks any
    block they live in -- `if x:` followed by a blank line is a SyntaxError --
    so substitute `pass` at the same indentation and keep the structure.
    """
    lines = []
    for line in source.split("\n"):
        stripped = line.lstrip()
        if stripped.startswith(("!", "%")):
            lines.append(" " * (len(line) - len(stripped)) + "pass")
        else:
            lines.append(line)
    return "\n".join(lines)


def check_cells(cells):
    problems = []
    for index, cell in enumerate(cells):
        if cell["cell_type"] != "code":
            continue
        try:
            ast.parse(_as_python(cell["source"]))
        except SyntaxError as error:
            problems.append(f"cell {index}: line {error.lineno}: {error.msg}")
    return problems


def write(cells, name):
    problems = check_cells(cells)
    if problems:
        print(f"REFUSING TO WRITE {name} -- code cells do not parse:")
        for problem in problems:
            print("  " + problem)
        return False

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    path = KAGGLE / name
    path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    code_cells = sum(1 for c in cells if c["cell_type"] == "code")
    print(f"wrote kaggle/{name} ({len(cells)} cells, {code_cells} code, all parse)")
    return True


def main():
    ok = write(CELLS, "01_setup_and_data.ipynb")
    ok = write(PREPROCESS_CELLS, "02_preprocess.ipynb") and ok
    ok = write(BUILD_L1000_CELLS, "02b_build_l1000.ipynb") and ok
    for experiment in EXPERIMENTS:
        ok = write(train_cells(experiment), experiment["file"]) and ok
    ok = write(COMPARE_CELLS, "06_compare.ipynb") and ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
