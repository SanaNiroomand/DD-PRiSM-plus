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

OUT = Path(__file__).resolve().parents[1] / "kaggle" / "01_setup_and_data.ipynb"


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

md("## 3. Check the model code (23 tests, ~5 s)"),

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

code("!python scripts/get_data.py --dest {DATA} --include-optional --attempts 3"),

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


def main():
    problems = check_cells(CELLS)
    if problems:
        print("REFUSING TO WRITE -- code cells do not parse:")
        for problem in problems:
            print("  " + problem)
        return 1

    notebook = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    OUT.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    code_cells = sum(1 for c in CELLS if c["cell_type"] == "code")
    print(f"wrote {OUT.relative_to(Path.cwd())} "
          f"({len(CELLS)} cells, {code_cells} code, all parse)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
