"""Fetch and verify the DD-PRiSM source datasets.

Every URL was checked on 2026-08-06 and all of them are scriptable, so this
runs anywhere with an internet connection -- including a Kaggle notebook, which
is the point: nothing has to be downloaded locally and re-uploaded.

    python scripts/get_data.py --dest data           # fetch everything (~1 GB)
    python scripts/get_data.py --dest data --check   # verify what is present
    python scripts/get_data.py --dest data --only nci60_doseresp

A note on the NCI wiki, which hosts three of these. It answers **403 to a HEAD
request but serves a GET perfectly well**, as long as a browser User-Agent is
set. Checking these links with `curl -I` makes them look dead when they are
not.

Version pinning matters. The paper used DOSERESP version 10 (January 2024). The
current version 20 (July 2026 release) reports one row per experiment (EXPID)
instead of aggregating across experiments, so row counts will not match the
supplement's Table S2. Version 10 keeps the published checkpoints meaningful.
"""

import argparse
import sys
import urllib.request
from pathlib import Path

BROWSER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                 "(KHTML, like Gecko) Chrome/140.0 Safari/537.36")

SOURCES = [
    {
        "key": "kegg",
        "filename": "c2.cp.kegg_legacy.v2023.2.Hs.symbols.gmt",
        "url": ("https://data.broadinstitute.org/gsea-msigdb/msigdb/release/"
                "2023.2.Hs/c2.cp.kegg_legacy.v2023.2.Hs.symbols.gmt"),
        "approx_mb": 0.10,
        "note": "186 KEGG legacy pathways. The Broad's data host needs no login, "
                "unlike the MSigDB web portal.",
    },
    {
        "key": "nci60_doseresp",
        "filename": "DOSERESP.zip",
        "url": ("https://wiki.nci.nih.gov/download/attachments/147193864/"
                "DOSERESP.zip?version=10&modificationDate=1704733010000&api=v2"),
        "approx_mb": 333.4,
        "note": "NCI60 dose-response, VERSION 10 to match the paper. Expands to "
                "2.37 GB -- read it straight from the zip rather than extracting.",
    },
    {
        "key": "almanac",
        "filename": "ComboDrugGrowth_Nov2017.zip",
        "url": ("https://wiki.nci.nih.gov/download/attachments/338237347/"
                "ComboDrugGrowth_Nov2017.zip?version=1&"
                "modificationDate=1510057275000&api=v2"),
        "approx_mb": 86.0,
        "note": "NCI-ALMANAC combination responses. Never revised.",
    },
    {
        "key": "nsc_smiles",
        "filename": "nsc_smiles.csv",
        "url": ("https://wiki.nci.nih.gov/download/attachments/155844992/"
                "nsc_smiles.csv?version=11&modificationDate=1783565838764&api=v2"),
        "approx_mb": 17.3,
        "note": "NSC to SMILES directly. Simpler than the paper's route of "
                "parsing Chem2D with RDKit.",
    },
    {
        "key": "depmap_expression",
        "filename": "OmicsExpressionProteinCodingGenesTPMLogp1.csv",
        "url": "https://ndownloader.figshare.com/files/43347204",
        "approx_mb": 449.8,
        "note": "DepMap Public 23Q4, log2(TPM+1) for protein-coding genes.",
    },
    {
        "key": "depmap_samples",
        "filename": "sample_info_18q3.csv",
        "url": "https://ndownloader.figshare.com/files/12704612",
        "approx_mb": 0.06,
        "note": "DepMap 18Q3 cell-line annotation. Columns are Broad_ID, "
                "CCLE_name, aliases -- the notebook expects CCLE_Name and "
                "Aliases, so rename before use.",
    },
    {
        "key": "oneil",
        "filename": "oneil_combination_response.xls",
        "url": ("https://aacr.silverchair-cdn.com/aacr/content_public/journal/mct/"
                "15/6/10.1158_1535-7163.mct-15-0843/2/"
                "15357163mct150843-sup-156849_1_supp_1_w2lrww.xls"),
        "approx_mb": 38.8,
        "note": "O'Neil et al. external validation set. Despite the .xls name "
                "it is really an xlsx, so read it with engine='openpyxl'. Single "
                "sheet 'combination response'; the viability label is 'X/X0'.",
    },
    {
        "key": "chem2d",
        "filename": "Chem2D_Jun2016.zip",
        "url": ("https://wiki.nci.nih.gov/download/attachments/155844992/"
                "Chem2D_Jun2016.zip?version=1&modificationDate=1486993270000&api=v2"),
        "approx_mb": 80.6,
        "optional": True,
        "note": "The SDF the paper parsed with RDKit to get SMILES. Only needed "
                "if you want to reproduce that route exactly rather than using "
                "nsc_smiles.csv.",
    },
]

BY_KEY = {source["key"]: source for source in SOURCES}


def fetch(source, dest):
    """Stream to disk so a 450 MB file never sits fully in memory."""
    target = dest / source["filename"]
    if target.exists() and target.stat().st_size > 1024:
        print(f"  [skip]     {source['filename']} ({target.stat().st_size / 1e6:.1f} MB)")
        return True

    print(f"  [get]      {source['filename']} (~{source['approx_mb']:.1f} MB)", flush=True)
    request = urllib.request.Request(source["url"],
                                     headers={"User-Agent": BROWSER_AGENT})
    partial = target.with_suffix(target.suffix + ".part")
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            first = response.read(512)
            # An HTML body means a login wall or a bot check, not the file.
            if first.lstrip().lower().startswith((b"<!doctype", b"<html")):
                print("             FAILED: got an HTML page, not the file")
                return False
            written = 0
            with partial.open("wb") as handle:
                handle.write(first)
                written += len(first)
                while True:
                    chunk = response.read(1 << 20)
                    if not chunk:
                        break
                    handle.write(chunk)
                    written += len(chunk)
                    if written % (50 << 20) < (1 << 20):
                        print(f"             {written / 1e6:.0f} MB...", flush=True)
    except Exception as error:
        print(f"             FAILED: {error}")
        partial.unlink(missing_ok=True)
        return False

    partial.replace(target)
    print(f"             saved {written / 1e6:.1f} MB")
    return True


def check(dest):
    print(f"Checking {dest.resolve()}\n")
    missing = []
    for source in SOURCES:
        target = dest / source["filename"]
        tag = " (optional)" if source.get("optional") else ""
        if not target.exists():
            state, detail = ("skipped" if source.get("optional") else "MISSING"), ""
            if not source.get("optional"):
                missing.append(source)
        else:
            size_mb = target.stat().st_size / 1e6
            ratio = size_mb / source["approx_mb"] if source["approx_mb"] else 1
            state = "ok" if 0.7 < ratio < 1.4 else "SUSPECT SIZE"
            detail = f"{size_mb:8.1f} MB (expected ~{source['approx_mb']:.1f})"
        print(f"  {state:<13} {source['filename'] + tag:<58} {detail}")

    if missing:
        print("\nStill needed -- rerun without --check:")
        for source in missing:
            print(f"  {source['filename']}\n    {source['url']}")
        return 1
    print("\nAll required files present.")
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", default="data", type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--only", nargs="+", choices=sorted(BY_KEY),
                        help="fetch just these keys")
    parser.add_argument("--include-optional", action="store_true",
                        help="also fetch Chem2D_Jun2016.zip")
    args = parser.parse_args()

    dest = args.dest / "Raw" if (args.dest / "Raw").exists() else args.dest
    dest.mkdir(parents=True, exist_ok=True)

    if args.check:
        return check(dest)

    if args.only:
        wanted = [BY_KEY[key] for key in args.only]
    else:
        wanted = [s for s in SOURCES
                  if not s.get("optional") or args.include_optional]

    total = sum(s["approx_mb"] for s in wanted)
    print(f"Fetching {len(wanted)} files (~{total / 1000:.1f} GB) into "
          f"{dest.resolve()}\n")

    failed = [s["filename"] for s in wanted if not fetch(s, dest)]
    print()
    if failed:
        print(f"{len(failed)} failed: {', '.join(failed)}")
        return 1
    return check(dest)


if __name__ == "__main__":
    raise SystemExit(main())
