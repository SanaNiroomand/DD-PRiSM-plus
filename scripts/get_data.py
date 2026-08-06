"""Fetch and verify the DD-PRiSM source datasets.

Every URL here was checked on 2026-08-06. Three NCI files cannot be scripted:
the NCI wiki serves them fine to a browser but returns 403 to any programmatic
request, so they are marked browser_only and must be downloaded by hand.

    python scripts/get_data.py --dest data          # fetch what can be fetched
    python scripts/get_data.py --dest data --check  # verify everything present

Version pinning matters. The paper used DOSERESP version 10 (January 2024).
The current version 20 (July 2026 release) reports one row per experiment
(EXPID) instead of aggregating across experiments, so row counts will not match
the supplement's Table S2. Use version 10 to keep the published checkpoints
meaningful.
"""

import argparse
import sys
import urllib.request
from pathlib import Path

BROWSER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"

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
        "key": "nci60_doseresp",
        "filename": "DOSERESP.zip",
        "url": ("https://wiki.nci.nih.gov/download/attachments/147193864/"
                "DOSERESP.zip?version=10&modificationDate=1704733010000&api=v2"),
        "approx_mb": 333.4,
        "browser_only": True,
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
        "browser_only": True,
        "note": "NCI-ALMANAC combination responses. Never revised.",
    },
    {
        "key": "nsc_smiles",
        "filename": "nsc_smiles.csv",
        "url": ("https://wiki.nci.nih.gov/download/attachments/155844992/"
                "nsc_smiles.csv?version=11&modificationDate=1783565838764&api=v2"),
        "approx_mb": 17.3,
        "browser_only": True,
        "note": "NSC to SMILES directly. Simpler than the paper's route of "
                "parsing Chem2D_Jun2016.zip (80.6 MB) with RDKit. Use "
                "Chem2D_Jun2016.zip instead if you want to match the paper exactly.",
    },
]


def fetch(source, dest):
    target = dest / source["filename"]
    if target.exists() and target.stat().st_size > 1024:
        print(f"  [skip]     {source['filename']} already present "
              f"({target.stat().st_size / 1e6:.1f} MB)")
        return True

    print(f"  [download] {source['filename']} (~{source['approx_mb']:.1f} MB)")
    request = urllib.request.Request(source["url"],
                                     headers={"User-Agent": BROWSER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = response.read()
    except Exception as error:
        print(f"             FAILED: {error}")
        return False

    # An HTML body means a login wall or a bot check, not the file.
    if payload[:200].lstrip().lower().startswith((b"<!doctype", b"<html")):
        print("             FAILED: got an HTML page, not the file")
        return False

    target.write_bytes(payload)
    print(f"             saved {len(payload) / 1e6:.1f} MB")
    return True


def check(dest):
    print(f"Checking {dest.resolve()}\n")
    missing = []
    for source in SOURCES:
        target = dest / source["filename"]
        if not target.exists():
            state, detail = "MISSING", ""
            missing.append(source)
        else:
            size_mb = target.stat().st_size / 1e6
            ratio = size_mb / source["approx_mb"] if source["approx_mb"] else 1
            state = "ok" if 0.5 < ratio < 2.0 else "SUSPECT SIZE"
            detail = f"{size_mb:8.1f} MB (expected ~{source['approx_mb']:.1f})"
        print(f"  {state:<13} {source['filename']:<48} {detail}")

    if missing:
        print("\nStill needed:")
        for source in missing:
            how = "download in your BROWSER" if source.get("browser_only") else "rerun this script"
            print(f"\n  {source['filename']}  -- {how}")
            print(f"    {source['url']}")
            print(f"    {source['note']}")
    else:
        print("\nAll present.")
    return 0 if not missing else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", default="data", type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--all", action="store_true",
                        help="also attempt the browser-only files (expect 403)")
    args = parser.parse_args()

    dest = args.dest / "Raw" if (args.dest / "Raw").exists() else args.dest
    dest.mkdir(parents=True, exist_ok=True)

    if args.check:
        return check(dest)

    print(f"Fetching into {dest.resolve()}\n")
    blocked = []
    for source in SOURCES:
        if source.get("browser_only") and not args.all:
            blocked.append(source)
            continue
        fetch(source, dest)

    if blocked:
        print("\n" + "=" * 70)
        print("DOWNLOAD THESE THREE BY HAND -- the NCI wiki blocks scripts (403)")
        print("=" * 70)
        for source in blocked:
            print(f"\n{source['filename']}  (~{source['approx_mb']:.1f} MB)")
            print(f"  {source['url']}")
            print(f"  {source['note']}")
        print(f"\nSave them into {dest.resolve()}, then run with --check.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
