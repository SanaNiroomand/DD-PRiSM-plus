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
import hashlib
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path


class TransientResponse(Exception):
    """Server said 'not now' rather than failing outright."""

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
        "note": "NCI60 dose-response, VERSION 10 to match the paper (the current "
                "version 20 carries newer experiments, so row counts drift). "
                "23,636,946 rows, 2.44 GB uncompressed. Compressed with "
                "Deflate64, which the standard library cannot decompress: "
                "pip install zipfile-deflate64, then `import zipfile_deflate64` "
                "before opening it. Read from the zip rather than extracting.",
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
        "md5": "9402aa25a19279bb20a5d6cf8791a88f",
        "note": "DepMap Public 23Q4, log2(TPM+1) for protein-coding genes.",
    },
    {
        "key": "depmap_samples",
        "filename": "sample_info_18q3.csv",
        "url": "https://ndownloader.figshare.com/files/12704612",
        "approx_mb": 0.06,
        "md5": "bc72d3be18f1f9b69e8ca05422f26bdb",
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


def fetch(source, dest, attempts=4):
    """Stream to disk, resuming on failure, and refuse to accept a short file.

    A dropped connection looks exactly like a clean end-of-stream to
    ``response.read()``, so the byte count is checked against Content-Length.
    Without that a truncated download is silently written out as if complete --
    which produces a corrupt archive that only fails much later.
    """
    target = dest / source["filename"]
    if target.exists() and target.stat().st_size > 1024:
        print(f"  [skip]     {source['filename']} ({target.stat().st_size / 1e6:.1f} MB)")
        return True

    partial = target.with_suffix(target.suffix + ".part")
    print(f"  [get]      {source['filename']} (~{source['approx_mb']:.1f} MB)", flush=True)

    for attempt in range(1, attempts + 1):
        have = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": BROWSER_AGENT}
        if have:
            headers["Range"] = f"bytes={have}-"
            print(f"             resuming at {have / 1e6:.0f} MB "
                  f"(attempt {attempt}/{attempts})", flush=True)

        try:
            request = urllib.request.Request(source["url"], headers=headers)
            with urllib.request.urlopen(request, timeout=180) as response:
                # figshare answers 202 with an empty body while it prepares a
                # large file or throttles you. It is a 2xx, so urlopen does not
                # raise, and read() returns b"" -- indistinguishable from a
                # finished download unless the status is checked.
                if response.status == 202:
                    raise TransientResponse("202 Accepted (server busy)")
                if have and response.status != 206:
                    # Server ignored the Range header; start over cleanly.
                    partial.unlink(missing_ok=True)
                    have = 0

                declared = response.headers.get("Content-Length")
                expected = have + int(declared) if declared else None

                first = response.read(512)
                if not have and first.lstrip().lower().startswith((b"<!doctype", b"<html")):
                    print("             FAILED: got an HTML page, not the file")
                    return False

                written = have
                with partial.open("ab" if have else "wb") as handle:
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
            print(f"             interrupted: {error}")
            if attempt < attempts:
                delay = min(300, 5 * 2 ** (attempt - 1))
                print(f"             waiting {delay}s before retry", flush=True)
                time.sleep(delay)
            continue

        # Three ways a transfer can end early and still look clean.
        # figshare redirects to an S3 link signed for ten seconds; miss that
        # window and the body comes back empty with no Content-Length, which
        # read() reports as a perfectly ordinary end of stream.
        floor = 0.5 * source["approx_mb"] * 1e6
        if written == 0:
            problem = "empty response"
        elif expected is not None and written < expected:
            problem = f"short: {written / 1e6:.1f} of {expected / 1e6:.1f} MB"
        elif expected is None and written < floor:
            problem = (f"suspiciously small: {written / 1e6:.1f} MB vs "
                       f"~{source['approx_mb']:.1f} MB expected")
        else:
            problem = None

        if problem:
            print(f"             {problem} -- retrying")
            partial.unlink(missing_ok=True)   # signed URLs cannot be resumed
            continue

        partial.replace(target)
        print(f"             saved {written / 1e6:.1f} MB")
        return True

    print(f"             FAILED after {attempts} attempts; "
          f"partial kept at {partial.name} for resume")
    return False


def _md5(path, chunk=1 << 20):
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _zip_ok(path):
    """Does the archive's central directory parse?

    Deliberately does not call ``testzip()``. That decompresses every member,
    which costs minutes on a 2.4 GB payload and, worse, raises
    NotImplementedError on DOSERESP because it is Deflate64 -- a method the
    standard library can read the directory of but not decompress. A perfectly
    good download would be reported corrupt.
    """
    try:
        with zipfile.ZipFile(path) as archive:
            return bool(archive.namelist())
    except Exception:
        return False


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
            detail = f"{size_mb:8.1f} MB (expected ~{source['approx_mb']:.1f})"
            if not 0.95 < ratio < 1.05:
                state = "SUSPECT SIZE"
                missing.append(source)
            elif target.suffix == ".zip" and not _zip_ok(target):
                state, detail = "CORRUPT ZIP", detail + "  -- will not open"
                missing.append(source)
            elif source.get("md5") and _md5(target) != source["md5"]:
                state, detail = "BAD CHECKSUM", detail + "  -- md5 mismatch"
                missing.append(source)
            else:
                state = "ok"
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
    parser.add_argument("--attempts", type=int, default=5,
                        help="retries per file; raise it when figshare is "
                             "rate-limiting with 202s (backoff caps at 5 min)")
    args = parser.parse_args()

    # Files land directly in --dest. An earlier version silently appended
    # "Raw" when that folder happened to exist, so the same command wrote to
    # different places on different machines.
    dest = args.dest
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

    # When a host is throttling, every file on it will fail the same way.
    # Retrying each one separately burned 30 of 32 minutes in a Kaggle session,
    # so give up on the host after the first file exhausts its attempts.
    failed, throttled = [], set()
    for source in wanted:
        host = urllib.parse.urlparse(source["url"]).netloc
        if host in throttled:
            print(f"  [defer]    {source['filename']} -- {host} is throttling; "
                  f"rerun this file later")
            failed.append(source["filename"])
            continue
        if not fetch(source, dest, args.attempts):
            failed.append(source["filename"])
            throttled.add(host)
    print()
    if failed:
        print(f"{len(failed)} failed: {', '.join(failed)}")
        return 1
    return check(dest)


if __name__ == "__main__":
    raise SystemExit(main())
