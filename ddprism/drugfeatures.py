"""Drug representations for the Monotherapy model.

The published model sees each drug as a 512-bit Morgan fingerprint and nothing
else. That is the paper's own stated limitation -- "we need more informative
drug features for the phenotypic prediction" -- and our held-out numbers show
exactly where it bites: the unseen-drug split scores RMSE 0.1604 / PCC 0.7585
against 0.0828 / 0.9386 on unseen pairs. A fingerprint says which substructures
are present; it says nothing about what the molecule *does*.

This module assembles the drug feature matrix from one or more sources, so the
same training code runs on Morgan alone (the paper), on a pretrained chemical
language model embedding, or on the two concatenated:

    ids, matrix, report = load_drug_features(processed, ["morgan", "chemberta"])

Every source is a parquet indexed by NSC. Nothing downstream of the first
Linear in each drug branch knows the difference.

Two details that decide whether fusion works at all:

  * **Scale.** Morgan bits are 0/1 and about 5% dense; transformer embeddings
    are dense floats with per-dimension scales spanning orders of magnitude.
    Concatenated raw, the first Linear is dominated by whichever block happens
    to be larger. Embedding blocks are therefore standardised per dimension
    across the drug library. This uses no labels, so it is not leakage.
  * **Coverage.** Sources are intersected on NSC, and the intersection is
    reported. A source missing a drug silently shrinks the dataset, which looks
    like a modelling result and is not one.
"""

import numpy as np
import pandas as pd

# name -> file under the processed directory
FEATURE_SOURCES = {
    "morgan": "fingerprints.parquet",
    "chemberta": "drug_embeddings_chemberta.parquet",
    "molformer": "drug_embeddings_molformer.parquet",
}

# Morgan is binary and stays uint8; everything else is a dense float embedding
# that gets standardised.
BINARY_SOURCES = {"morgan"}


def parse_spec(spec):
    """Accept 'morgan+chemberta', 'morgan,chemberta' or ['morgan', 'chemberta'].

    Order matters only for reproducibility -- it fixes the column layout, and
    therefore which checkpoint can be loaded into which model.
    """
    if isinstance(spec, str):
        names = [part for part in spec.replace(",", "+").split("+") if part]
    else:
        names = list(spec)

    unknown = [n for n in names if n not in FEATURE_SOURCES]
    if unknown:
        raise SystemExit(
            f"unknown drug feature source(s): {', '.join(unknown)}. "
            f"Known: {', '.join(sorted(FEATURE_SOURCES))}")
    if not names:
        raise SystemExit("--drug-features needs at least one source")
    if len(set(names)) != len(names):
        raise SystemExit(f"duplicate drug feature source in {names}")
    return names


def source_path(processed, name):
    return processed / FEATURE_SOURCES[name]


def missing_sources(processed, names):
    return [n for n in names if not source_path(processed, n).exists()]


def standardise(values):
    """Zero mean, unit variance per dimension, over the whole drug library.

    Dimensions that are constant across every drug carry no information; they
    are left at zero rather than divided by ~0.
    """
    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return ((values - mean) / std).astype(np.float32)


def load_drug_features(processed, spec, restrict=None, standardize=True):
    """Assemble the per-drug feature matrix.

    Args:
        processed: directory holding the preprocessing output.
        spec: source names, as a list or a '+'-joined string.
        restrict: optional iterable of NSCs to keep. The full Chem2D library is
            281,264 compounds; the study touches about 51,000 of them, and
            holding embeddings for the rest costs GPU memory for nothing.
        standardize: z-score each embedding source. Leave on unless you are
            deliberately testing the effect.

    Returns:
        (drug_ids, matrix, report) -- matrix is (drugs, width), uint8 when the
        only source is Morgan and float32 otherwise.
    """
    names = parse_spec(spec)
    missing = missing_sources(processed, names)
    if missing:
        raise SystemExit(
            f"missing drug feature file(s) for: {', '.join(missing)}\n"
            f"  expected {', '.join(str(source_path(processed, n)) for n in missing)}\n"
            f"  Build them with: python scripts/embed_drugs.py "
            f"--model {missing[0]} --data <raw> --out {processed}")

    blocks, report = [], {"sources": [], "standardized": standardize}
    index = None
    for name in names:
        frame = pd.read_parquet(source_path(processed, name))
        frame.index = frame.index.astype(np.int64)
        frame = frame[~frame.index.duplicated(keep="first")]
        report["sources"].append({"name": name, "drugs": len(frame),
                                  "width": frame.shape[1]})
        index = frame.index if index is None else index.intersection(frame.index)
        blocks.append((name, frame))

    if restrict is not None:
        wanted = pd.Index(sorted({int(d) for d in restrict}), dtype=np.int64)
        report["requested"] = len(wanted)
        report["unavailable"] = len(wanted.difference(index))
        index = index.intersection(wanted)

    index = index.sort_values()
    if len(index) == 0:
        raise SystemExit(
            "no drug is present in every requested feature source; the sources "
            "do not describe the same compound library")

    parts = []
    for name, frame in blocks:
        values = frame.loc[index].to_numpy()
        if name in BINARY_SOURCES:
            parts.append(values.astype(np.uint8))
        else:
            values = values.astype(np.float32)
            parts.append(standardise(values) if standardize else values)

    if all(name in BINARY_SOURCES for name, _ in blocks):
        matrix = np.concatenate(parts, axis=1).astype(np.uint8)
    else:
        matrix = np.concatenate([p.astype(np.float32) for p in parts], axis=1)

    report["drugs"] = len(index)
    report["width"] = matrix.shape[1]
    report["dtype"] = str(matrix.dtype)
    report["spec"] = "+".join(names)
    return index.to_numpy(), matrix, report


def describe(report):
    parts = [f"{s['name']}({s['width']})" for s in report["sources"]]
    line = (f"  drug features: {' + '.join(parts)} -> {report['width']} dims, "
            f"{report['drugs']:,} drugs, {report['dtype']}")
    if report.get("unavailable"):
        line += (f"\n  WARNING: {report['unavailable']:,} of "
                 f"{report['requested']:,} drugs used by the response tables "
                 f"have no feature vector and will fail on lookup")
    return line


# --------------------------------------------------------------------------
# which drugs are actually used
# --------------------------------------------------------------------------

# (relative path, columns holding NSCs). Split directories first, because those
# are what training and evaluation actually read; the unsplit tables are a
# fallback for a processed directory built before the splits existed.
_DRUG_TABLES = [
    ("nci60_splits", ("NSC", "NSC1")),
    ("almanac_mono_splits", ("NSC", "NSC1")),
    ("almanac_combo_splits", ("NSC1", "NSC2")),
]
_FALLBACK_TABLES = [
    ("nci60_filtered.parquet", ("NSC",)),
    ("almanac_mono.parquet", ("NSC", "NSC1")),
    ("almanac_combo.parquet", ("NSC1", "NSC2")),
]


def _read_ids(path, columns):
    import pyarrow.parquet as pq

    available = set(pq.read_schema(path).names)
    found = [c for c in columns if c in available]
    if not found:
        return set()
    frame = pd.read_parquet(path, columns=found)
    return {int(v) for c in found for v in frame[c].unique()}


def used_drug_ids(processed):
    """Every NSC any response table refers to, or None if none can be read.

    Reading one integer column out of each table is cheap next to holding
    feature vectors for a compound library five times larger than the study.
    """
    ids = set()
    for folder, columns in _DRUG_TABLES:
        directory = processed / folder
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.parquet")):
            ids |= _read_ids(path, columns)

    for filename, columns in _FALLBACK_TABLES:
        path = processed / filename
        if path.exists():
            ids |= _read_ids(path, columns)

    return ids or None
