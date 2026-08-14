"""Turn the raw downloads into training-ready tables.

Follows 01_Preprocessing.ipynb step for step, with two deliberate departures:

  * DOSERESP is read in chunks with six columns and tight dtypes. The notebook
    reads all 18 columns at default dtypes, which costs 11.1 GB for 23,636,946
    rows before pandas' parsing overhead. Filtering inside the loop keeps the
    peak in the hundreds of MB.
  * Intermediates are Parquet, not CSV -- smaller, faster, and dtypes survive.

Targets from the supplement, which are how you know it worked:
  Table S2  NCI60 training rows        7,915,900
  Table S4  combination training rows  1,387,317

    python scripts/preprocess.py --data data --out processed
    python scripts/preprocess.py --data data --out processed --stage nci60
"""

import argparse
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_sets import ALMANAC_CELL_LINES, ALMANAC_DRUGS  # noqa: E402

try:                       # patches zipfile to handle DOSERESP's Deflate64
    import zipfile_deflate64  # noqa: F401
except ImportError:
    pass

# NCI60 cell-line names that no annotation lookup resolves. Taken verbatim from
# the authors' notebook, which sourced them from Cellosaurus by hand.
MANUAL_CELL_LINES = {
    'CAKI-1': 'CAKI1', 'RXF 393': 'RXF393', '786-0': '786O', 'A549/ATCC': 'A549',
    'SF-268': 'SF268', 'HCT-116': 'HCT116', 'OVCAR-5': 'OVCAR5', 'UO-31': 'UO31',
    'HOP-62': 'HOP62', 'MALME-3M': 'MALME3M', 'UACC-257': 'UACC257',
    'SF-539': 'SF539', 'TK-10': 'TK10', 'NCI-H322M': 'NCIH322M',
    'MDA-MB-231/ATCC': 'MDAMB231', 'HCC-2998': 'HCC2998', 'RPMI-8226': 'RPMI8226',
    'SNB-75': 'SNB75', 'HS 578T': 'HS578T', 'U251': 'U251MG', 'SW-620': 'SW620',
    'SK-MEL-2': 'SKMEL2', '769-P': '769P', 'SW-156': 'SW156', 'SW-1573': 'SW1573',
    'SW 1088': 'SW1088', 'RPMI-7951': 'RPMI7951', 'SF-767': 'SF767',
    'MCF7/ATCC': 'MCF7', 'CALU-1': 'CALU1', 'CACO-2': 'CACO2',
}

DOSERESP_COLUMNS = ['NSC', 'CONCENTRATION', 'CELL_NAME',
                    'AVERAGE_GIPRCNT', 'CONCENTRATION_UNIT']
DOSERESP_DTYPES = {'NSC': 'int32', 'CONCENTRATION': 'float32',
                   'AVERAGE_GIPRCNT': 'float32',
                   'CELL_NAME': 'category', 'CONCENTRATION_UNIT': 'category'}


def banner(text):
    print(f"\n{'=' * 66}\n{text}\n{'=' * 66}", flush=True)


def report(label, value, expected=None):
    line = f"  {label:<34} {value:>12,}"
    if expected is not None:
        delta = value - expected
        mark = "MATCH" if delta == 0 else f"off by {delta:+,}"
        line += f"   (paper: {expected:>11,}  {mark})"
    print(line, flush=True)


# --------------------------------------------------------------------------
# stage: fingerprints
# --------------------------------------------------------------------------

def stage_fingerprints(data, out, source="chem2d"):
    """512-bit Morgan fingerprints, radius 2.

    Which compound file you start from decides the size of the whole dataset,
    so this defaults to the paper's route.

      chem2d      Chem2D_Jun2016.sdf, what the authors parsed. Covers 52,076 of
                  the 57,041 NSCs tested in NCI60.
      nsc_smiles  nsc_smiles.csv, simpler but *more* complete -- it covers all
                  57,041, which inflates every downstream count by about 11%
                  and puts the drug total at ~56,160 against the paper's 50,893.
    """
    banner(f"Morgan fingerprints (source: {source})")
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")

    bits, kept, seen = [], [], 0

    if source == "chem2d":
        archive = zipfile.ZipFile(data / "Chem2D_Jun2016.zip")
        with archive.open(archive.namelist()[0]) as handle:
            for mol in Chem.ForwardSDMolSupplier(handle):
                seen += 1
                if mol is None:
                    continue
                try:
                    nsc = int(mol.GetProp("_Name"))
                except (KeyError, ValueError):
                    continue
                bits.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=512))
                kept.append(nsc)
    else:
        smiles = pd.read_csv(data / "nsc_smiles.csv")
        seen = len(smiles)
        for nsc, smi in zip(smiles.NSC.values, smiles.SMILES.values):
            mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
            if mol is None:
                continue
            bits.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=512))
            kept.append(int(nsc))

    report("compound records", seen)
    frame = pd.DataFrame(np.array(bits, dtype=np.uint8), index=pd.Index(kept, name="NSC"))
    frame = frame[~frame.index.duplicated(keep="first")]
    frame.columns = [str(c) for c in frame.columns]
    report("fingerprints produced", len(frame))
    report("unusable, dropped", seen - len(frame))

    out.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out / "fingerprints.parquet")
    print(f"  -> {out / 'fingerprints.parquet'}")
    return frame


# --------------------------------------------------------------------------
# stage: expression
# --------------------------------------------------------------------------

def stage_expression(data, out):
    banner("Cell-line expression: z-score per cell line")
    frame = pd.read_csv(data / "OmicsExpressionProteinCodingGenesTPMLogp1.csv",
                        index_col=0)
    # Column headers look like "TSPAN6 (7105)"; the pathway sets use bare symbols.
    frame.columns = [gene.split(" (")[0] for gene in frame.columns]
    report("cell lines", frame.shape[0])
    report("genes", frame.shape[1])

    # z-score across genes within each cell line, so every row sums to zero.
    values = frame.to_numpy(dtype=np.float32)
    centred = values - values.mean(axis=1, keepdims=True)
    scaled = centred / values.std(axis=1, ddof=0, keepdims=True)
    zscored = pd.DataFrame(scaled, index=frame.index, columns=frame.columns)

    out.mkdir(parents=True, exist_ok=True)
    zscored.to_parquet(out / "expression_zscore.parquet")
    print(f"  -> {out / 'expression_zscore.parquet'}")
    return zscored


# --------------------------------------------------------------------------
# stage: nci60
# --------------------------------------------------------------------------

def normalise(name):
    """Fold a cell-line name to letters and digits: 'HS 578T' -> 'HS578T'."""
    return re.sub(r"[^A-Z0-9]", "", str(name).upper())


def map_cell_lines(names, model):
    """NCI60 CELL_NAME -> DepMap ModelID, using the 23Q4 Model table.

    The authors used DepMap-2018q3-celllines.csv, the full 18Q3 cell-line list.
    Model.csv is its modern replacement and a better fit here: its ModelID is
    exactly what indexes the expression matrix, so the join needs no second hop.

    Note the 18Q3 Achilles ``sample_info.csv`` is *not* a substitute -- it lists
    only the 485 lines that were CRISPR-screened and is missing most of NCI60
    (ACHN, A498, CCRF-CEM, COLO 205 among them), which caps the match at 31 of
    the 66 the paper reports.
    """
    lookup = {}
    for column in ("StrippedCellLineName", "CellLineName"):
        if column not in model.columns:
            continue
        for model_id, value in zip(model.ModelID, model[column]):
            if isinstance(value, str) and value:
                lookup.setdefault(normalise(value), model_id)

    mapping, unresolved = {}, []
    for name in names:
        model_id = lookup.get(normalise(name))
        if model_id is None:
            # A handful of NCI60 names disagree with DepMap's spelling; the
            # authors resolved these by hand against Cellosaurus.
            alias = MANUAL_CELL_LINES.get(name)
            if alias:
                model_id = lookup.get(normalise(alias))
        if model_id:
            mapping[name] = model_id
        else:
            unresolved.append(name)
    return mapping, unresolved


def stage_nci60(data, out, chunksize=2_000_000):
    banner("NCI60 dose-response")
    fingerprints = pd.read_parquet(out / "fingerprints.parquet")
    expression = pd.read_parquet(out / "expression_zscore.parquet")
    valid_nsc = set(fingerprints.index.astype(int))

    model = pd.read_csv(data / "Model.csv")

    archive = zipfile.ZipFile(data / "DOSERESP.zip")
    member = archive.namelist()[0]

    kept, total = [], 0
    with archive.open(member) as handle:
        for chunk in pd.read_csv(handle, usecols=DOSERESP_COLUMNS,
                                 dtype=DOSERESP_DTYPES, chunksize=chunksize):
            total += len(chunk)
            chunk = chunk[chunk.CONCENTRATION_UNIT == "M"]
            chunk = chunk[chunk.NSC.isin(valid_nsc)]
            kept.append(chunk[["NSC", "CONCENTRATION", "CELL_NAME",
                               "AVERAGE_GIPRCNT"]])
    frame = pd.concat(kept, ignore_index=True)
    del kept
    frame["CELL_NAME"] = frame.CELL_NAME.astype(str)

    report("rows in DOSERESP", total, 23_636_946)
    report("after unit + fingerprint filter", len(frame))

    # log10(M) -> log10(uM), and -100..100 growth percent -> 0..1 viability.
    frame["CONCENTRATION"] = frame.CONCENTRATION + 6
    frame["VIABILITY"] = (frame.AVERAGE_GIPRCNT + 100) / 200
    frame = frame.drop(columns=["AVERAGE_GIPRCNT"])

    mapping, unresolved = map_cell_lines(sorted(frame.CELL_NAME.unique()), model)
    report("cell lines in NCI60", frame.CELL_NAME.nunique())
    report("mapped to a DepMap ModelID", len(mapping))
    report("unmapped, dropped", len(unresolved))

    frame = frame[frame.CELL_NAME.isin(mapping)]
    frame["depmap_id"] = frame.CELL_NAME.map(mapping)

    common = set(expression.index) & set(frame.depmap_id)
    frame = frame[frame.depmap_id.isin(common)]
    report("cell lines with expression", len(common), 66)
    report("rows after cell-line filter", len(frame))

    frame = frame.rename(columns={"CELL_NAME": "CELLNAME"})
    frame = frame[frame.VIABILITY < 1.5]
    report("rows after viability < 1.5", len(frame))

    # One representative per (cell line, drug, dose): median over replicates.
    names = frame[["depmap_id", "CELLNAME"]].drop_duplicates("depmap_id")
    frame = (frame.groupby(["depmap_id", "NSC", "CONCENTRATION"], observed=True)
                  .VIABILITY.median().reset_index())
    frame = frame.merge(names, on="depmap_id", how="left")
    report("after median over replicates", len(frame))

    frame = filter_dilution_and_variance(frame)
    report("NCI60 filtered total", len(frame))

    out.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out / "nci60_filtered.parquet")
    print(f"  -> {out / 'nci60_filtered.parquet'}")
    return frame


def filter_dilution_and_variance(frame):
    """Drop pairs not on a 10-fold dilution series, then flat pairs.

    A dilution step other than one log unit means several experiments with
    different batch effects were pooled. Zero variance across concentrations
    means the pair carries no dose information at all.
    """
    frame = frame.sort_values(["CELLNAME", "NSC", "CONCENTRATION"]).reset_index(drop=True)
    frame["CONCENTRATION"] = frame.CONCENTRATION.round(5)

    same_pair = ((frame.NSC.shift(1) == frame.NSC) &
                 (frame.CELLNAME.shift(1) == frame.CELLNAME))
    delta = (frame.CONCENTRATION.shift(1) - frame.CONCENTRATION).round(5)
    offenders = frame[same_pair & (delta != -1.0)][["CELLNAME", "NSC"]].drop_duplicates()
    report("pairs off the 10-fold series", len(offenders))

    keys = pd.MultiIndex.from_frame(frame[["CELLNAME", "NSC"]])
    frame = frame[~keys.isin(pd.MultiIndex.from_frame(offenders))]

    spread = frame.groupby(["CELLNAME", "NSC"], observed=True).VIABILITY.std()
    flat = spread[spread == 0].index
    report("pairs with zero variance", len(flat))
    keys = pd.MultiIndex.from_frame(frame[["CELLNAME", "NSC"]])
    return frame[~keys.isin(flat)]


# --------------------------------------------------------------------------
# stage: splits
# --------------------------------------------------------------------------

def stage_splits(out, seed=0):
    banner("NCI60 splits")
    frame = pd.read_parquet(out / "nci60_filtered.parquet")

    # 90/10 overall; the 10% test half is split between unseen cell lines and
    # unseen drugs, so each stratum takes 1-sqrt(0.95) of its entity list.
    factor_test = 1 - (1 - 0.05) ** 0.5

    def every_nth(counts):
        stride = int(len(counts) / (factor_test * len(counts)))
        held_out = np.arange(0, len(counts), stride)
        return set(counts.iloc[held_out].index)

    cell_counts = frame.groupby("CELLNAME", observed=True).VIABILITY.count().sort_values(ascending=False)
    drug_counts = frame.groupby("NSC", observed=True).VIABILITY.count().sort_values(ascending=False)
    unseen_cells = every_nth(cell_counts)
    unseen_drugs = every_nth(drug_counts)
    report("unseen cell lines", len(unseen_cells))
    report("unseen drugs", len(unseen_drugs))

    cell_out = frame.CELLNAME.isin(unseen_cells)
    drug_out = frame.NSC.isin(unseen_drugs)

    splits = {
        "unseen_all": frame[cell_out & drug_out],
        "unseen_drug": frame[drug_out & ~cell_out],
        "unseen_cellline": frame[cell_out & ~drug_out],
    }
    seen = frame[~cell_out & ~drug_out]

    pairs = seen[["CELLNAME", "NSC"]].drop_duplicates()
    holdout = pairs.sample(frac=1 / 18, random_state=seed)
    is_holdout = pd.MultiIndex.from_frame(seen[["CELLNAME", "NSC"]]).isin(
        pd.MultiIndex.from_frame(holdout))
    splits["unseen_pair"] = seen[is_holdout]
    splits["trainval"] = seen[~is_holdout]

    print()
    report("training + validation", len(splits["trainval"]), 7_915_900 + 989_488)
    for name in ("unseen_pair", "unseen_cellline", "unseen_drug", "unseen_all"):
        report(name, len(splits[name]))

    directory = out / "nci60_splits"
    directory.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        part.to_parquet(directory / f"{name}.parquet")
    print(f"  -> {directory}")
    return splits


# --------------------------------------------------------------------------
# stage: almanac
# --------------------------------------------------------------------------

def stage_almanac(data, out):
    banner("NCI-ALMANAC")
    # Supplementary Data 1 names the exact 102 drugs and 44 cell lines. ALMANAC
    # screened 61 cell lines, so deriving the set from what NCI60 kept pulls in
    # 17 the paper excluded. Using the published lists reproduces all three
    # counts exactly.
    valid_nsc = set(ALMANAC_DRUGS)
    valid_cells = set(ALMANAC_CELL_LINES)

    archive = zipfile.ZipFile(data / "ComboDrugGrowth_Nov2017.zip")
    with archive.open(archive.namelist()[0]) as handle:
        frame = pd.read_csv(handle, usecols=["NSC1", "CONC1", "NSC2", "CONC2",
                                             "CELLNAME", "PERCENTGROWTH"],
                            low_memory=False)
    report("ALMANAC rows", len(frame))
    report("paper drugs / cell lines", len(valid_nsc) + len(valid_cells), 146)

    frame["VIABILITY"] = (frame.PERCENTGROWTH + 100) / 200
    frame = frame[frame.VIABILITY < 1.5]
    frame = frame[frame.CELLNAME.isin(valid_cells)]

    mono = frame[frame.NSC2.isna()].copy()
    mono["CONCENTRATION"] = np.log10(mono.CONC1) + 6
    mono = mono[mono.NSC1.isin(valid_nsc)]
    mono = (mono.groupby(["NSC1", "CONCENTRATION", "CELLNAME"], observed=True)
                .VIABILITY.median().reset_index()
                .rename(columns={"NSC1": "NSC"}))
    report("monotherapy rows", len(mono), 35_041)

    combo = frame[frame.NSC2.notna()].copy()
    combo["CONCENTRATION1"] = np.log10(combo.CONC1) + 6
    combo["CONCENTRATION2"] = np.log10(combo.CONC2) + 6
    combo["NSC2"] = combo.NSC2.astype("int64")
    combo = combo[combo.NSC1.isin(valid_nsc) & combo.NSC2.isin(valid_nsc)]
    combo = (combo.groupby(["NSC1", "NSC2", "CONCENTRATION1", "CONCENTRATION2",
                            "CELLNAME"], observed=True)
                  .VIABILITY.median().reset_index())
    report("combination rows", len(combo), 1_981_135)
    report("distinct drug pairs", len(combo[["NSC1", "NSC2"]].drop_duplicates()), 5_032)

    out.mkdir(parents=True, exist_ok=True)
    mono.to_parquet(out / "almanac_mono.parquet")
    combo.to_parquet(out / "almanac_combo.parquet")
    print(f"  -> {out / 'almanac_mono.parquet'}, {out / 'almanac_combo.parquet'}")
    return mono, combo


# --------------------------------------------------------------------------
# stage: almanac splits
# --------------------------------------------------------------------------

def held_out_entities(counts, factor_test=None):
    """Every Nth entity by descending row count, as the authors' notebook does."""
    if factor_test is None:
        factor_test = 1 - (1 - 0.05) ** 0.5
    stride = int(len(counts) / (factor_test * len(counts)))
    return set(counts.iloc[np.arange(0, len(counts), stride)].index)


def stage_almanac_splits(out, seed=0):
    """Split ALMANAC the way Tables S3 and S4 describe.

    Without this the fine-tune and combination stages train on everything and
    are scored on a random slice of it, so their numbers cannot be compared with
    the paper's -- those are measured on entities held out entirely.

    Combination rows are labelled by how much of them is unseen: the cell line,
    one drug, both drugs, or a cell line together with a drug.
    """
    banner("NCI-ALMANAC splits")

    mono = pd.read_parquet(out / "almanac_mono.parquet")
    combo = pd.read_parquet(out / "almanac_combo.parquet")

    cells = combo.groupby("CELLNAME", observed=True).VIABILITY.count().sort_values(ascending=False)
    drugs = pd.concat([combo.NSC1, combo.NSC2]).value_counts()
    unseen_cells = held_out_entities(cells)
    unseen_drugs = held_out_entities(drugs)
    report("unseen cell lines", len(unseen_cells), 2)
    report("unseen drugs", len(unseen_drugs), 3)

    # --- monotherapy (Table S3 shape) -------------------------------------
    cell_out = mono.CELLNAME.isin(unseen_cells)
    drug_out = mono.NSC.isin(unseen_drugs)
    mono_splits = {
        "unseen_all": mono[cell_out & drug_out],
        "unseen_drug": mono[drug_out & ~cell_out],
        "unseen_cellline": mono[cell_out & ~drug_out],
    }
    seen = mono[~cell_out & ~drug_out]
    pairs = seen[["CELLNAME", "NSC"]].drop_duplicates()
    holdout = pairs.sample(frac=1 / 18, random_state=seed)
    is_holdout = pd.MultiIndex.from_frame(seen[["CELLNAME", "NSC"]]).isin(
        pd.MultiIndex.from_frame(holdout))
    mono_splits["unseen_pair"] = seen[is_holdout]
    mono_splits["trainval"] = seen[~is_holdout]

    print()
    print("  monotherapy")
    for name, part in mono_splits.items():
        report("    " + name, len(part))

    # --- combination (Table S4 shape) -------------------------------------
    cell_out = combo.CELLNAME.isin(unseen_cells)
    drugs_out = combo.NSC1.isin(unseen_drugs).astype(int) + combo.NSC2.isin(unseen_drugs).astype(int)

    combo_splits = {
        "unseen_all": combo[cell_out & (drugs_out > 0)],
        "unseen_two_drug": combo[~cell_out & (drugs_out == 2)],
        "unseen_one_drug": combo[~cell_out & (drugs_out == 1)],
        "unseen_cellline": combo[cell_out & (drugs_out == 0)],
    }
    seen = combo[~cell_out & (drugs_out == 0)]
    keys = ["CELLNAME", "NSC1", "NSC2"]
    triples = seen[keys].drop_duplicates()
    holdout = triples.sample(frac=1 / 9, random_state=seed)
    is_holdout = pd.MultiIndex.from_frame(seen[keys]).isin(
        pd.MultiIndex.from_frame(holdout))
    combo_splits["unseen_pair"] = seen[is_holdout]
    combo_splits["trainval"] = seen[~is_holdout]

    expected = {"trainval": 1_387_317 + 198_189, "unseen_pair": 197_934,
                "unseen_cellline": 84_865, "unseen_one_drug": 111_732,
                "unseen_two_drug": 1_044, "unseen_all": 54}
    print()
    print("  combination")
    for name, part in combo_splits.items():
        report("    " + name, len(part), expected.get(name))
    report("    total", sum(len(p) for p in combo_splits.values()), 1_981_135)

    for label, splits in (("almanac_mono_splits", mono_splits),
                          ("almanac_combo_splits", combo_splits)):
        directory = out / label
        directory.mkdir(parents=True, exist_ok=True)
        for name, part in splits.items():
            part.to_parquet(directory / f"{name}.parquet")
        print(f"  -> {directory}")
    return mono_splits, combo_splits


STAGES = ["fingerprints", "expression", "nci60", "splits", "almanac",
          "almanac_splits"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("processed"))
    parser.add_argument("--stage", choices=STAGES + ["all"], default="all")
    parser.add_argument("--chunksize", type=int, default=2_000_000)
    parser.add_argument("--fingerprint-source", choices=["chem2d", "nsc_smiles"],
                        default="chem2d",
                        help="chem2d matches the paper; nsc_smiles is more "
                             "complete and inflates every count by ~11%")
    args = parser.parse_args()

    wanted = STAGES if args.stage == "all" else [args.stage]
    args.out.mkdir(parents=True, exist_ok=True)

    if "fingerprints" in wanted:
        stage_fingerprints(args.data, args.out, args.fingerprint_source)
    if "expression" in wanted:
        stage_expression(args.data, args.out)
    if "nci60" in wanted:
        stage_nci60(args.data, args.out, args.chunksize)
    if "splits" in wanted:
        stage_splits(args.out)
    if "almanac" in wanted:
        stage_almanac(args.data, args.out)
    if "almanac_splits" in wanted:
        stage_almanac_splits(args.out)

    banner("done")
    for item in sorted(args.out.rglob("*.parquet")):
        print(f"  {item.relative_to(args.out)}  "
              f"{item.stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
