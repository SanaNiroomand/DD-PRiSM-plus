"""Build an L1000 transcriptomic signature for every drug.

A Morgan fingerprint describes a molecule by its parts; a ChemBERTa embedding
describes it by its chemistry. An **L1000 signature** describes it by what it
*does to a cell* -- the change it induces across 978 landmark genes. That is a
different kind of information from anything the model currently sees, and the
unseen-drug split is exactly where "what does this molecule do" should help.

This writes one parquet indexed by NSC, aligned with fingerprints.parquet, ready
for ``--drug-features morgan+l1000``.

    python scripts/fetch_l1000.py --lincs lincs --data data --out processed

The standard source is LINCS Phase I (GSE92742), the three files below plus the
compound table, from https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE92742
(GEO FTP mirror under series/GSE92nnn/GSE92742/suppl/):

  GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx   (~23 GB)
  GSE92742_Broad_LINCS_sig_info.txt
  GSE92742_Broad_LINCS_gene_info.txt
  GSE92742_Broad_LINCS_pert_info.txt

The .gctx is large, but only the 978 landmark rows and the few thousand
signature columns belonging to drugs this study actually uses are ever read, so
memory stays small.

How a drug gets one vector:

  * **Which genes.** The 978 landmark genes (``pr_is_lm == 1``). The other
    ~11,350 are model-inferred from these, so they carry no independent signal
    and only add noise and width.
  * **Which signatures.** Level-5 MODZ signatures are already a moderated-z
    consensus over replicates, one per (compound, cell line, dose, time). A
    drug has many. We take the mean over all its ``trt_cp`` signatures as the
    per-compound consensus -- the simplest standard choice, and robust because
    it averages across cell lines and doses rather than betting on one.
  * **How it maps to NSC.** L1000 keys on its own compound ids; this study keys
    on NSC. The bridge is the InChIKey: canonicalise each NSC's SMILES to an
    InChIKey with RDKit, do the same for every L1000 compound, and match. A
    full-InChIKey match is exact; a first-block (skeleton) match ignores salt
    and stereochemistry and is reported separately so it is never mistaken for
    an exact one.

Coverage is reported, not hidden: a drug with no L1000 signature has no row, and
``load_drug_features`` intersects sources, so the ``l1000`` experiment runs on
the subset of drugs L1000 actually covers. That subset, not a silent shrink, is
the honest scope of the result.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ddprism.drugfeatures import used_drug_ids  # noqa: E402

MISSING = {"-666", "", "nan", "NA", "None", None}


def banner(text):
    print(f"\n{'=' * 66}\n{text}\n{'=' * 66}", flush=True)


# --------------------------------------------------------------------------
# pure helpers (no GCTX, no cmapPy) -- these are what the tests exercise
# --------------------------------------------------------------------------

def canonical_inchikey(smiles):
    """InChIKey of a SMILES, or None if RDKit cannot parse it.

    The InChIKey is a hash of the standard InChI, so two records for the same
    compound reached by different SMILES strings produce the same key. That is
    what lets an NSC and an L1000 compound be recognised as the same molecule.
    """
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    if not isinstance(smiles, str) or not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToInchiKey(mol) or None


def skeleton(inchikey):
    """First block of an InChIKey -- the connectivity, without stereo or salt."""
    return inchikey.split("-")[0] if inchikey else None


def nsc_inchikeys(data_dir, wanted=None):
    """{NSC: InChIKey} from nsc_smiles.csv, via RDKit canonicalisation."""
    frame = pd.read_csv(Path(data_dir) / "nsc_smiles.csv")
    out, unparsed = {}, 0
    for nsc, smi in zip(frame.NSC.values, frame.SMILES.values):
        try:
            nsc = int(nsc)
        except (TypeError, ValueError):
            continue
        if wanted is not None and nsc not in wanted:
            continue
        if nsc in out:
            continue
        key = canonical_inchikey(smi)
        if key is None:
            unparsed += 1
            continue
        out[nsc] = key
    print(f"  NSC compounds with an InChIKey : {len(out):,} "
          f"({unparsed:,} unparseable)")
    return out


def pert_inchikeys(pert_info):
    """{pert_id: InChIKey} for small-molecule perturbagens.

    LINCS records an ``inchi_key`` directly but marks missing values ``-666``;
    where it is absent the key is recomputed from ``canonical_smiles`` so a
    compound is not dropped for a blank metadata cell.
    """
    frame = pert_info[pert_info.get("pert_type", "trt_cp") == "trt_cp"] \
        if "pert_type" in pert_info.columns else pert_info
    out, recovered = {}, 0
    for _, row in frame.iterrows():
        pid = row["pert_id"]
        key = str(row.get("inchi_key", "")).strip()
        if key in MISSING or "-" not in key:
            key = canonical_inchikey(str(row.get("canonical_smiles", "")))
            if key is not None:
                recovered += 1
        if key and key not in MISSING:
            out[pid] = key
    print(f"  L1000 compounds with an InChIKey : {len(out):,} "
          f"({recovered:,} recomputed from SMILES)")
    return out


def match_nsc_to_pert(nsc_keys, pert_keys):
    """Map each NSC to the L1000 pert_ids for the same molecule.

    Returns (exact, skeleton_only): two dicts NSC -> [pert_id, ...]. A full
    InChIKey match is exact. If none exists, a first-block match (same
    connectivity, differing salt or stereochemistry) is offered separately so
    the caller can decide whether to trust it.
    """
    by_full, by_skel = {}, {}
    for pid, key in pert_keys.items():
        by_full.setdefault(key, []).append(pid)
        by_skel.setdefault(skeleton(key), []).append(pid)

    exact, skel = {}, {}
    for nsc, key in nsc_keys.items():
        if key in by_full:
            exact[nsc] = sorted(by_full[key])
        elif skeleton(key) in by_skel:
            skel[nsc] = sorted(by_skel[skeleton(key)])
    return exact, skel


def consensus_by_pert(landmark_df, sig_to_pert):
    """Per-compound consensus: mean landmark signature over a drug's signatures.

    Args:
        landmark_df: DataFrame of shape (genes, signatures), the 978 landmark
            rows of the Level-5 MODZ matrix, columns are sig_ids.
        sig_to_pert: {sig_id: pert_id} for the columns to use.

    Returns a DataFrame (pert_id, genes): each row the mean of that compound's
    signature columns, in the gene order of ``landmark_df.index``.
    """
    cols = [c for c in landmark_df.columns if c in sig_to_pert]
    sub = landmark_df[cols]
    perts = pd.Index([sig_to_pert[c] for c in cols], name="pert_id")
    # mean over all signatures sharing a pert_id
    grouped = sub.T.groupby(perts).mean()
    return grouped  # (pert_id, genes)


def expand_to_nsc(consensus, nsc_to_perts):
    """Turn a (pert_id, genes) table into an (NSC, genes) table.

    A drug mapped to several pert_ids (e.g. salts measured separately) takes the
    mean of their consensus vectors.
    """
    rows, index = [], []
    genes = consensus.columns
    for nsc, perts in nsc_to_perts.items():
        have = [p for p in perts if p in consensus.index]
        if not have:
            continue
        rows.append(consensus.loc[have].to_numpy().mean(axis=0))
        index.append(int(nsc))
    if not rows:
        return pd.DataFrame(columns=genes, index=pd.Index([], name="NSC"))
    frame = pd.DataFrame(np.vstack(rows), index=pd.Index(index, name="NSC"),
                         columns=genes)
    return frame.sort_index()


# --------------------------------------------------------------------------
# GCTX reading (needs cmapPy; imported lazily so the helpers above stay light)
# --------------------------------------------------------------------------

def landmark_genes(gene_info_path):
    """(ids, symbols) for the 978 landmark genes, sorted by gene id."""
    info = pd.read_csv(gene_info_path, sep="\t")
    lm = info[info["pr_is_lm"] == 1].sort_values("pr_gene_id")
    ids = [str(g) for g in lm["pr_gene_id"].tolist()]
    symbols = lm["pr_gene_symbol"].tolist()
    return ids, symbols


def read_landmark_matrix(gctx_path, gene_ids, sig_ids):
    """The landmark-by-signature submatrix, as a (genes, signatures) DataFrame.

    Only these rows and columns are pulled off disk, so the 23 GB file costs a
    few hundred MB of memory, not its full size.
    """
    from cmapPy.pandasGEXpress.parse import parse
    gct = parse(str(gctx_path), rid=gene_ids, cid=sig_ids)
    return gct.data_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lincs", type=Path, required=True,
                        help="directory with the GSE92742 Level5 gctx, "
                             "sig_info, gene_info and pert_info files")
    parser.add_argument("--data", type=Path, default=Path("data"),
                        help="raw directory, for nsc_smiles.csv")
    parser.add_argument("--out", type=Path, default=Path("processed"))
    parser.add_argument("--all-compounds", action="store_true",
                        help="map every NSC with SMILES, not just the drugs the "
                             "response tables use")
    parser.add_argument("--include-skeleton", action="store_true",
                        help="also accept first-block InChIKey matches (same "
                             "connectivity, differing salt/stereo)")
    parser.add_argument("--gctx", default=None,
                        help="explicit gctx filename if autodetection fails")
    args = parser.parse_args()

    banner("L1000 drug signatures (GSE92742 Level-5 MODZ)")

    def find(pattern, required=True):
        hits = sorted(args.lincs.glob(pattern))
        if not hits and required:
            raise SystemExit(f"missing {pattern} under {args.lincs}")
        return hits[0] if hits else None

    gctx = Path(args.gctx) if args.gctx else find("*Level5_COMPZ.MODZ*.gctx")
    gene_info = find("*gene_info*.txt")
    sig_info = find("*sig_info*.txt")
    pert_info = find("*pert_info*.txt")

    wanted = None if args.all_compounds else used_drug_ids(args.out)
    if wanted is not None:
        print(f"  drugs used by the response tables : {len(wanted):,}")

    nsc_keys = nsc_inchikeys(args.data, wanted)
    pert_df = pd.read_csv(pert_info, sep="\t", low_memory=False)
    pert_keys = pert_inchikeys(pert_df)

    exact, skel = match_nsc_to_pert(nsc_keys, pert_keys)
    mapping = dict(exact)
    print(f"  NSC matched to L1000 (exact)     : {len(exact):,}")
    if args.include_skeleton:
        for nsc, perts in skel.items():
            mapping.setdefault(nsc, perts)
        print(f"  NSC matched (skeleton, added)    : {len(skel):,}")
    else:
        print(f"  NSC skeleton-only (not used)     : {len(skel):,}")

    if not mapping:
        raise SystemExit("no NSC matched an L1000 compound by InChIKey")

    needed_perts = {p for perts in mapping.values() for p in perts}

    sig = pd.read_csv(sig_info, sep="\t", low_memory=False)
    sig = sig[(sig["pert_type"] == "trt_cp") & (sig["pert_id"].isin(needed_perts))]
    sig_to_pert = dict(zip(sig["sig_id"], sig["pert_id"]))
    print(f"  signatures to read               : {len(sig_to_pert):,} "
          f"for {sig['pert_id'].nunique():,} compounds")

    gene_ids, symbols = landmark_genes(gene_info)
    print(f"  landmark genes                   : {len(gene_ids)}")

    started = time.time()
    landmark_df = read_landmark_matrix(gctx, gene_ids, list(sig_to_pert))
    print(f"  read {landmark_df.shape} from gctx in {time.time() - started:.0f}s")

    consensus = consensus_by_pert(landmark_df, sig_to_pert)
    frame = expand_to_nsc(consensus, mapping)
    frame.columns = [f"g{gid}" for gid in gene_ids]
    frame = frame.astype(np.float32)

    args.out.mkdir(parents=True, exist_ok=True)
    destination = args.out / "drug_signatures_l1000.parquet"
    frame.to_parquet(destination)

    covered = len(frame)
    pool = len(wanted) if wanted is not None else len(nsc_keys)
    print(f"\n  shape      : {frame.shape}")
    print(f"  coverage   : {covered:,} of {pool:,} study drugs "
          f"({covered / max(pool, 1):.0%}) have an L1000 signature")
    print(f"  -> {destination}  ({destination.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
