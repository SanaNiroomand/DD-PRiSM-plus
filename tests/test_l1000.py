"""Tests for the L1000 drug-signature builder.

These cover the parts that decide correctness -- InChIKey matching between NSC
and L1000 compounds, and the per-compound consensus -- without needing the
23 GB GCTX or cmapPy. The GCTX read itself is a thin cmapPy call.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.fetch_l1000 import (canonical_inchikey, consensus_by_pert,
                                 expand_to_nsc, match_nsc_to_pert,
                                 pert_inchikeys, skeleton)
from ddprism.drugfeatures import FEATURE_SOURCES


def test_l1000_is_registered():
    assert FEATURE_SOURCES["l1000"] == "drug_signatures_l1000.parquet"


def test_canonical_inchikey_is_canonical():
    # two different but equivalent SMILES for aspirin -> same InChIKey
    a = canonical_inchikey("CC(=O)Oc1ccccc1C(=O)O")
    b = canonical_inchikey("O=C(O)c1ccccc1OC(C)=O")
    assert a is not None and a == b
    assert len(a.split("-")) == 3          # standard InChIKey shape


def test_canonical_inchikey_rejects_garbage():
    assert canonical_inchikey("not a molecule") is None
    assert canonical_inchikey("") is None
    assert canonical_inchikey(None) is None


def test_skeleton_is_first_block():
    key = canonical_inchikey("CC(=O)Oc1ccccc1C(=O)O")
    assert skeleton(key) == key.split("-")[0]
    assert len(skeleton(key)) == 14


def test_pert_inchikeys_recomputes_missing_key():
    aspirin = canonical_inchikey("CC(=O)Oc1ccccc1C(=O)O")
    frame = pd.DataFrame([
        {"pert_id": "BRD-A", "pert_type": "trt_cp",
         "inchi_key": aspirin, "canonical_smiles": "CC(=O)Oc1ccccc1C(=O)O"},
        {"pert_id": "BRD-B", "pert_type": "trt_cp",
         "inchi_key": "-666", "canonical_smiles": "CC(=O)Oc1ccccc1C(=O)O"},
        {"pert_id": "SH-1", "pert_type": "trt_sh",
         "inchi_key": "-666", "canonical_smiles": "-666"},
    ])
    keys = pert_inchikeys(frame)
    assert keys["BRD-A"] == aspirin
    assert keys["BRD-B"] == aspirin        # recovered from SMILES
    assert "SH-1" not in keys              # not a compound, no usable key


def test_match_exact_and_skeleton_are_separated():
    # same connectivity, one a salt variant -> same skeleton, different full key
    nsc_keys = {101: "AAAAAAAAAAAAAA-BBBBBBBBBB-N",
                102: "CCCCCCCCCCCCCC-DDDDDDDDDD-N"}
    pert_keys = {"P1": "AAAAAAAAAAAAAA-BBBBBBBBBB-N",      # exact for 101
                 "P2": "CCCCCCCCCCCCCC-ZZZZZZZZZZ-N"}      # skeleton for 102
    exact, skel = match_nsc_to_pert(nsc_keys, pert_keys)
    assert exact == {101: ["P1"]}
    assert skel == {102: ["P2"]}


def test_consensus_is_mean_over_a_compounds_signatures():
    # 2 genes x 3 signatures; P1 has two signatures, P2 one
    landmark = pd.DataFrame(
        [[1.0, 3.0, 10.0],
         [2.0, 4.0, 20.0]],
        index=["g1", "g2"], columns=["s1", "s2", "s3"])
    sig_to_pert = {"s1": "P1", "s2": "P1", "s3": "P2"}
    consensus = consensus_by_pert(landmark, sig_to_pert)
    assert set(consensus.index) == {"P1", "P2"}
    np.testing.assert_allclose(consensus.loc["P1"].to_numpy(), [2.0, 3.0])
    np.testing.assert_allclose(consensus.loc["P2"].to_numpy(), [10.0, 20.0])


def test_expand_to_nsc_averages_multiple_perts():
    consensus = pd.DataFrame(
        [[2.0, 3.0], [4.0, 5.0]],
        index=["P1", "P2"], columns=["g1", "g2"])
    frame = expand_to_nsc(consensus, {101: ["P1"], 102: ["P1", "P2"]})
    assert list(frame.index) == [101, 102]
    np.testing.assert_allclose(frame.loc[101].to_numpy(), [2.0, 3.0])
    np.testing.assert_allclose(frame.loc[102].to_numpy(), [3.0, 4.0])  # mean


def test_expand_to_nsc_drops_unmatched():
    consensus = pd.DataFrame([[1.0]], index=["P1"], columns=["g1"])
    frame = expand_to_nsc(consensus, {101: ["P_absent"]})
    assert len(frame) == 0


def test_select_landmark_phase1_schema():
    from scripts.fetch_l1000 import select_landmark
    info = pd.DataFrame({
        "pr_gene_id": [10, 5, 7],
        "pr_is_lm": [1, 1, 0],
        "pr_gene_symbol": ["GJ", "GE", "NO"],
    })
    ids, symbols = select_landmark(info)
    assert ids == ["5", "10"]            # landmark only, sorted by id
    assert symbols == ["GE", "GJ"]


def test_select_landmark_beta_schema():
    from scripts.fetch_l1000 import select_landmark
    info = pd.DataFrame({
        "gene_id": [10, 5, 7],
        "feature_space": ["landmark", "landmark", "inferred"],
        "gene_symbol": ["GJ", "GE", "NO"],
    })
    ids, symbols = select_landmark(info)
    assert ids == ["5", "10"]
    assert symbols == ["GE", "GJ"]
