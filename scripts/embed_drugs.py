"""Embed every drug with a pretrained chemical language model.

The paper represents a drug as a 512-bit Morgan fingerprint: a bag of
substructures, learned from nothing. A model pretrained on hundreds of millions
of SMILES has seen chemistry instead, and its embedding carries information a
fingerprint cannot -- which is precisely what the unseen-drug split is short of.

This writes one parquet indexed by NSC, aligned with fingerprints.parquet, ready
for ``--drug-features morgan+chemberta``.

    python scripts/embed_drugs.py --data data --out processed --model chemberta

Models:
  chemberta   DeepChem/ChemBERTa-77M-MTR   384 dims, ~3M params, no remote code.
              Pretrained on 77M PubChem SMILES with a multitask regression head
              over molecular properties, which makes its embeddings noticeably
              more property-aware than the plain MLM variant.
  molformer   ibm/MoLFormer-XL-both-10pct  768 dims, ~44M params.
              Stronger, but it ships a custom attention implementation that
              transformers will only run with trust_remote_code=True -- i.e. it
              executes code downloaded from the Hub. That is opt-in here, via
              --trust-remote-code, rather than something this script does
              quietly on your behalf.

SMILES come from the same Chem2D_Jun2016 file the fingerprints are built from,
so coverage matches exactly and no drug gains or loses a representation.
"""

import argparse
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ddprism.drugfeatures import used_drug_ids  # noqa: E402

MODELS = {
    "chemberta": {"repo": "DeepChem/ChemBERTa-77M-MTR", "remote_code": False},
    "chemberta-mlm": {"repo": "DeepChem/ChemBERTa-77M-MLM", "remote_code": False},
    "chemberta-zinc": {"repo": "seyonec/ChemBERTa-zinc-base-v1", "remote_code": False},
    "molformer": {"repo": "ibm/MoLFormer-XL-both-10pct", "remote_code": True},
}

# Which feature source each model writes. The variants all land in the
# chemberta slot, so swapping between them needs no change downstream.
OUTPUT_SLOT = {"chemberta": "chemberta", "chemberta-mlm": "chemberta",
               "chemberta-zinc": "chemberta", "molformer": "molformer"}


def banner(text):
    print(f"\n{'=' * 66}\n{text}\n{'=' * 66}", flush=True)


def read_smiles(data, source, wanted=None):
    """(NSC, canonical SMILES) for every compound we can parse.

    Canonicalising through RDKit matters: the pretrained tokenizers were trained
    on canonical SMILES, and feeding them the SDF's arbitrary atom ordering
    would put the model off-distribution for no reason.
    """
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    pairs, seen, unparsed = {}, 0, 0

    if source == "chem2d":
        archive = zipfile.ZipFile(data / "Chem2D_Jun2016.zip")
        with archive.open(archive.namelist()[0]) as handle:
            for mol in Chem.ForwardSDMolSupplier(handle):
                seen += 1
                if mol is None:
                    unparsed += 1
                    continue
                try:
                    nsc = int(mol.GetProp("_Name"))
                except (KeyError, ValueError):
                    continue
                if wanted is not None and nsc not in wanted:
                    continue
                if nsc in pairs:
                    continue
                pairs[nsc] = Chem.MolToSmiles(mol)
    else:
        frame = pd.read_csv(data / "nsc_smiles.csv")
        seen = len(frame)
        for nsc, smi in zip(frame.NSC.values, frame.SMILES.values):
            nsc = int(nsc)
            if wanted is not None and nsc not in wanted:
                continue
            if nsc in pairs:
                continue
            mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
            if mol is None:
                unparsed += 1
                continue
            pairs[nsc] = Chem.MolToSmiles(mol)

    print(f"  compound records read : {seen:,}")
    print(f"  unparseable, dropped  : {unparsed:,}")
    print(f"  SMILES kept           : {len(pairs):,}")
    return pairs


def load_model(name, trust_remote_code, device):
    import torch
    from transformers import AutoModel, AutoTokenizer

    spec = MODELS[name]
    if spec["remote_code"] and not trust_remote_code:
        raise SystemExit(
            f"{spec['repo']} ships a custom model implementation that only runs "
            f"with trust_remote_code=True, which executes Python downloaded from "
            f"the Hugging Face Hub.\n"
            f"  Pass --trust-remote-code if you accept that, or use "
            f"--model chemberta, which needs no remote code.")

    kwargs = {"trust_remote_code": True} if spec["remote_code"] else {}
    tokenizer = AutoTokenizer.from_pretrained(spec["repo"], **kwargs)
    model = AutoModel.from_pretrained(spec["repo"], **kwargs)
    model.eval().to(device)

    parameters = sum(p.numel() for p in model.parameters())
    print(f"  model      : {spec['repo']}")
    print(f"  parameters : {parameters:,}")
    return tokenizer, model


def embed(tokenizer, model, smiles, device, batch_size=256, max_length=256,
          pooling="mean"):
    """Mean-pooled last hidden state, one vector per molecule.

    Mean pooling over real tokens, not the CLS vector: without task
    fine-tuning a RoBERTa-style CLS embedding is close to uninformative, while
    the token mean is a solid sentence-level summary. The attention mask keeps
    padding out of the average -- forgetting it makes every vector a function of
    how long the longest molecule in its batch happened to be.
    """
    import torch

    # Batches of similar length waste far less compute on padding. The original
    # order is restored before returning.
    order = np.argsort([len(s) for s in smiles], kind="stable")
    vectors = [None] * len(smiles)
    started = time.time()

    with torch.no_grad():
        for start in range(0, len(order), batch_size):
            picked = order[start:start + batch_size]
            encoded = tokenizer([smiles[i] for i in picked], padding=True,
                                truncation=True, max_length=max_length,
                                return_tensors="pt").to(device)
            output = model(**encoded).last_hidden_state

            if pooling == "cls":
                pooled = output[:, 0]
            else:
                mask = encoded["attention_mask"].unsqueeze(-1).to(output.dtype)
                pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

            pooled = pooled.float().cpu().numpy()
            for row, index in enumerate(picked):
                vectors[index] = pooled[row]

            done = start + len(picked)
            if done % (batch_size * 20) == 0 or done == len(order):
                rate = done / max(time.time() - started, 1e-9)
                print(f"    {done:>7,} / {len(order):,}   {rate:,.0f} mol/s",
                      flush=True)

    return np.stack(vectors)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("processed"))
    parser.add_argument("--model", choices=sorted(MODELS), default="chemberta")
    parser.add_argument("--source", choices=["chem2d", "nsc_smiles"],
                        default="chem2d",
                        help="chem2d matches what the fingerprints are built from")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--pooling", choices=["mean", "cls"], default="mean")
    parser.add_argument("--trust-remote-code", action="store_true",
                        help="allow models that execute code from the Hub")
    parser.add_argument("--all-compounds", action="store_true",
                        help="embed the whole 281k Chem2D library rather than "
                             "just the drugs the response tables use")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    import torch
    device = torch.device(args.device or
                          ("cuda" if torch.cuda.is_available() else "cpu"))

    banner(f"drug embeddings: {args.model}  (device: {device})")

    wanted = None
    if not args.all_compounds:
        wanted = used_drug_ids(args.out)
        if wanted is None:
            print("  no response tables found under --out; embedding everything")
        else:
            print(f"  drugs used by the response tables : {len(wanted):,}")

    pairs = read_smiles(args.data, args.source, wanted)
    if not pairs:
        raise SystemExit("no SMILES survived; is --data pointing at the raw "
                         "download directory?")

    nsc = sorted(pairs)
    smiles = [pairs[n] for n in nsc]
    tokenizer, model = load_model(args.model, args.trust_remote_code, device)

    print(f"  embedding {len(smiles):,} molecules")
    vectors = embed(tokenizer, model, smiles, device, args.batch_size,
                    args.max_length, args.pooling)

    frame = pd.DataFrame(vectors.astype(np.float32),
                         index=pd.Index(nsc, name="NSC"))
    frame.columns = [f"e{i}" for i in range(frame.shape[1])]

    args.out.mkdir(parents=True, exist_ok=True)
    slot = OUTPUT_SLOT[args.model]
    destination = args.out / f"drug_embeddings_{slot}.parquet"
    frame.to_parquet(destination)

    print(f"\n  shape      : {frame.shape}")
    print(f"  norm       : mean {np.linalg.norm(vectors, axis=1).mean():.3f}")
    print(f"  dead dims  : {int((vectors.std(axis=0) < 1e-6).sum())} of "
          f"{vectors.shape[1]}")
    print(f"  -> {destination}  ({destination.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
