"""Offline self-test for the workshop: reference structures and data invariants.

    python utils/selftest.py

Run it after editing `src_nb/01_cyp3a4_chembl_vs_openadmet.py` or regenerating `data/`.
It needs no network: every expected value is pinned here, and the structures are checked by
**InChIKey**, which is what a chemical identity check should use.

Why this file exists: during development, two of the named reference drugs in the notebook were
silently wrong (a ritonavir SMILES with one CH2 too many, and an "itraconazole" that was a
different molecule altogether). Both parsed fine and drew fine. Only an identity check caught
them, so now the identity check is part of the repository.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src_nb" / "01_cyp3a4_chembl_vs_openadmet.py"
DATA = ROOT / "data"

# name -> (ChEMBL id, standard InChIKey, molecular formula). Verified against the ChEMBL API.
REFERENCE_STRUCTURES = {
    "ketoconazole": ("CHEMBL157101", "XMAYWYJOQHXEEK-UHFFFAOYSA-N", "C26H28Cl2N4O4"),
    "ritonavir":    ("CHEMBL163",    "NCDNCNXCDXHOMX-XGKFQTDJSA-N", "C37H48N6O5S2"),
    "midazolam":    ("CHEMBL655",    "DDLIGBOFAVUZHB-UHFFFAOYSA-N", "C18H13ClFN3"),
    "itraconazole": ("CHEMBL64391",  "VHVPQPYKVGDNFY-UHFFFAOYSA-N", "C35H38Cl2N8O4"),
    "erythromycin": ("CHEMBL532",    "ULGZDMOVFRHVEP-RWJQBGPGSA-N", "C37H67NO13"),
    "caffeine":     ("CHEMBL113",    "RYYVLZVUVIJVGH-UHFFFAOYSA-N", "C8H10N4O2"),
    "aspirin":      ("CHEMBL25",     "BSYNRYMUTXBXSQ-UHFFFAOYSA-N", "C9H8O4"),
}

# The three stereochemistry twins discussed in section 5 - the reason exact overlap is defined on
# isomeric structure identity and not on a fingerprint similarity of 1.0.
STEREO_TWINS = {
    "curcumin (OpenADMET, no C=C geometry)": "VFLDPWHFBUODDF-UHFFFAOYSA-N",
    "curcumin (ChEMBL, E,E)":                "VFLDPWHFBUODDF-FCXRPNKRSA-N",
    "betamethasone (OpenADMET)":             "UREBDLICKHMUKA-DVTGEIKXSA-N",
    "dexamethasone (ChEMBL)":                "UREBDLICKHMUKA-CXSFZGCWSA-N",
}

EXPECTED_COUNTS = {
    "chembl_cyp3a4_ic50_raw.csv.gz": 13887,
    "chembl_cyp3a4_assays.csv.gz": 2810,
    "chembl_cyp3a4_assay_review.csv": 2810,
    "chembl_cyp3a4_curated.csv": 3980,
    "chembl_cyp3a4_censored.csv": 3710,
    "chembl_cyp3a4_structures.csv": 11237,
    "openadmet_cyp_train_inhibition.csv": 4905,
    "openadmet_cyp3a4_curated.csv": 2335,
}

failures: list[str] = []


def check(ok: bool, message: str) -> None:
    print(f"  {'ok  ' if ok else 'FAIL'}  {message}")
    if not ok:
        failures.append(message)


def extract_smiles(source: str) -> dict[str, str]:
    """Pull the named reference SMILES out of the notebook source (KNOWN table and PAIRS dict)."""
    found = {}
    for name in REFERENCE_STRUCTURES:
        m = (re.search(r'\("' + name + r'",\s*"([^"]+)"', source)
             or re.search(r'"' + name + r'":\s*"([^"]+)"', source))
        if m:
            found[name] = m.group(1)
    return found


def main() -> int:
    source = SRC.read_text()

    print("reference structures in the notebook (identity by InChIKey):")
    found = extract_smiles(source)
    for name, (chembl_id, inchikey, formula) in REFERENCE_STRUCTURES.items():
        if name not in found:
            check(False, f"{name}: no SMILES found in {SRC.name}")
            continue
        mol = Chem.MolFromSmiles(found[name])
        if mol is None:
            check(False, f"{name}: SMILES does not parse")
            continue
        got_key = Chem.MolToInchiKey(mol)
        got_formula = rdMolDescriptors.CalcMolFormula(mol)
        check(got_key == inchikey and got_formula == formula,
              f"{name:14s} {chembl_id:12s} {got_key} {got_formula}"
              + ("" if got_key == inchikey else f"   expected {inchikey} {formula}"))

    print("\nstereochemistry twins (section 5) still resolve to distinct InChIKeys:")
    keys = list(STEREO_TWINS.values())
    check(len(set(keys)) == len(keys), "all four twin InChIKeys are distinct")
    check(keys[0].split("-")[0] == keys[1].split("-")[0],
          "curcumin pair shares its connectivity layer")
    check(keys[2].split("-")[0] == keys[3].split("-")[0],
          "betamethasone/dexamethasone pair shares its connectivity layer")

    print("\ncached data row counts:")
    for filename, expected in EXPECTED_COUNTS.items():
        path = DATA / filename
        if not path.exists():
            check(False, f"{filename}: missing")
            continue
        n = len(pd.read_csv(path, low_memory=False))
        check(n == expected, f"{filename:38s} {n:6,} rows (expected {expected:,})")

    print("\ncuration invariants:")
    quant = pd.read_csv(DATA / "chembl_cyp3a4_curated.csv")
    censored = pd.read_csv(DATA / "chembl_cyp3a4_censored.csv")
    oa = pd.read_csv(DATA / "openadmet_cyp3a4_curated.csv")
    structures = pd.read_csv(DATA / "chembl_cyp3a4_structures.csv")
    prov = json.loads((DATA / "provenance.json").read_text())

    check((quant["standard_relation"] == "=").all(), "quantitative set is all relation '='")
    check(censored["standard_relation"].isin([">", "<", ">=", "<="]).all(),
          "censored set contains only censored relations")
    check((quant["standard_units"] == "nM").all(), "quantitative set is all nM")
    check((quant["standard_value"] > 0).all(), "quantitative values are all positive")
    import numpy as np
    check(np.allclose(quant["pIC50"], 9 - np.log10(quant["standard_value"])),
          "pIC50 == 9 - log10(IC50 in nM) throughout")
    check(structures["smiles_std"].is_unique, "the reference structure table is one row per structure")
    check(oa["smiles_std"].notna().all(), "every OpenADMET compound standardised")
    check(oa[["pIC50_conf_low", "pIC50_conf_high", "pIC50_std"]].notna().all().all(),
          "OpenADMET uncertainty columns preserved for every row")
    check((oa["weak_potency_flag"] == (oa["pIC50"] < 4)).all(),
          "weak_potency_flag is exactly pIC50 < 4 (a flag, not a filter)")

    overlap = len(set(structures["smiles_std"]) & set(oa["smiles_std"]))
    check(overlap == prov["curation"]["exact_overlap_structures"],
          f"exact structure overlap = {overlap}, matching provenance.json")

    print(f"\n{'FAILED' if failures else 'PASSED'}: "
          f"{len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
