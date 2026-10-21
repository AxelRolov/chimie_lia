"""Prepare the cached teaching data for the CYP3A4 workshop.

Run this **before the class**, with a working network connection:

    python utils/prepare_cached_data.py            # refresh everything
    python utils/prepare_cached_data.py --offline  # re-derive only, from the cached raw files

What it produces in `data/`:

| file | what it is |
|---|---|
| `chembl_cyp3a4_ic50_raw.csv.gz`   | every ChEMBL IC50 activity record for CHEMBL340, as returned by the REST API |
| `chembl_cyp3a4_assays.csv.gz`     | the assay record of every assay appearing in the file above |
| `chembl_cyp3a4_assay_review.csv`  | one row per assay with the automatic context flags (the instructor-reviewed selection) |
| `chembl_cyp3a4_curated.csv`       | the core quantitative subset: one row per measurement, standardised structures, pIC50 |
| `chembl_cyp3a4_censored.csv`      | the records with `>` / `<` relations, kept apart — never mixed into the quantitative set |
| `chembl_cyp3a4_structures.csv`    | one row per unique standardised ChEMBL structure (the similarity reference set) |
| `openadmet_cyp_train_inhibition.csv` | byte-for-byte copy of the pinned Hugging Face training CSV |
| `openadmet_cyp3a4_curated.csv`    | the 2 335 CYP3A4 rows, standardised, uncertainty columns preserved |
| `chembl_ui_export_sample.csv`     | 200 rows of a ChEMBL *website* CSV export, for the manual-download section |
| `provenance.json`                 | release, retrieval date, every query URL, row counts and SHA-256 checksums |

Nothing here invents data: if a download fails the script stops and says so, rather than
substituting a different target or synthetic molecules.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import sys
import time
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

EBI = "https://www.ebi.ac.uk"
TARGET = "CHEMBL340"          # human CYP3A4
UNIPROT = "P08684"
ACTIVITY_URL = f"{EBI}/chembl/api/data/activity.json?target_chembl_id={TARGET}&standard_type=IC50&limit=1000"

# The OpenADMET training file, pinned to one commit so the numbers in the notebook stay true.
HF_REVISION = "146282d2ce1532f1803679a834a9bec9dac1e482"
HF_URL = (f"https://huggingface.co/datasets/openadmet/cyp-challenge-train-test/resolve/"
          f"{HF_REVISION}/cyp-challenge-TRAIN_inhibition.csv")
HF_SHA256 = "b8f79addd266fb6f9f4c222c5e4e73d926362328b6a8d2841871a54e46bd2278"

CYP = "CYP3A4_pIC50_direct_inhibition"


def session() -> requests.Session:
    s = requests.Session()
    s.mount("https://", HTTPAdapter(max_retries=Retry(
        total=5, backoff_factor=1.5, status_forcelist=[429, 500, 502, 503, 504])))
    s.headers.update({"User-Agent": "chimie_lia-workshop/1.0 (teaching material)"})
    return s


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------------------
# 1. Download
# --------------------------------------------------------------------------------------

def verify_target(s: requests.Session) -> dict:
    """Fetch the target record and check it really is the human CYP3A4 single protein."""
    t = s.get(f"{EBI}/chembl/api/data/target/{TARGET}.json", timeout=60).json()
    accessions = [c["accession"] for c in t["target_components"]]
    assert t["organism"] == "Homo sapiens", t["organism"]
    assert t["target_type"] == "SINGLE PROTEIN", t["target_type"]
    assert UNIPROT in accessions, accessions
    print(f"target  {TARGET}  {t['pref_name']}  {t['organism']}  {t['target_type']}  {accessions}")
    return t


def download_activities(s: requests.Session) -> pd.DataFrame:
    """Every IC50 activity record for the target, following the API's own paging links."""
    url, records, t0 = ACTIVITY_URL, [], time.time()
    while url:
        r = s.get(url, timeout=120)
        r.raise_for_status()
        page = r.json()
        records += page["activities"]
        nxt = page["page_meta"]["next"]
        url = EBI + nxt if nxt else None
        print(f"\r  {len(records):6d} / {page['page_meta']['total_count']} records "
              f"({time.time() - t0:.0f} s)", end="", flush=True)
    print()
    return pd.DataFrame(records)


def download_assays(s: requests.Session, assay_ids: list[str]) -> pd.DataFrame:
    """Assay metadata, 100 ids per request (the API's `__in` filter)."""
    out = []
    for i in range(0, len(assay_ids), 100):
        chunk = assay_ids[i:i + 100]
        r = s.get(f"{EBI}/chembl/api/data/assay.json",
                  params={"assay_chembl_id__in": ",".join(chunk), "limit": 1000}, timeout=120)
        r.raise_for_status()
        out += r.json()["assays"]
        print(f"\r  {len(out):5d} / {len(assay_ids)} assays", end="", flush=True)
    print()
    return pd.DataFrame(out)


def download_ui_export_sample(s: requests.Session) -> pd.DataFrame | None:
    """Ask the ChEMBL *website* for the same activity list as a CSV.

    This is exactly what the "CSV" button on https://www.ebi.ac.uk/chembl/explore/activities/...
    does: it submits a download job and polls it. We keep 200 rows of the result so that the
    manual-download section of the notebook has a realistic file to parse (`;` separated,
    quoted, UI column names) even when a student's own export fails.
    """
    api = f"{EBI}/chembl/interface_api/delayed_jobs"
    q = {"bool": {"must": [{"query_string": {"query": f'target_chembl_id:{TARGET} AND standard_type:"IC50"'}}],
                  "filter": [], "should": [], "must_not": []}}
    try:
        r = s.post(f"{api}/submit/download_job",
                   data={"index_name": "chembl_activity", "query": json.dumps(q),
                         "format": "CSV", "download_columns_group": "download",
                         "dl__ignore_cache": "false"}, timeout=60)
        r.raise_for_status()
        job = r.json()["job_id"]
        for _ in range(60):
            st = s.get(f"{api}/status/{job}", timeout=30).json()
            if st["status"] in ("FINISHED", "ERROR"):
                break
            time.sleep(5)
        if st["status"] != "FINISHED":
            raise RuntimeError(st["status"])
        url = list(st["output_files_urls"].values())[0]
        url = url if url.startswith("http") else "https://" + url.lstrip("/")
        z = zipfile.ZipFile(io.BytesIO(s.get(url, timeout=180).content))
        raw = z.read(z.namelist()[0]).decode("utf-8")
    except Exception as e:                                     # the website export is a bonus
        print(f"  website export unavailable ({type(e).__name__}: {e}) - keeping the previous sample")
        return None
    lines = raw.splitlines()
    return pd.read_csv(io.StringIO("\n".join(lines[:201])), sep=";")


def download_openadmet(s: requests.Session) -> Path:
    out = DATA / "openadmet_cyp_train_inhibition.csv"
    r = s.get(HF_URL, timeout=180)
    r.raise_for_status()
    out.write_bytes(r.content)
    got = sha256(out)
    if got != HF_SHA256:
        raise SystemExit(f"checksum mismatch for the OpenADMET file:\n  expected {HF_SHA256}\n  got      {got}")
    print(f"openadmet  {out.name}  sha256 ok")
    return out


# --------------------------------------------------------------------------------------
# 2. Assay context: what was actually measured?
# --------------------------------------------------------------------------------------
# CYP3A4 appears in ChEMBL under very different experiments. Before comparing numbers we read
# the assay descriptions and label them. These patterns were written by reading the 100 largest
# assay descriptions and spot-checking the rest; `chembl_cyp3a4_assay_review.csv` is the result
# and is meant to be re-read (and edited) by the instructor before a class.

CONTEXT_PATTERNS = {
    # pre-incubating the compound *with NADPH* before adding substrate measures time-dependent
    # (mechanism-based) inhibition, not direct inhibition - a different quantity.
    "time_dependent": r"time[- ]dependent|pre-?incubat|mechanism[- ]based|inactivat|ic50 shift|\btdi\b",
    "induction": r"\binduc(?:tion|ed|ing)\b",
    "activation": r"\bactivation\b|\bactivator\b",
}

PROBE_PATTERNS = {
    "midazolam": r"midazolam",
    "testosterone": r"testosteron",
    "BFC": r"\bbfc\b|benzyloxy-?4-?\(?trifluoromethyl|benzoyloxy-4-trifluoromethyl",
    "BQ": r"benzyloxyquinoline|\bbq\b",
    "DBF": r"dibenzylfluorescein|\bdbf\b|diethoxyfluorescein",
    "luciferin": r"luciferin|p450-?glo",
    "nifedipine": r"nifedipine|felodipine",
    "erythromycin": r"erythromycin",
    "Vivid/fluorogenic": r"vivid|bomcc|bomr|dbomf",
}

SYSTEM_PATTERNS = {
    "human liver microsomes": r"microsom|\bhlm\b",
    "recombinant enzyme": r"recombinant|supersome|baculosome|bactosome|insect cell|expressed in|cdna-expressed",
    "cell or hepatocyte": r"\bcells?\b|hepatocyte",
}


# The description must give POSITIVE evidence that the experiment is about this enzyme. Defaulting
# to "direct inhibition unless a keyword says otherwise" let through assays that are not CYP3A4
# experiments at all: ChEMBL links some activities to CHEMBL340 that come from an MDM2-p53 HTRF
# assay, a pig-liver microsomal preparation and an ATRA 4-hydroxylase assay - and the MDM2 one
# supplied the four most potent values in the whole curated set. Confidence score 9 does not
# protect against this, because the score describes the target *assignment*, not the experiment.
MENTIONS_TARGET = r"3a4|\bcyp\b|cyp\d|p450|cytochrome"

# Assays an instructor has looked at and rejected, with the reason. This is the manual half of the
# review: edit it, re-run this script, and the funnel changes accordingly.
INSTRUCTOR_EXCLUDED = {
    "CHEMBL3705484": "HTRF MDM2-p53 binding assay, not a CYP3A4 experiment",
    "CHEMBL3705428": "pig liver microsomal preparation, not human CYP3A4",
    "CHEMBL5737369": "generic microsomal enzyme-activity assay, enzyme not identified",
}


def first_match(text: str, patterns: dict[str, str]) -> str:
    import re
    for name, pat in patterns.items():
        if re.search(pat, text):
            return name
    return ""


def review_assays(assays: pd.DataFrame, acts: pd.DataFrame) -> pd.DataFrame:
    import re
    n = acts.groupby("assay_chembl_id").agg(
        n_records=("activity_id", "size"),
        n_equals=("standard_relation", lambda s: int((s == "=").sum())),
        n_censored=("standard_relation", lambda s: int(s.isin([">", "<", ">=", "<="]).sum())),
    )
    a = assays.set_index("assay_chembl_id").join(n).reset_index()
    desc = a["description"].fillna("").str.lower()
    for name, pat in CONTEXT_PATTERNS.items():
        a[name] = desc.str.contains(pat, regex=True)
    a["probe_substrate"] = [first_match(d, PROBE_PATTERNS) for d in desc]
    a["enzyme_system"] = [first_match(d, SYSTEM_PATTERNS) for d in desc]

    # Positive evidence that the description is about this enzyme at all.
    a["mentions_target"] = desc.str.contains(MENTIONS_TARGET, regex=True)
    a["excluded_reason"] = a["assay_chembl_id"].map(INSTRUCTOR_EXCLUDED).fillna("")
    a.loc[~a["mentions_target"] & (a["excluded_reason"] == ""), "excluded_reason"] = \
        "description never mentions CYP3A4 / CYP / P450"
    a["instructor_excluded"] = a["excluded_reason"] != ""

    # "direct inhibition" = positive evidence it is a CYP3A4 experiment, no competing context,
    # and not excluded by review.
    a["direct_inhibition"] = (a["mentions_target"]
                              & ~(a["time_dependent"] | a["induction"] | a["activation"])
                              & ~a["instructor_excluded"])
    cols = ["assay_chembl_id", "n_records", "n_equals", "n_censored", "confidence_score",
            "assay_type", "bao_label", "assay_organism", "relationship_type", "src_id",
            "direct_inhibition", "mentions_target", "time_dependent", "induction", "activation",
            "instructor_excluded", "excluded_reason", "probe_substrate", "enzyme_system",
            "description"]
    return a[cols].sort_values("n_records", ascending=False)


# --------------------------------------------------------------------------------------
# 3. One standardisation policy, applied to both sources
# --------------------------------------------------------------------------------------
# Documented policy, applied identically to ChEMBL and OpenADMET structures:
#   1. parse the SMILES (record failures, never drop them silently);
#   2. Cleanup()           - sanitise, normalise functional groups, reionise;
#   3. LargestFragmentChooser(preferOrganic=True) - strip counter-ions and solvents;
#   4. Uncharger()         - neutralise where chemically sensible;
#   5. canonical *isomeric* SMILES - stereochemistry is KEPT.
# No tautomer canonicalisation: it is slow, it is a modelling choice, and ChEMBL's own
# structure pipeline does not do it either. The original SMILES is always preserved.

_cleanup_params = rdMolStandardize.CleanupParameters()
_largest_fragment = rdMolStandardize.LargestFragmentChooser(preferOrganic=True)
_uncharger = rdMolStandardize.Uncharger()


def standardize(smiles: str) -> dict:
    out = {"smiles_std": None, "inchikey": None, "n_fragments": None,
           "had_charge": None, "std_status": "ok"}
    if not isinstance(smiles, str) or not smiles.strip():
        out["std_status"] = "empty"
        return out
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        out["std_status"] = "unparsable"
        return out
    out["n_fragments"] = len(Chem.GetMolFrags(mol))
    out["had_charge"] = any(a.GetFormalCharge() != 0 for a in mol.GetAtoms())
    try:
        mol = rdMolStandardize.Cleanup(mol, _cleanup_params)
        mol = _largest_fragment.choose(mol)
        mol = _uncharger.uncharge(mol)
        Chem.SanitizeMol(mol)
        out["smiles_std"] = Chem.MolToSmiles(mol, isomericSmiles=True)
        out["inchikey"] = Chem.MolToInchiKey(mol)
    except Exception:
        out["std_status"] = "standardisation failed"
    return out


DESCRIPTORS = {
    "MW": Descriptors.MolWt,
    "cLogP": Descriptors.MolLogP,
    "TPSA": rdMolDescriptors.CalcTPSA,
    "HBD": rdMolDescriptors.CalcNumHBD,
    "HBA": rdMolDescriptors.CalcNumHBA,
    "RotB": rdMolDescriptors.CalcNumRotatableBonds,
    "AromRings": rdMolDescriptors.CalcNumAromaticRings,
    "FracCSP3": rdMolDescriptors.CalcFractionCSP3,
    "HeavyAtoms": lambda m: m.GetNumHeavyAtoms(),
}


def describe(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {k: np.nan for k in DESCRIPTORS}
    return {k: f(mol) for k, f in DESCRIPTORS.items()}


def murcko(smiles: str) -> tuple[str, bool]:
    """Bemis-Murcko scaffold; the flag says whether the molecule has any ring at all."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "", False
    scaf = MurckoScaffold.GetScaffoldForMol(mol)
    smi = Chem.MolToSmiles(scaf)
    return smi, smi != ""


# --------------------------------------------------------------------------------------
# 4. Curation
# --------------------------------------------------------------------------------------

def curate_chembl(acts: pd.DataFrame, review: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (core quantitative set, censored set, funnel table).

    Rules, all of them visible in the funnel table the notebook prints:
      - the assay has ChEMBL confidence score 9 (direct single-protein assignment);
      - the assay description is consistent with *direct* inhibition (no pre-incubation / TDI,
        no induction, no activation);
      - IC50 with `standard_relation` "=", units nM, a finite value > 0;
      - no ChEMBL data-validity comment;
      - a parsable structure.
    Records failing only the relation test are kept in the censored table, not thrown away.
    Assay type is NOT filtered: most CYP3A4 records are typed `A` (ADME), not `B` (binding).
    """
    rows = []
    df = acts.copy()
    rows.append(("all IC50 activity records for CHEMBL340", len(df)))

    rv = review.set_index("assay_chembl_id")
    df["confidence_score"] = df["assay_chembl_id"].map(rv["confidence_score"])
    df["direct_inhibition"] = df["assay_chembl_id"].map(rv["direct_inhibition"])
    df["probe_substrate"] = df["assay_chembl_id"].map(rv["probe_substrate"])
    df["enzyme_system"] = df["assay_chembl_id"].map(rv["enzyme_system"])

    df = df[df["target_organism"] == "Homo sapiens"]
    rows.append(("target organism is human", len(df)))
    df = df[df["confidence_score"] == 9]
    rows.append(("assay confidence score 9 (direct protein assignment)", len(df)))
    df = df[df["direct_inhibition"]]
    rows.append(("assay context is direct inhibition (no TDI / induction)", len(df)))
    df = df[df["canonical_smiles"].notna()]
    rows.append(("has a structure", len(df)))
    df = df[df["data_validity_comment"].isna()]
    rows.append(("no ChEMBL data-validity comment", len(df)))

    value = pd.to_numeric(df["standard_value"], errors="coerce")
    df = df[np.isfinite(value) & (value > 0) & (df["standard_units"] == "nM")]
    rows.append(("finite positive value in nM", len(df)))

    censored = df[df["standard_relation"].isin([">", "<", ">=", "<="])].copy()
    core = df[df["standard_relation"] == "="].copy()
    rows.append(("standard_relation '=' (the quantitative set)", len(core)))
    other = df[~df["standard_relation"].isin(["=", ">", "<", ">=", "<="])]
    if len(other):
        print(f"  note: {len(other)} record(s) with relation "
              f"{other['standard_relation'].value_counts().to_dict()} are neither exact nor censored; dropped")

    for frame in (core, censored):
        frame["standard_value"] = pd.to_numeric(frame["standard_value"])
        frame["pIC50"] = 9 - np.log10(frame["standard_value"])

    funnel = pd.DataFrame(rows, columns=["step", "records"])
    funnel["kept"] = (funnel["records"] / len(acts) * 100).round(1)
    return core, censored, funnel


def add_structures(df: pd.DataFrame, smiles_col: str) -> pd.DataFrame:
    std = pd.DataFrame([standardize(s) for s in df[smiles_col]], index=df.index)
    return pd.concat([df, std], axis=1)


# --------------------------------------------------------------------------------------
# 5. Main
# --------------------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true",
                    help="skip the downloads and re-derive everything from the cached raw files")
    args = ap.parse_args()

    raw_act = DATA / "chembl_cyp3a4_ic50_raw.csv.gz"
    raw_asy = DATA / "chembl_cyp3a4_assays.csv.gz"
    prov: dict = {"prepared_on": date.today().isoformat()}

    if args.offline:
        if not raw_act.exists():
            raise SystemExit(f"{raw_act} is missing - run once without --offline first")
        acts = pd.read_csv(raw_act, low_memory=False)
        assays = pd.read_csv(raw_asy, low_memory=False)
        print(f"offline: {len(acts)} activity records, {len(assays)} assays from the cache")
        # An offline re-derivation must NOT erase where the data came from. Carry the recorded
        # release, retrieval date and target forward, and say that this run did not re-download.
        existing_path = DATA / "provenance.json"
        if existing_path.exists():
            previous = json.loads(existing_path.read_text())
            for key in ("chembl_release", "chembl_release_date", "target", "downloaded_on"):
                if key in previous:
                    prov[key] = previous[key]
            prov["downloaded_on"] = previous.get("downloaded_on", previous.get("prepared_on"))
            prov["rederived_offline_on"] = prov.pop("prepared_on")
            prov["prepared_on"] = prov["downloaded_on"]
            print(f"  carried provenance forward: ChEMBL {prov.get('chembl_release', '?')}, "
                  f"downloaded {prov.get('downloaded_on', '?')}")
        else:
            print("  WARNING: no previous provenance.json — the release and target cannot be recorded")
    else:
        s = session()
        status = s.get(f"{EBI}/chembl/api/data/status.json", timeout=60).json()
        prov["chembl_release"] = status["chembl_db_version"]
        prov["chembl_release_date"] = status["chembl_release_date"]
        prov["downloaded_on"] = prov["prepared_on"]
        print(f"ChEMBL {status['chembl_db_version']} (released {status['chembl_release_date']})")

        target = verify_target(s)
        prov["target"] = {"chembl_id": TARGET, "uniprot": UNIPROT,
                          "pref_name": target["pref_name"], "organism": target["organism"],
                          "target_type": target["target_type"]}

        print("downloading activities ...")
        acts = download_activities(s)
        print("downloading assay metadata ...")
        assays = download_assays(s, sorted(acts["assay_chembl_id"].dropna().unique()))

        keep_act = ["activity_id", "assay_chembl_id", "assay_description", "assay_type",
                    "molecule_chembl_id", "molecule_pref_name", "canonical_smiles",
                    "standard_type", "standard_relation", "standard_value", "standard_units",
                    "pchembl_value", "data_validity_comment", "potential_duplicate",
                    "activity_comment", "bao_label", "target_chembl_id", "target_organism",
                    "document_chembl_id", "document_journal", "document_year", "src_id"]
        acts[keep_act].to_csv(raw_act, index=False, compression="gzip")
        keep_asy = ["assay_chembl_id", "description", "assay_type", "assay_organism",
                    "confidence_score", "confidence_description", "relationship_type",
                    "bao_label", "assay_tissue", "assay_cell_type", "assay_subcellular_fraction",
                    "document_chembl_id", "src_id", "target_chembl_id"]
        assays[keep_asy].to_csv(raw_asy, index=False, compression="gzip")
        acts, assays = acts[keep_act], assays[keep_asy]

        sample = download_ui_export_sample(s)
        if sample is not None:
            sample.to_csv(DATA / "chembl_ui_export_sample.csv", index=False, sep=";")
            print(f"website export sample: {sample.shape}")
        download_openadmet(s)

    # ---- assay review ----------------------------------------------------------------
    review = review_assays(assays, acts)
    review.to_csv(DATA / "chembl_cyp3a4_assay_review.csv", index=False)
    print(f"assay review: {len(review)} assays, "
          f"{int(review['direct_inhibition'].sum())} direct-inhibition, "
          f"{int(review['time_dependent'].sum())} time-dependent-like")

    # ---- ChEMBL curation -------------------------------------------------------------
    core, censored, funnel = curate_chembl(acts, review)
    print(funnel.to_string(index=False))

    print("standardising ChEMBL structures ...")
    core = add_structures(core, "canonical_smiles")
    censored = add_structures(censored, "canonical_smiles")
    bad = core["smiles_std"].isna().sum()
    core = core[core["smiles_std"].notna()].copy()
    censored = censored[censored["smiles_std"].notna()].copy()
    print(f"  {bad} structures could not be standardised (dropped from the quantitative set)")

    out_cols = ["activity_id", "molecule_chembl_id", "molecule_pref_name", "assay_chembl_id",
                "canonical_smiles", "smiles_std", "inchikey", "standard_type", "standard_relation",
                "standard_value", "standard_units", "pIC50", "pchembl_value", "probe_substrate",
                "enzyme_system", "assay_type", "bao_label", "document_chembl_id", "document_year",
                "n_fragments", "had_charge"]
    core[out_cols].to_csv(DATA / "chembl_cyp3a4_curated.csv", index=False)
    censored[out_cols].to_csv(DATA / "chembl_cyp3a4_censored.csv", index=False)
    funnel.to_csv(DATA / "chembl_cyp3a4_funnel.csv", index=False)

    # ---- the structural reference set -------------------------------------------------
    # Every unique structure ChEMBL has an IC50 record for against CYP3A4, whatever the
    # relation or the assay context: this is "the chemical space ChEMBL covers for this target",
    # and it is the reference set for the nearest-neighbour search.
    allstd = add_structures(acts[acts["canonical_smiles"].notna()].copy(), "canonical_smiles")
    allstd = allstd[allstd["smiles_std"].notna()]
    structures = (allstd.sort_values("activity_id")
                  .groupby("smiles_std")
                  .agg(molecule_chembl_id=("molecule_chembl_id", "first"),
                       molecule_pref_name=("molecule_pref_name", "first"),
                       inchikey=("inchikey", "first"),
                       n_records=("activity_id", "size"))
                  .reset_index())
    desc = pd.DataFrame([describe(s) for s in structures["smiles_std"]], index=structures.index)
    scaf = [murcko(s) for s in structures["smiles_std"]]
    structures["scaffold"] = [x[0] for x in scaf]
    structures["has_ring"] = [x[1] for x in scaf]
    structures = pd.concat([structures, desc], axis=1)
    structures.to_csv(DATA / "chembl_cyp3a4_structures.csv", index=False)
    print(f"ChEMBL structure reference set: {len(structures)} unique standardised structures")

    # ---- OpenADMET --------------------------------------------------------------------
    oa_path = DATA / "openadmet_cyp_train_inhibition.csv"
    oa = pd.read_csv(oa_path)
    assert len(oa) == 4905, f"expected 4905 rows in the pinned snapshot, got {len(oa)}"
    assert oa[CYP].notna().sum() == 2335, f"expected 2335 CYP3A4 values, got {oa[CYP].notna().sum()}"
    print(f"OpenADMET snapshot: {len(oa)} rows, {oa[CYP].notna().sum()} with a CYP3A4 value")

    oa3 = oa[oa[CYP].notna()].copy()
    oa3 = add_structures(oa3, "SMILES")
    bad = oa3["smiles_std"].isna().sum()
    oa3 = oa3[oa3["smiles_std"].notna()].copy()
    print(f"  {bad} OpenADMET structures could not be standardised")
    oa3 = oa3.rename(columns={
        CYP: "pIC50", f"{CYP}_conf_low": "pIC50_conf_low",
        f"{CYP}_conf_high": "pIC50_conf_high", f"{CYP}_std": "pIC50_std"})
    oa3["pIC50_ci_width"] = oa3["pIC50_conf_high"] - oa3["pIC50_conf_low"]
    oa3["weak_potency_flag"] = oa3["pIC50"] < 4          # a caution flag, never a filter
    oa_cols = ["Molecule_Name", "SMILES", "smiles_std", "inchikey", "pIC50", "pIC50_conf_low",
               "pIC50_conf_high", "pIC50_std", "pIC50_ci_width", "weak_potency_flag",
               "n_fragments", "had_charge"]
    desc = pd.DataFrame([describe(s) for s in oa3["smiles_std"]], index=oa3.index)
    scaf = [murcko(s) for s in oa3["smiles_std"]]
    oa3["scaffold"] = [x[0] for x in scaf]
    oa3["has_ring"] = [x[1] for x in scaf]
    pd.concat([oa3[oa_cols + ["scaffold", "has_ring"]], desc], axis=1).to_csv(
        DATA / "openadmet_cyp3a4_curated.csv", index=False)

    # ---- overlap, for the provenance record -------------------------------------------
    overlap = set(structures["smiles_std"]) & set(oa3["smiles_std"])
    print(f"exact structure overlap (standardised isomeric SMILES): {len(overlap)}")

    prov.update({
        "sources": {
            "chembl_activities": {"url": ACTIVITY_URL, "records": int(len(acts)),
                                  "note": "all pages followed via page_meta.next"},
            "chembl_assays": {"url": f"{EBI}/chembl/api/data/assay.json?assay_chembl_id__in=...",
                              "records": int(len(assays))},
            "openadmet": {"url": HF_URL, "revision": HF_REVISION, "sha256": HF_SHA256,
                          "rows": int(len(oa)), "cyp3a4_values": int(oa[CYP].notna().sum()),
                          "licence": "Apache-2.0 (dataset card)"},
        },
        "curation": {
            "chembl_core_records": int(len(core)),
            "chembl_core_structures": int(core["smiles_std"].nunique()),
            "chembl_censored_records": int(len(censored)),
            "chembl_reference_structures": int(len(structures)),
            "openadmet_structures": int(oa3["smiles_std"].nunique()),
            "exact_overlap_structures": int(len(overlap)),
            "standardisation": "Cleanup + LargestFragmentChooser(preferOrganic) + Uncharger; "
                               "isomeric canonical SMILES; no tautomer canonicalisation",
        },
        "files": {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size}
                  for p in sorted(DATA.glob("*")) if p.is_file() and p.name != "provenance.json"},
    })
    (DATA / "provenance.json").write_text(json.dumps(prov, indent=2) + "\n")
    print(f"\nwrote {len(prov['files'])} files to {DATA}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
