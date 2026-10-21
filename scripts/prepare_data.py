"""Prepare the cached teaching data for the CYP3A4 workshop (run before class, not during it).

    python scripts/prepare_data.py            # download everything again and rewrite data/ + provenance
    python scripts/prepare_data.py --check    # only verify the cached files against data/provenance.json

What it does
------------
1. OpenADMET: downloads the pinned training and blinded-test CSVs and checks the expected row counts.
2. ChEMBL API: release status, the CHEMBL340 target record, every IC50 activity page and the assay metadata
   (batches of 100) - with the very functions the notebook uses (imported from it, so there is one implementation).
3. ChEMBL web export: submits the same export jobs as the CSV button of the ChEMBL website (activities filtered to
   IC50, and the assays of CHEMBL340), waits for them and saves the zip files untouched.
4. Assay review: classifies every confidence-9 assay with the notebook's keyword rules and merges the reviewed
   decisions already present in data/chembl/assay_review.csv (they are never overwritten).
5. Writes data/provenance.json: release, retrieval date, queries, URLs, record counts and SHA-256 of every file.

Every file is written next to the notebook's expectations (see CACHED in the notebook's setup cell).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))
import cyp3a4_chembl_vs_openadmet as nb  # noqa: E402  (the workshop notebook, imported as a module)

DELAYED_JOBS = "https://www.ebi.ac.uk/chembl/interface_api/delayed_jobs"
WEB_EXPORTS = {
    "ui_activities": ("chembl_activity", 'target_chembl_id:CHEMBL340 AND standard_type:"IC50"'),
    "ui_assays": ("chembl_assay", "target_chembl_id:CHEMBL340"),
}

# Assays whose descriptions were read by hand before class. Keys: assay ChEMBL ID -> (decision, reason, note).
# Keep this list in sync with what the instructor actually checked; prepare_data.py never overwrites reviewed rows
# already in assay_review.csv, so edits made directly in the CSV win.
REVIEWED = {
    "CHEMBL3706081": ("include", "direct inhibition", "BFC, cDNA-expressed CYP3A4 supersomes; NADPH added with the substrate"),
    "CHEMBL5732086": ("include", "direct inhibition", "patent CYP panel; CYP3A4 measured with BFC and BQ under one assay ID (probe not resolvable)"),
    "CHEMBL5733771": ("include", "direct inhibition", "BFC, recombinant enzyme (yeast expression)"),
    "CHEMBL4422403": ("include", "direct inhibition", "HLM, testosterone 6-hydroxylation"),
    "CHEMBL3707706": ("include", "direct inhibition", "patent CYP panel; BFC and BQ under one assay ID (probe not resolvable)"),
    "CHEMBL3880170": ("include", "direct inhibition", "BQ, 30 min; enzyme origin not stated"),
    "CHEMBL3880169": ("include", "direct inhibition", "BFC, 30 min; enzyme origin not stated"),
    "CHEMBL2448771": ("include", "direct inhibition", "DNDi, testosterone 6beta-hydroxylation"),
    "CHEMBL2060578": ("include", "direct inhibition", "HLM, midazolam, 10 min"),
    "CHEMBL4059171": ("include", "direct inhibition", "no protocol details; enzyme origin not stated"),
    "CHEMBL5161151": ("include", "direct inhibition", "incubation (not pre-incubation) with NADPH"),
    "CHEMBL5530621": ("include", "direct inhibition", "midazolam with NADPH-regenerating system"),
    "CHEMBL4046491": ("include", "direct inhibition", "recombinant CYP3A4 from insect cells (expression system, not a cell assay); BFC"),
    "CHEMBL4349925": ("include", "direct inhibition", "HLM, midazolam; 5-min pre-incubation *before* NADPH addition"),
    "CHEMBL4349926": ("include", "direct inhibition", "HLM, testosterone; 5-min pre-incubation before NADPH addition"),
    "CHEMBL4671512": ("include", "direct inhibition", "HLM, midazolam"),
    "CHEMBL4388242": ("include", "direct inhibition", "recombinant, luciferin; pre-incubation without NADPH (direct arm)"),
    "CHEMBL3705120": ("include", "direct inhibition", "recombinant, erythromycin N-demethylation"),
    "CHEMBL1047872": ("include", "direct inhibition", "no protocol details"),
    "CHEMBL1953081": ("include", "direct inhibition", "supersomes, midazolam"),
    "CHEMBL4413476": ("include", "direct inhibition", "recombinant, BFC; pre-incubation before substrate, NADPH timing not stated"),
    "CHEMBL1100026": ("include", "direct inhibition", "recombinant, 7-benzyloxyquinoline"),
    "CHEMBL3374295": ("include", "direct inhibition", "recombinant, BFC"),
    "CHEMBL4150908": ("include", "direct inhibition", "HLM, midazolam; pre-incubation before NADPH addition"),
    "CHEMBL5366037": ("include", "direct inhibition", "HLM, probe not stated"),
    "CHEMBL4009748": ("include", "direct inhibition", "dealkylation probe not named"),
    "CHEMBL5579615": ("include", "direct inhibition", "HLM, sorafenib as probe; NADPH added with the substrate"),
    "CHEMBL2437635": ("include", "direct inhibition", "no protocol details"),
    "CHEMBL5525096": ("exclude", "ambiguous: whole-cell or in vivo context", "human hepatocytes (whole cells)"),
    "CHEMBL3423902": ("include", "direct inhibition", "BFC; compound pre-incubated before enzyme/substrate addition"),
    "CHEMBL4156816": ("include", "direct inhibition", "no protocol details"),
    "CHEMBL4481757": ("include", "direct inhibition", "no protocol details; enzyme origin not stated"),
    "CHEMBL2444436": ("include", "direct inhibition", "BFC; enzyme origin not stated"),
    "CHEMBL5525095": ("include", "direct inhibition", "HLM, luciferin-IPA; pre-incubation before NADPH addition"),
    "CHEMBL3706297": ("include", "direct inhibition", "HLM, midazolam and testosterone (several probes)"),
    "CHEMBL4367264": ("include", "direct inhibition", "midazolam; enzyme origin not stated"),
    "CHEMBL3117563": ("include", "direct inhibition", "midazolam 1'-hydroxylation"),
    "CHEMBL3118928": ("include", "direct inhibition", "midazolam 1'-hydroxylation"),
    "CHEMBL2020724": ("include", "direct inhibition", "HLM, midazolam, 5-min pre-incubation (NADPH timing not stated)"),
    "CHEMBL4195548": ("include", "direct inhibition", "HLM, midazolam"),
    # examples of exclusions checked by hand
    "CHEMBL3404695": ("exclude", "time-dependent inhibition or NADPH pre-incubation", "explicit TDI: 30-min pre-incubation with NADPH"),
    "CHEMBL5367488": ("exclude", "time-dependent inhibition or NADPH pre-incubation", "30-min pre-incubation in presence of NADPH"),
    "CHEMBL5135144": ("exclude", "time-dependent inhibition or NADPH pre-incubation", "pretreated with NADPH; its direct twin is CHEMBL5135143"),
    "CHEMBL1038133": ("exclude", "time-dependent inhibition or NADPH pre-incubation", "HLM pre-incubated with NADPH"),
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(url: str, path: Path) -> bytes:
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(response.content)
    return response.content


def web_export(index_name: str, query_string: str, path: Path) -> None:
    """Run the ChEMBL website's own CSV export (a server-side job) and save the zip it produces."""
    es_query = {"bool": {"must": [{"query_string": {"query": query_string}}], "filter": [], "should": [], "must_not": []}}
    submit = requests.post(
        f"{DELAYED_JOBS}/submit/download_job",
        data={"index_name": index_name, "query": json.dumps(es_query), "format": "CSV",
              "download_columns_group": "download", "dl__ignore_cache": "false"},
        timeout=60,
    )
    submit.raise_for_status()
    job_id = submit.json()["job_id"]
    for _ in range(120):
        status = requests.get(f"{DELAYED_JOBS}/status/{job_id}", timeout=60).json()
        if status["status"] == "FINISHED":
            url = next(iter(status["output_files_urls"].values()))
            download("https://" + url if not url.startswith("http") else url, path)
            return
        if status["status"] == "ERROR":
            raise RuntimeError(f"ChEMBL export job {job_id} failed: {status.get('status_description')}")
        time.sleep(5)
    raise TimeoutError(f"ChEMBL export job {job_id} did not finish in 10 minutes")


def build_assay_review(assays: pd.DataFrame, activities: pd.DataFrame, existing: Path) -> pd.DataFrame:
    meta = nb.assay_meta_from_api(assays)
    classified = nb.classify_assays(meta)
    counts = activities.groupby("assay_chembl_id").size().rename("n_ic50_records")
    review = classified[classified["confidence_score"] == 9].join(counts, on="assay_chembl_id")
    review = review[["assay_chembl_id", "n_ic50_records", "rule_decision", "rule_reason", "enzyme_source",
                     "probe_substrate", "description"]].sort_values("n_ic50_records", ascending=False)
    review["reviewed"] = False
    review["review_decision"] = ""
    review["review_reason"] = ""
    review["review_note"] = ""
    review = review.set_index("assay_chembl_id")
    for assay_id, (decision, reason, note) in REVIEWED.items():
        if assay_id in review.index:
            review.loc[assay_id, ["reviewed", "review_decision", "review_reason", "review_note"]] = [True, decision, reason, note]
    if existing.is_file():                                     # hand edits in the CSV always win
        old = pd.read_csv(existing).set_index("assay_chembl_id")
        old = old[old["reviewed"].astype(str).str.lower() == "true"]
        keep = old.index.intersection(review.index)
        review.loc[keep, ["reviewed", "review_decision", "review_reason", "review_note"]] = old.loc[
            keep, ["reviewed", "review_decision", "review_reason", "review_note"]].to_numpy()
    review["description"] = review["description"].str.slice(0, 300)
    return review.reset_index()


def check() -> int:
    provenance = json.loads((ROOT / nb.CACHED["provenance"]).read_text())
    bad = 0
    for key, info in provenance["files"].items():
        path = ROOT / info["path"]
        ok = path.is_file() and sha256_file(path) == info["sha256"]
        bad += not ok
        print(f"{'ok ' if ok else 'BAD'} {info['path']}")
    return 1 if bad else 0


def prepare() -> int:
    paths = {key: ROOT / rel for key, rel in nb.CACHED.items()}
    retrieved = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    print("1/5 OpenADMET (pinned revision)")
    train = download(nb.OPENADMET_TRAIN_URL, paths["openadmet_train"])
    download(nb.OPENADMET_TEST_URL, paths["openadmet_test"])
    oa = pd.read_csv(io.BytesIO(train))
    assert len(oa) == nb.OPENADMET_EXPECTED["rows"], len(oa)
    assert oa[nb.CYP3A4].notna().sum() == nb.OPENADMET_EXPECTED["cyp3a4_values"]
    test = pd.read_csv(paths["openadmet_test"])

    print("2/5 ChEMBL API")
    status = nb.chembl_get(f"{nb.CHEMBL_API}/status.json")
    target = nb.chembl_get(f"{nb.CHEMBL_API}/target/{nb.TARGET_ID}.json")
    assert (nb.verify_target(target)["ok"] == "✅").all(), "CHEMBL340 is not what we expect"
    paths["api_target"].parent.mkdir(parents=True, exist_ok=True)
    paths["api_target"].write_text(json.dumps(target, indent=1, sort_keys=True))
    records = nb.fetch_activity_pages(on_page=lambda done, total: print(f"   activities {done:>6}/{total}"))
    assays = nb.fetch_assays([r["assay_chembl_id"] for r in records],
                             on_batch=lambda done, total: print(f"   assays {done:>5}/{total}") if done % 500 < 100 else None)
    nb.write_jsonl_gz(records, paths["api_activities"])
    nb.write_jsonl_gz(assays, paths["api_assays"])

    print("3/5 ChEMBL web exports (same jobs as the website's CSV button)")
    for key, (index_name, query_string) in WEB_EXPORTS.items():
        web_export(index_name, query_string, paths[key])
        print(f"   {paths[key].name}: {paths[key].stat().st_size:,} bytes")
    ui_acts, _ = nb.read_ui_export(paths["ui_activities"].read_bytes())
    ui_assays, _ = nb.read_ui_export(paths["ui_assays"].read_bytes())

    print("4/5 assay review")
    review = build_assay_review(pd.DataFrame(assays), pd.DataFrame(records), paths["assay_review"])
    review.to_csv(paths["assay_review"], index=False)
    print(f"   {len(review):,} confidence-9 assays, {int(review['reviewed'].astype(str).str.lower().eq('true').sum())} reviewed by hand")

    print("5/5 provenance")
    counts = {"openadmet_train": len(oa), "openadmet_test": len(test), "api_activities": len(records),
              "api_assays": len(assays), "api_target": 1, "ui_activities": len(ui_acts), "ui_assays": len(ui_assays),
              "assay_review": len(review)}
    provenance = {
        "prepared_utc": retrieved,
        "chembl": {
            "release": status["chembl_db_version"],
            "release_date": status["chembl_release_date"],
            "retrieved_utc": retrieved,
            "target": nb.TARGET_ID,
            "uniprot": nb.UNIPROT_ID,
            "activity_query_url": nb.ACTIVITY_URL,
            "assay_query": f"{nb.CHEMBL_API}/assay.json?assay_chembl_id__in=<100 IDs>&limit=1000",
            "web_export_activity_query": WEB_EXPORTS["ui_activities"][1],
            "web_export_assay_query": WEB_EXPORTS["ui_assays"][1],
            "web_export_pages": {"activities": nb.UI_ACTIVITIES_URL, "assays": nb.UI_ASSAYS_URL},
            "n_activity_records": len(records),
            "n_assays": len(assays),
            "licence": "CC BY-SA 3.0",
        },
        "openadmet": {
            "dataset": nb.HF_DATASET,
            "config": "default",
            "split": "train",
            "revision": nb.HF_REVISION,
            "train_url": nb.OPENADMET_TRAIN_URL,
            "test_url": nb.OPENADMET_TEST_URL,
            "rows": len(oa),
            "cyp3a4_values": int(oa[nb.CYP3A4].notna().sum()),
            "licence": "Apache-2.0 (dataset card)",
        },
        "files": {
            key: {"path": nb.CACHED[key], "sha256": sha256_file(paths[key]), "records": counts.get(key)}
            for key in ("openadmet_train", "openadmet_test", "api_activities", "api_assays", "api_target",
                        "ui_activities", "ui_assays", "assay_review")
        },
    }
    paths["provenance"].write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance["chembl"], indent=1))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="only verify cached files against provenance.json")
    args = parser.parse_args()
    sys.exit(check() if args.check else prepare())
