"""Verify the workshop notebook end to end, on CPU, before class.

    python scripts/verify_workshop.py           # clean run + interaction run (cached data only)
    python scripts/verify_workshop.py --live    # also download ChEMBL live and push it through the same pipeline
    python scripts/verify_workshop.py --notebook cyp3a4_chembl_vs_openadmet_fr   # the French version

The notebook is run headless with marimo's script runner, exactly as `python notebooks/...py` would, once with the
default widget values ("clean kernel") and once with widget values replaced by fakes that stand for a student's
clicks (a row selected in each table, a brushed region on each map, a file uploaded, other options chosen…).
`app.run(defs=...)` skips the cells that create those widgets, so every downstream cell runs on the fake value.

Checks: dataset counts; target identity; both ChEMBL routes identical (raw records, retained records, retained
activity IDs) and identical downstream; unit conversions; censored and missing values never turned into numbers;
alignment of IDs, structures, descriptors and nearest neighbours (recomputed independently); reproducible sampling
and UMAP. Exit code 0 only if every check passes.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace as Widget

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))
import importlib  # noqa: E402

nb = None  # the notebook module, imported in main() (English by default, --notebook for the French one)

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""), flush=True)


class Upload:
    """Stands in for a `mo.ui.file` widget into which a student dropped a file."""

    def __init__(self, path: Path):
        self.value = [path.name]
        self._data, self._name = path.read_bytes(), path.name

    def contents(self):
        return self._data

    def name(self):
        return self._name


def run(label: str, overrides: dict | None = None) -> dict:
    print(f"\n== {label}", flush=True)
    start = time.time()
    _, defs = nb.app.run(defs=overrides or {})
    print(f"   ran every cell without error in {time.time() - start:.0f} s", flush=True)
    return dict(defs)


def checks_on_clean_run(a: dict) -> None:
    raw = a["openadmet_raw"]
    check("OpenADMET training file has 4,905 rows and 2,335 CYP3A4 values",
          len(raw) == 4905 and raw[nb.CYP3A4].notna().sum() == 2335)
    check("CHEMBL340 verified as the human single protein P08684", (a["target_checks"]["ok"] == "✅").all())

    rc = a["route_comparison"].set_index("route")
    manual = rc.loc["Manual web export · cached snapshot"]
    check("manual export and API: same raw records (composite key)", manual["same raw records as reference"])
    check("manual export and API: same retained records after curation", manual["same retained records"])
    check("manual export and API: same retained activity IDs", manual["same retained activity IDs"])

    m = a["chembl_std"]
    q, c = m[m["subset"] == "quantitative"], m[m["subset"] == "censored"]
    check("every quantitative value is an '=' IC50 in nM", (q["standard_relation"] == "=").all()
          and (q["standard_units"] == "nM").all() and (q["standard_type"] == "IC50").all())
    diff = (q["pIC50"] - q["pchembl_value"]).abs().max()
    check("pIC50 = 9 − log10(IC50/nM) agrees with ChEMBL's pchembl_value", diff < 0.01, f"max |Δ| = {diff:.4f}")
    api = a["api_cached"]["activities"]
    um = api[(api["units"] == "uM") & api["standard_units"].eq("nM") & api["value"].notna() & api["standard_value"].notna()]
    rel = (pd.to_numeric(um["standard_value"]) / (1000 * pd.to_numeric(um["value"])) - 1).abs().max()
    check("µM → nM conversion in ChEMBL's standard values", rel < 1e-6, f"{len(um):,} µM records, max rel. error {rel:.1e}")
    check("censored records keep a bound, never a pIC50", c["pIC50"].isna().all() and c["pIC50_bound"].notna().all(),
          f"{len(c):,} censored")
    check("only verified, confidence-9, IC50 records are retained",
          ((m["subset"] == "excluded") | ((m["confidence_score"] == 9) & (m["target_chembl_id"] == nb.TARGET_ID))).all())
    check("retained records come only from direct-inhibition assays", (m.loc[m["subset"] != "excluded", "decision"] == "include").all())

    oa = a["oa_meas"]
    check("OpenADMET: missing CYP3A4 values are dropped, not filled", len(oa) == 2335 and oa["pIC50"].notna().all())
    check("OpenADMET: pIC50 < 4 flagged but neither clipped nor removed",
          oa["weak_flag"].sum() == (raw[nb.CYP3A4] < 4).sum() and oa["pIC50"].min() == raw[nb.CYP3A4].min(),
          f"{int(oa['weak_flag'].sum())} flagged")
    check("OpenADMET uncertainty columns preserved", oa[["pIC50_ci_low", "pIC50_ci_high", "pIC50_std"]].notna().all().all())

    st, f = a["structures"], a["features"]
    check("one row per unique standardised structure", st["smiles_std"].is_unique and len(st) == len(f))
    check("descriptor table aligned with the structure table", (st["smiles_std"].to_numpy() == f["smiles_std"].to_numpy()).all())
    rng = np.random.default_rng(0)
    rows = rng.choice(len(f), size=25, replace=False)
    redo = pd.DataFrame([nb.descriptor_row(Chem.MolFromSmiles(s)) for s in f["smiles_std"].iloc[rows]])
    check("descriptors recomputed independently for 25 random rows", np.allclose(redo.to_numpy(), f[nb.DESCRIPTORS].iloc[rows].to_numpy()))
    both = set(st.loc[st["membership"] == "both", "smiles_std"])
    in_oa = set(a["oa_std"]["smiles_std"].dropna())
    in_ch = set(a["chembl_std"].loc[a["chembl_std"]["subset"] != "excluded", "smiles_std"].dropna())
    check("'both' membership = exact standardised identity in both sources", both == (in_oa & in_ch), f"{len(both)} structures")

    nn, refs = a["nn_table"], a["sim_refs"]
    mask = refs["ref_curated"].to_numpy()
    ref_smiles = refs["smiles_std"].to_numpy()
    ref_fps = [nb.FP_GEN.GetFingerprint(Chem.MolFromSmiles(s)) for s in ref_smiles[mask]]
    ok = True
    for i in rng.choice(len(nn), size=25, replace=False):
        row = nn.iloc[i]
        sims = np.array(DataStructs.BulkTanimotoSimilarity(nb.FP_GEN.GetFingerprint(Chem.MolFromSmiles(row["smiles_std"])), ref_fps))
        sims[ref_smiles[mask] == row["smiles_std"]] = -1            # self-match excluded
        ok &= abs(sims.max() - row["nn_tanimoto"]) < 1e-3 and row["nn_smiles"] != row["smiles_std"]
        ok &= bool(row["exact_in_chembl"]) == (row["smiles_std"] in set(ref_smiles[mask]))
    check("nearest neighbours recomputed independently for 25 random compounds", ok,
          f"reference set {int(mask.sum()):,} structures")

    s = a["map_sample"]
    redo = pd.concat([f[f["membership"].isin([n, "both"])].pipe(lambda d: d.sample(n=min(nb.MAX_PER_SOURCE, len(d)), random_state=nb.SEED))
                      for n in ("ChEMBL", "OpenADMET")]).drop_duplicates("smiles_std")
    check("map sample: ≤ 1,000 per source, seed 42, reproducible", (redo["smiles_std"].to_numpy() == s["smiles_std"].to_numpy()).all()
          and s["membership"].isin(["ChEMBL", "both"]).sum() <= nb.MAX_PER_SOURCE + s["membership"].eq("both").sum(),
          f"{len(s):,} structures")


def interaction_overrides(a: dict) -> dict:
    nn_view = a["nn_table"].drop(columns=["query_row"]).sort_values("nn_tanimoto").reset_index(drop=True)
    changed = a["chembl_std"][a["chembl_std"]["std_status"] == "changed"]
    return {
        "ui_activity_upload": Upload(ROOT / nb.CACHED["ui_activities"]),
        "ui_assay_upload": Upload(ROOT / nb.CACHED["ui_assays"]),
        "chembl_route": Widget(value="Manual web export · your upload"),
        "ic50_value": Widget(value=250.0), "ic50_unit": Widget(value=1.0),
        "drug_picker": Widget(value="erythromycin"),
        "exercise1_q1": Widget(value="b"), "exercise1_q2": Widget(value="a"),
        "oa_uncertainty_chart": Widget(value=a["oa_meas"].head(30).assign(flag="pIC50 ≥ 4")),
        "fig_uncertainty": a["fig_uncertainty"], "fig_pca": a["fig_pca"], "fig_umap": a["fig_umap"],  # same cells
        "std_examples_table": Widget(value=changed[["smiles", "smiles_std", "std_changes"]].head(1)),
        "whatif_rules": Widget(value=[k for k in nb.CURATION_RULES if k not in ("assay_context", "confidence")]),
        "assay_view_filter": Widget(value="instructor-reviewed"),
        "descriptor_picker": Widget(value="HBD"), "descriptor_bins": Widget(value=20),
        "corr_scope": Widget(value="difference (OpenADMET − ChEMBL)"),
        "pic50_group_by": Widget(value="enzyme_source"),
        "pair_picker": Widget(value="testosterone vs epitestosterone (T = 1, not identical!)"),
        "pair_presets": a["pair_presets"],
        "reference_choice": Widget(value="ref_any"), "coverage_threshold": Widget(value=0.5),
        "nn_browser": Widget(value=nn_view.iloc[[0]]),
        "test_set_button": Widget(value=True),
        "top_n_scaffolds": Widget(value=12),
        "scaffold_status_filter": Widget(value="shared"),
        "scaffold_browser": Widget(value=a["scaffold_table"].iloc[[0, 3]]),
        "map_color": Widget(value="scaffold (your selection in part 6, else the 8 most common)"),
        "pca_chart": Widget(value=a["map_sample_pca"].head(15)),
        "umap_params": Widget(value={"n_neighbors": 15, "min_dist": 0.1}),
        "umap_chart": Widget(value=a["map_sample_umap"].tail(15)),
        "outlier_scope": Widget(value="all"), "outlier_top_k": Widget(value=8),
        "unusual_percent": Widget(value=5.0), "novelty_threshold": Widget(value=0.25),
        "export_button": Widget(value=False),
    }


def checks_on_interaction_run(a: dict, b: dict) -> None:
    rc = b["route_comparison"].set_index("route")
    up = rc.loc["Manual web export · your upload"]
    check("uploaded web export goes through the same normalisation", up[["same raw records as reference", "same retained records",
                                                                      "same retained activity IDs"]].all())
    cols = ["smiles_std", "membership", "chembl_median_pIC50", "openadmet_pIC50"]
    check("downstream results identical whichever ChEMBL route is used",
          a["structures"][cols].equals(b["structures"][cols]) and a["features"][nb.DESCRIPTORS].equals(b["features"][nb.DESCRIPTORS]))
    check("UMAP is reproducible (same sample, same parameters, seed 42)",
          np.allclose(a["map_sample_umap"][["UMAP1", "UMAP2"]].to_numpy(), b["map_sample_umap"][["UMAP1", "UMAP2"]].to_numpy()))
    check("loosest reference set contains the default one", b["nn_reference_size"] >= len(a["sim_refs"][a["sim_refs"]["ref_curated"]]))
    for name, fn in [("numeric colouring", "OpenADMET pIC50"), ("ChEMBL colouring", "ChEMBL median pIC50 (quantitative records)"),
                     ("weak-flag colouring", "OpenADMET pIC50 < 4 (cautious)"), ("source colouring", "source")]:
        d = nb.map_frame(a["map_sample_pca"], a["map_sample_pca"][["PC1", "PC2"]].to_numpy(), fn, [], [])
        nb.map_chart(d, "x", "y", "t").to_dict()
        check(f"map chart builds with {name}", len(d) == len(a["map_sample_pca"]))


def checks_on_live_run(a: dict, c: dict) -> None:
    live = c["live_api"]
    check("live ChEMBL download completed", live is not None)
    if live is None:
        return
    same_release = live["release"] == a["api_cached"]["release"]
    row = c["route_comparison"].set_index("route").loc["API · live download (this session)"]
    detail = f"live {live['release']} vs cached {a['api_cached']['release']}"
    if same_release:
        check("live API = cached snapshot (same release)", row[["same raw records as reference", "same retained records"]].all(), detail)
    else:
        print(f"  NOTE  new ChEMBL release ({detail}): differences are expected; refresh the cache with prepare_data.py")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--live", action="store_true", help="also test a live ChEMBL download (≈ 2 min, needs network)")
    parser.add_argument("--notebook", default="cyp3a4_chembl_vs_openadmet", help="notebook module in notebooks/")
    args = parser.parse_args()
    global nb
    nb = importlib.import_module(args.notebook)
    print(f"notebook: notebooks/{args.notebook}.py")

    a = run("clean run, default widget values")
    checks_on_clean_run(a)
    b = run("interaction run: selections, upload, other options", interaction_overrides(a))
    checks_on_interaction_run(a, b)
    if args.live:
        c = run("live run: ChEMBL API downloaded now", {"api_fetch_button": Widget(value=True),
                                                         "chembl_route": Widget(value="API · live download (this session)")})
        checks_on_live_run(a, c)

    failed = [r for r in RESULTS if not r[1]]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
