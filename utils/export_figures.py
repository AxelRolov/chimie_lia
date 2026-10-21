"""Export the workshop's headline figures and summary tables to `figures/`.

For instructors who want the key results on slides without running the notebook:

    python utils/export_figures.py

Reads only the cached snapshot in `data/` — no network access. Everything it produces is
also produced inside the notebook; this script just writes it to disk at 150 dpi.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Draw, rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")
sns.set_theme(style="whitegrid", palette="deep")

ROOT = Path(__file__).resolve().parent.parent
DATA, FIG = ROOT / "data", ROOT / "figures"
FIG.mkdir(exist_ok=True)

SRC_COLOR = {"ChEMBL": "#4C72B0", "OpenADMET": "#DD8452", "both": "#55A868"}
DESC_COLS = ["MW", "cLogP", "TPSA", "HBD", "HBA", "RotB", "AromRings", "FracCSP3", "HeavyAtoms"]
SEED = 42


def save(fig, name):
    path = FIG / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path.relative_to(ROOT)}")


def main():
    prov = json.loads((DATA / "provenance.json").read_text())
    ch = pd.read_csv(DATA / "chembl_cyp3a4_structures.csv")
    oa = pd.read_csv(DATA / "openadmet_cyp3a4_curated.csv")
    quant = pd.read_csv(DATA / "chembl_cyp3a4_curated.csv")
    censored = pd.read_csv(DATA / "chembl_cyp3a4_censored.csv")
    funnel = pd.read_csv(DATA / "chembl_cyp3a4_funnel.csv")
    print(f"ChEMBL {prov.get('chembl_release')} · {len(ch):,} structures · {len(oa):,} OpenADMET compounds\n")

    print("figures:")

    # 1 ─ curation funnel ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.barh(funnel["step"], funnel["records"], color=SRC_COLOR["ChEMBL"])
    ax.invert_yaxis(); ax.set_xlabel("records surviving")
    for y, n in enumerate(funnel["records"]):
        ax.text(n + 150, y, f"{n:,}", va="center", fontsize=9)
    ax.set_title("ChEMBL CYP3A4 curation funnel")
    save(fig, "01_curation_funnel.png")

    # 2 ─ descriptor distributions ------------------------------------------------------
    fig, axes = plt.subplots(3, 3, figsize=(13, 9))
    for ax, col in zip(axes.ravel(), DESC_COLS):
        a, b = ch[col].dropna(), oa[col].dropna()
        lo, hi = np.nanpercentile(np.concatenate([a, b]), [0.5, 99.5])
        bins = np.linspace(lo, hi, 40)
        ax.hist(a, bins=bins, alpha=.62, density=True, color=SRC_COLOR["ChEMBL"], label="ChEMBL")
        ax.hist(b, bins=bins, alpha=.62, density=True, color=SRC_COLOR["OpenADMET"], label="OpenADMET")
        ax.set_title(col, fontsize=10)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Descriptor distributions by source", y=1.0)
    save(fig, "02_descriptor_distributions.png")

    # 3 ─ pIC50 distributions, never pooled ---------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 3.6))
    bins = np.linspace(1.5, 10.5, 50)
    ax.hist(quant["pIC50"], bins=bins, alpha=.7, density=True, color=SRC_COLOR["ChEMBL"],
            label=f"ChEMBL exact (n={len(quant):,})")
    ax.hist(censored["pIC50"], bins=bins, alpha=.5, density=True, color="#b0b0b0",
            label=f"ChEMBL censored (n={len(censored):,}) — NOT measurements")
    ax.hist(oa["pIC50"], bins=bins, alpha=.7, density=True, color=SRC_COLOR["OpenADMET"],
            label=f"OpenADMET (n={len(oa):,})")
    ax.axvline(4, color="crimson", ls=":", lw=2)
    ax.set_xlabel("pIC50"); ax.set_ylabel("density"); ax.legend(fontsize=8)
    ax.set_title("Activity distributions — labelled by source and kind, never pooled")
    save(fig, "03_pic50_distributions.png")

    # 4 ─ uncertainty vs potency --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    weak = oa["pIC50"] < 4
    ax.scatter(oa.loc[~weak, "pIC50"], oa.loc[~weak, "pIC50_ci_width"], s=6, alpha=.6,
               c="#55A868", label="pIC50 ≥ 4")
    ax.scatter(oa.loc[weak, "pIC50"], oa.loc[weak, "pIC50_ci_width"], s=6, alpha=.6,
               c="#C44E52", label=f"pIC50 < 4  (n={int(weak.sum()):,})")
    ax.set_xlabel("OpenADMET pIC50"); ax.set_ylabel("width of the 95 % credible interval")
    ax.legend(); ax.set_title("Weak compounds have imprecise pIC50s")
    save(fig, "04_uncertainty_vs_potency.png")

    # 5 ─ nearest-neighbour similarity --------------------------------------------------
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    ch_fps = [fpgen.GetFingerprint(Chem.MolFromSmiles(s)) for s in ch["smiles_std"]]
    oa_fps = [fpgen.GetFingerprint(Chem.MolFromSmiles(s)) for s in oa["smiles_std"]]
    ch_index = {s: i for i, s in enumerate(ch["smiles_std"])}
    max_sim, nn_idx = np.zeros(len(oa_fps)), np.zeros(len(oa_fps), dtype=int)
    for k, (q, smi) in enumerate(zip(oa_fps, oa["smiles_std"])):
        sims = np.array(DataStructs.BulkTanimotoSimilarity(q, ch_fps))
        if smi in ch_index:
            sims[ch_index[smi]] = -1
        j = int(sims.argmax()); max_sim[k], nn_idx[k] = sims[j], j
    oa["max_tanimoto_to_chembl"] = max_sim

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
    axes[0].hist(max_sim, bins=60, color=SRC_COLOR["OpenADMET"])
    axes[0].axvline(np.median(max_sim), c="k", ls="--", label=f"median {np.median(max_sim):.2f}")
    axes[0].axvline(0.4, c="crimson", ls=":", label="T = 0.4")
    axes[0].set_xlabel(f"max Tanimoto to the ChEMBL reference set (n = {len(ch):,})")
    axes[0].set_ylabel("OpenADMET compounds"); axes[0].legend()
    thr = np.linspace(0, 1, 101)
    axes[1].plot(thr, [(max_sim >= t).mean() for t in thr], lw=2, color=SRC_COLOR["OpenADMET"])
    axes[1].set_xlabel("Tanimoto threshold"); axes[1].set_ylabel("fraction covered")
    axes[1].grid(True, alpha=.3); axes[1].set_title("Coverage curve")
    fig.suptitle("How close is the nearest ChEMBL molecule?", y=1.02)
    save(fig, "05_nearest_neighbour_similarity.png")

    # 6 ─ worst-covered molecules -------------------------------------------------------
    order = np.argsort(max_sim)
    worst = oa.iloc[order[:8]]
    # useSVG=False makes MolsToGridImage return a PIL image, which we can save directly
    img = Draw.MolsToGridImage([Chem.MolFromSmiles(s) for s in worst["smiles_std"]], molsPerRow=4,
                               subImgSize=(260, 200), useSVG=False, returnPNG=False,
                               legends=[f"{n}  T={t:.2f}" for n, t in
                                        zip(worst["Molecule_Name"], worst["max_tanimoto_to_chembl"])])
    img.save(FIG / "06_worst_covered_compounds.png")
    print(f"  figures/06_worst_covered_compounds.png")

    # 7 ─ PCA / UMAP joint map ----------------------------------------------------------
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.ensemble import IsolationForest
    import umap

    both = set(ch["smiles_std"]) & set(oa["smiles_std"])
    cs = ch.sample(min(1000, len(ch)), random_state=SEED)
    os_ = oa.sample(min(1000, len(oa)), random_state=SEED)
    samp = (pd.concat([cs[["smiles_std"] + DESC_COLS].assign(source="ChEMBL"),
                       os_[["smiles_std"] + DESC_COLS].assign(source="OpenADMET")])
            .drop_duplicates("smiles_std").reset_index(drop=True))
    samp["membership"] = np.where(samp["smiles_std"].isin(both), "both", samp["source"])

    X = StandardScaler().fit_transform(samp[DESC_COLS])
    pca = PCA(n_components=2).fit(X); pcs = pca.transform(X)
    Xfp = np.array([fpgen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in samp["smiles_std"]], dtype=bool)
    emb = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="jaccard", random_state=SEED).fit_transform(Xfp)
    samp["outlier_score"] = -IsolationForest(n_estimators=300, random_state=SEED).fit(X).score_samples(X)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for src in ["ChEMBL", "OpenADMET", "both"]:
        m = samp["membership"] == src
        axes[0].scatter(pcs[m, 0], pcs[m, 1], s=10, alpha=.6, c=SRC_COLOR[src], label=f"{src} ({m.sum()})")
        axes[1].scatter(emb[m, 0], emb[m, 1], s=10, alpha=.6, c=SRC_COLOR[src], label=f"{src} ({m.sum()})")
    evr = pca.explained_variance_ratio_
    axes[0].set_xlabel(f"PC1 ({evr[0]:.0%})"); axes[0].set_ylabel(f"PC2 ({evr[1]:.0%})")
    axes[0].set_title("PCA of 9 descriptors"); axes[0].legend(markerscale=2)
    axes[1].set_xticks([]); axes[1].set_yticks([])
    axes[1].set_title("Joint UMAP of ECFP4 (Jaccard)"); axes[1].legend(markerscale=2)
    save(fig, "07_chemical_space.png")

    # 8 ─ PCA loadings ------------------------------------------------------------------
    loadings = pd.DataFrame(pca.components_.T, index=DESC_COLS, columns=["PC1", "PC2"])
    fig, ax = plt.subplots(figsize=(6, 4))
    loadings.plot.barh(ax=ax); ax.axvline(0, c="k", lw=.8)
    ax.set_title("PCA loadings: PC1 = size, PC2 = lipophilicity vs saturation")
    save(fig, "08_pca_loadings.png")

    # ── tables -------------------------------------------------------------------------
    print("\ntables:")
    scaf_ch = ch[ch["has_ring"].astype(bool)]
    scaf_oa = oa[oa["has_ring"].astype(bool)]
    S_ch, S_oa = set(scaf_ch["scaffold"]), set(scaf_oa["scaffold"])
    novel = ~scaf_oa["scaffold"].isin(S_ch)
    reps = [scaf_ch.sample(len(scaf_oa), random_state=SEED + i)["scaffold"].nunique() for i in range(20)]

    summary = pd.DataFrame([
        ("ChEMBL IC50 records (raw)", prov["sources"]["chembl_activities"]["records"]),
        ("ChEMBL records after curation (exact, direct inhibition, nM)", len(quant)),
        ("ChEMBL censored records kept separately", len(censored)),
        ("ChEMBL unique structures (similarity reference set)", len(ch)),
        ("OpenADMET rows in the pinned snapshot", prov["sources"]["openadmet"]["rows"]),
        ("OpenADMET compounds with a CYP3A4 value", len(oa)),
        ("exact structure overlap", len(both)),
        ("median max Tanimoto to ChEMBL", round(float(np.median(max_sim)), 3)),
        ("OpenADMET compounds with no ChEMBL neighbour at T >= 0.4", int((max_sim < 0.4).sum())),
        ("ChEMBL distinct Murcko scaffolds", len(S_ch)),
        ("OpenADMET distinct Murcko scaffolds", len(S_oa)),
        ("shared scaffolds", len(S_ch & S_oa)),
        ("OpenADMET ring compounds with a scaffold absent from ChEMBL", int(novel.sum())),
        (f"scaffolds in a ChEMBL sample of matched size ({len(scaf_oa)})", round(float(np.mean(reps)))),
        ("PCA explained variance, PC1 + PC2 (%)", round(float(evr.sum() * 100), 1)),
    ], columns=["quantity", "value"])
    summary.to_csv(FIG / "summary_numbers.csv", index=False)
    print(f"  figures/summary_numbers.csv")
    print(summary.to_string(index=False))

    ch.groupby(lambda _: "ChEMBL")[DESC_COLS].median().to_csv(FIG / "descriptor_medians.csv", mode="w")
    pd.concat([ch[DESC_COLS].median().rename("ChEMBL"),
               oa[DESC_COLS].median().rename("OpenADMET")], axis=1).to_csv(FIG / "descriptor_medians.csv")
    print(f"  figures/descriptor_medians.csv")

    (oa.nsmallest(25, "max_tanimoto_to_chembl")
     [["Molecule_Name", "smiles_std", "max_tanimoto_to_chembl", "pIC50", "pIC50_ci_width",
       "MW", "cLogP", "TPSA"]]
     .round(3).to_csv(FIG / "worst_covered_compounds.csv", index=False))
    print(f"  figures/worst_covered_compounds.csv")

    samp.nlargest(25, "outlier_score")[["smiles_std", "membership", "outlier_score"] + DESC_COLS] \
        .round(3).to_csv(FIG / "property_outliers.csv", index=False)
    print(f"  figures/property_outliers.csv")


if __name__ == "__main__":
    main()
