# Instructor guide: CYP3A4 inhibitors, ChEMBL versus OpenADMET

A two-hour workshop for students who know basic Python and pandas. Everything runs in
[molab](https://molab.marimo.io) (marimo's free cloud: 4 CPUs and 32 GB of RAM, with no installation). The
notebook is `notebooks/cyp3a4_chembl_vs_openadmet.py`; a French translation with identical code is
`notebooks/cyp3a4_chembl_vs_openadmet_fr.py` (labels of data values — curation reasons, assay classes,
option values — stay in English in both, because the code uses them).

> **Central question.** *How well does ChEMBL cover the chemical space of the OpenADMET compounds?*
> **Scientific thread.** The same protein does not make measurements interchangeable. Structures are compared
> jointly. Activities stay separate per source and per assay and are never averaged across them.

---

## 1. Before class

### One week before

- [ ] Run `python scripts/prepare_data.py --check`. Every file should report `ok`.
- [ ] Decide whether to refresh the data. `python scripts/prepare_data.py` downloads everything again (≈ 2 min)
      and rewrites `data/provenance.json`. Refresh only if you *want* a newer ChEMBL release. The numbers in this
      guide come from **ChEMBL 37**, retrieved on 2026-10-21.
- [ ] Read the **hand-reviewed assays** in `data/chembl/assay_review.csv`, the rows with `reviewed = True`. These are
      the 40 largest confidence-9 assays plus four excluded examples. Their descriptions were read before class, and
      they agree with the keyword rules. Change `review_decision` / `review_reason` if you disagree:
      `prepare_data.py` never overwrites reviewed rows. The notebook's assay table (filter "instructor-reviewed")
      shows your decisions.
- [ ] Run `python scripts/verify_workshop.py --live`. It runs the whole notebook twice on CPU: once with
      default widget values (a clean kernel), and once simulating student clicks, an uploaded export and another
      ChEMBL route. It then downloads ChEMBL live and pushes it through the same pipeline. Expect
      `N/N checks passed` (≈ 4 min).

### On the day, 15 minutes before

- [ ] Open the molab link from the README and click **Run all** once. The first run takes about a minute,
      including the UMAP compilation (warm-up). Later re-runs are fast.
- [ ] Check that the green callouts appear: target verified, routes agree, unit checks pass.
- [ ] Have the [ChEMBL activity browser](https://www.ebi.ac.uk/chembl/explore/activities/QUERYSTRING:target_chembl_id%3ACHEMBL340%20AND%20standard_type%3A%22IC50%22)
      open in a tab for the manual-download demonstration. The export job takes ≈ 30 s on the ChEMBL server.

### Fallbacks

| problem | what happens |
|---|---|
| ChEMBL website or API down | The live peek and the live-download button show a red or yellow callout. Everything else runs on the cached snapshot, and every fallback is labelled in the notebook. Nothing silently switches to other data. |
| Hugging Face down | OpenADMET loads from the cached copy, which has the same checksum. The notebook says so explicitly. |
| A student's export differs from the cached one | The route comparison table says which counts differ. The usual cause is a different filter, for example a missing `IC50` filter, or exporting only the visible page. |
| molab unavailable | Run locally with `uvx marimo edit --sandbox notebooks/cyp3a4_chembl_vs_openadmet.py`, or `pip install -r requirements.txt`. |
| Serving one copy to the whole class | Use molab, where each student has their own kernel. With `marimo run` on a shared server, all sessions share one Python process: the notebook serialises UMAP with a lock (numba is not thread-safe), but sessions will queue. |

---

## 2. Timing and talking points

The numbers are those of the cached snapshot (ChEMBL 37, OpenADMET revision `146282d`).

### 0–10′ · Introduction
- CYP3A4 metabolises a large share of drugs, so inhibiting it causes drug–drug interactions. Ketoconazole
  combined with midazolam is the classic example.
- pIC50 = 9 − log10(IC50/nM). Use the converter: 1 µM gives 6, and 100 µM gives 4.
- Show the target check: `CHEMBL340` is human, a single protein, UniProt P08684.
- **Say early:** OpenADMET uses a single protocol (recombinant CYP3A4 with the fluorogenic probe DBOMF, a 12-point
  dose–response and a Bayesian fit). ChEMBL mixes hundreds of protocols.

### 10–20′ · ChEMBL by hand
- Demonstrate the website: target card, then activities, then the IC50 filter, then **CSV**. The download is a
  `.zip` containing one `;`-delimited CSV. ChEMBL 37 has **13,887** records.
- Point out three pitfalls of the export file, all visible in the notebook:
  1. the relation is quoted inside the cell (`'='`);
  2. missing values are written as either `""` or `None`;
  3. there is **no activity ID and no confidence score**, so the notebook joins the assay export on
     `Assay ChEMBL ID`. The assay export also repeats the header `Compounds`.
- Students who could not download continue with the cached export. It is labelled as such.

### 20–30′ · ChEMBL by API
- Explain one query URL, `page_meta` and paging with `next`. The live download button takes 1–2 min, so let one
  volunteer press it rather than the whole room.
- **Key result:** after the same normalisation, the manual and API routes agree on every record, including after
  curation, and on the retained activity IDs (matched by composite key). The provenance table lists the release,
  the date, the queries and the SHA-256 checksums.

### 30–45′ · Load and clean
- OpenADMET has 4,905 rows covering four isoforms; **2,335** of them have a CYP3A4 value. A missing value is not an
  inactive one: show the table of isoform combinations.
- **944 (40 %)** OpenADMET CYP3A4 values fall below pIC50 4. The uncertainty chart shows their intervals widening.
  They are flagged but never clipped or deleted.
- **Exercise 1** (missing versus censored), as a quiz plus code. About 5 min.
- ChEMBL curation funnel: 13,887 records become **4,346 quantitative**, 4,022 censored (kept apart) and 5,519
  excluded. The main exclusion is assay confidence below 9 (3,861 records, including DrugMatrix). The others are
  assay context (538: time-dependent inhibition, hepatocytes, induction), units or missing values (548), potential
  duplicates (294), validity flags (126) and exact duplicates (126).
- Use the **what-if** selector live: switch off "assay confidence" and watch the numbers move.
- Standardisation: 375 ChEMBL records lose a counter-ion or solvent. Click a row to see the before and after
  structures (indinavir loses its water, for example).

### 45–60′ · Descriptors
- OpenADMET is smaller and less lipophilic than ChEMBL: median MW 344 vs 441 Da, cLogP 2.7 vs 4.0, two aromatic
  rings vs three.
- The activity boxplots show that ChEMBL's quantitative medians (≈ 5.2–5.7) are higher than OpenADMET's (4.3).
  **Make sure students see the grey row.** ChEMBL's weak compounds are mostly censored ("IC50 > x"), whereas
  OpenADMET reports a number for every compound. The difference is partly how results are reported, not chemistry.
- Only 8 molecules are in both curated sets (6 with quantitative ChEMBL values): one point per ChEMBL
  measurement, nothing averaged.
- **Exercise 2** (explain a descriptor difference). About 5 min. Answer: chemical selection.

### 60–80′ · Similarity
- Pair demo: midazolam vs alprazolam (T ≈ 0.48); alprazolam vs triazolam (0.77); **testosterone vs epitestosterone
  (T = 1.00 but different molecules)**. This is why identity is decided by standardised isomeric SMILES.
- Nearest neighbours: the default reference holds 7,249 ChEMBL structures. The median nearest-neighbour T is
  **0.31**, only **6.9 %** of compounds have a neighbour at T ≥ 0.4, and 41 % stay below 0.30.
- Sensitivity table: with *any* CHEMBL340 IC50 record, the exact overlap goes from 8 to **52**. Most of the extra
  matches are approved drugs tested in DrugMatrix (confidence 8). **The curation policy is part of the answer.**
- **Exercise 3**: click a compound and inspect the highlighted differences. About 7 min.

### 80–95′ · Scaffolds
- ChEMBL has 4,088 scaffolds and OpenADMET 2,094; they share only **39**. **96 %** of the cyclic OpenADMET
  compounds have a scaffold that never occurs in ChEMBL.
- The rarefaction plot compares diversity at matched sample sizes: the ratio of scaffolds per compound depends on
  set size.
- **Exercise 4**: find the largest OpenADMET-only scaffold. About 5 min.

### 95–110′ · PCA and UMAP
- The map sample holds 2,000 structures (999 ChEMBL, 998 OpenADMET, 3 both), drawn with seed 42 and not enriched.
- PCA: PC1 explains 50 % of the variance and is essentially **size** (loadings: MW and heavy atoms 0.44, TPSA and
  HBA 0.37, rotatable bonds 0.36). PC2 explains 19 % and separates **saturated from aromatic** molecules (FracSP3
  +0.59, aromatic rings −0.49, cLogP −0.46). OpenADMET sits on the small, more saturated side: its median PC1 is
  −1.2, against +0.8 for ChEMBL.
- Weakly covered compounds are *not* simply smaller (median 24 vs 25 heavy atoms). They are new chemotypes for
  ChEMBL's CYP3A4 data, which is a good point for discussion.
- UMAP: one joint map, fitted once (≈ 6 s). Colour it by source, then by each source's own pIC50. Drag a rectangle
  to draw molecules. The **Refit UMAP** form changes `n_neighbors` / `min_dist` without refitting on every slider
  move.

### 110–120′ · Outliers and answers
- An Isolation Forest on the descriptors flags very flexible or lipophilic chains (RotB +9σ) and very small or polar
  molecules. Low maximum similarity flags structural novelty. The 2 × 2 table shows that the two flags overlap
  only partly.
- **Exercise 5** (why a flagged compound can be valid chemistry). Discuss it as a class.
- Finish on part 9: the three answers are computed live from the students' settings.

---

## 3. Exercise solutions (also hidden in the notebook)

1. **Missing versus censored.** A blank CYP3A4 cell in OpenADMET means *not measured*. ChEMBL's
   "IC50 > 30,000 nM" means pIC50 **< 4.52**: a real measurement, but a bound. Code:
   `(openadmet_raw[CYP3A4].isna() & other_isoform.notna().any(axis=1)).sum()` gives about 2,570, and
   `chembl_meas.loc[chembl_meas.subset == "censored", "standard_relation"].value_counts()` is mostly `>`.
2. **Descriptor difference.** Structure descriptors depend only on which molecules are in each set. OpenADMET mixes
   diversity-library compounds and drugs, while ChEMBL holds optimised medicinal-chemistry series. Publication bias
   can link the selection to the assay outcome indirectly. The most potent ChEMBL inhibitors are larger and more
   lipophilic.
3. **Neighbour pair.** At T ≈ 0.45 the pair usually shares one ring system and differs in substituents or linkers.
   Above 0.7 the difference is usually a single substituent. A neighbour's activity comes from another assay, or is
   only a bound, so it is a hint and not a replacement.
4. **Novel scaffold.** Filter the scaffold table to *OpenADMET only* and sort by count, then check its members in
   `nn_table`. Scaffold novelty and fingerprint novelty are related but not identical.
5. **Valid outliers.** Approved drugs, natural products, macrocycles, fragments and long-chain compounds are real
   chemistry. Check the structure, the standardisation and the uncertainty (pIC50 < 4), and keep the compound.

---

## 4. Design decisions you may be asked about

- **Why confidence 9 only?** The brief requires a verified single-protein assignment. The sensitivity table shows
  what this costs (52 exact matches fall to 8), which makes a good discussion point.
- **Why keep "(unknown origin)" assays?** ChEMBL assigns them to the human target at confidence 9, and CYP3A4 is a
  human gene name. In the curated snapshot they make up about 27 % of the quantitative records. Excluding them is a
  one-line change in `classify_assay`.
- **Why not `assay_type = B`?** Most CYP inhibition assays are typed *A* (ADME) in ChEMBL.
- **Time-dependent inhibition detection.** The keywords are *time-dependent*, *mechanism-based*, *irreversible*,
  *IC50 shift*, and pre-incubation **with NADPH**. Pre-incubation *followed by* NADPH addition is the direct
  condition and is kept. Borderline wordings are listed in `assay_review.csv`.
- **Duplicates.** ChEMBL's `potential_duplicate` flag (a value re-reported from an earlier paper) is excluded, and
  exact duplicate records are kept once. Different measurements of one molecule are **kept**, one row each. The
  per-structure ChEMBL median is used only to colour the maps, and it says so.
- **Standardisation.** RDKit cleanup, then the largest organic fragment, then neutralisation. Stereochemistry is
  kept and tautomers are not canonicalised (too slow, and not needed here). Every change is recorded.
- **Fingerprint.** Binary Morgan, radius 2, 2,048 bits, without chirality. That is why the epimers have T = 1.
- **What is deliberately left out:** model training, docking, activity cliffs and other embeddings.

## 5. Files produced

`outputs/` holds the results of a clean run on the cached snapshot:

- **tables** as CSV, gzipped above 5,000 rows (pandas reads them as they are): before/after counts, route
  comparison, curated ChEMBL measurements, OpenADMET measurements, structures with descriptors, nearest
  neighbours, scaffolds and outlier scores;
- **figures** as interactive HTML and PNG: OpenADMET uncertainty, descriptor distribution, pIC50 by assay,
  nearest-neighbour similarity, scaffold rarefaction, PCA, UMAP and outliers on the maps;
- **molecule grids** as SVG: the least-covered OpenADMET compounds and the property outliers.

The notebook regenerates all of them with its export button, or with
`WORKSHOP_EXPORT=1 python notebooks/cyp3a4_chembl_vs_openadmet.py`. PNG files are written only when the optional
package `vl-convert-python` is installed, as it was for the committed outputs. Students can also save any chart
from its ⋯ menu.
