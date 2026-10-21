# Instructor guide

A two-hour practical, in person or hybrid. The notebook is prepared: students run cells and think, rather than
typing a pipeline from scratch — implementing all of this from zero would take a day, not two hours.

The scientific thread that must survive the two hours, whatever else gets cut:

> **The same protein does not mean interchangeable measurements.**
> We compare the two sources *structurally*, freely and jointly. We keep their *activity values* separate, labelled
> by source and by assay, from the first cell to the last.

---

## Before the class

1. **Run the notebook yourself, end to end, on a clean kernel.** It takes about a minute of compute.
   ```bash
   python src_nb/build.py 01 --execute
   ```
   Expect `0 error(s)`. The executed copy is written to `notebooks/.executed_*.ipynb` (git-ignored) for inspection.

2. **Run the self-test.** It is offline and takes a second:
   ```bash
   python utils/selftest.py
   ```
   It verifies the notebook's named reference drugs by **InChIKey**, the three stereochemistry twins of section 5,
   every cached row count, and the curation invariants (relations, units, the pIC50 formula, the uncertainty
   columns, the overlap count). It exists because during development two reference SMILES were silently wrong —
   they parsed and drew perfectly, and only an identity check caught them. Run it after any edit to the source or
   to `data/`.

3. **Check the cached data is current.** `data/provenance.json` records the ChEMBL release the snapshot was built
   from. If a new ChEMBL release has appeared and you want to use it:
   ```bash
   python utils/prepare_cached_data.py
   ```
   Then re-run the notebook — **the numbers in the markdown text will need updating** (see "Numbers to expect"
   below, and search the source for them). If you would rather not, the snapshot is self-consistent and dated;
   teaching from a dated snapshot is perfectly honest, and is itself a lesson about reproducibility.

4. **Warm up UMAP.** The first `umap` import in a fresh Colab session triggers numba compilation (~20–30 s). Have
   students run the setup cell at the very start of the session, not at minute 95.

5. **Review the assay selection.** `data/chembl_cyp3a4_assay_review.csv` holds one row per assay with automatic
   context flags from keyword rules (defined in `utils/prepare_cached_data.py`). Sort it by `n_records` and read
   the top 30 descriptions. The rules are deliberately simple and visible so you can disagree with them; if you
   change them, re-run the prep script and the funnel changes accordingly.

6. **Decide on the live-download policy.** With thirty students hitting the ChEMBL API simultaneously, consider
   setting `MAX_PAGES = 1` in section 2, or telling students to skip the live pull. Everything after section 3 uses
   the cached snapshot regardless, so nothing downstream depends on it.

---

## Minute-by-minute

| time | section | run it as |
|---|---|---|
| **0–10** | Introduction | Talk. CYP3A4 as an *anti-target*: high IC50 is good. Use the IC50 ⇄ pIC50 slider live on the projector — it fixes the log scale better than a formula on a slide. |
| **10–20** | ChEMBL route A: website | **Everyone opens a browser.** This is the only section that is not code, and it is the one students remember. Make them verify the target (organism! `SINGLE PROTEIN`!) before exporting. |
| **20–30** | ChEMBL route B: API | Show the paging loop, then the official client. The point is that the client is not magic. |
| **30–45** | Loading and curation | The heart of the session. Spend the time on the **assay browser** and the **funnel**. |
| **45–60** | Exploratory analysis | Let them drive the descriptor widget themselves. Exercise 4.1 here. |
| **60–80** | Similarity | The nearest-neighbour browser is the best part of the notebook — give it room. Exercise 5.1. |
| **80–95** | Scaffolds | Exercise 6.1. Make the matched-sample-size point explicitly; it is a real methodological trap. |
| **95–110** | PCA and UMAP | Read the PCA loadings aloud. Then warn about over-reading UMAP *before* showing it, not after. |
| **105–115** | Outliers (§8) | Exercise 8.1 at ~min 110. Make the two-kinds-of-outlier distinction explicit. |
| **115–120** | §9 dashboard and §10 answers | Slide through three or four compound cards, then read the §10 table. **Never cut §10** — it is the payoff. |

If you are running short: compress §2 (the API route) and §7 (PCA/UMAP) — the headline result is already established
by §5 and §6. Do **not** cut §3; a workshop about data that skips curation teaches the wrong thing.

---

## Numbers to expect

From the snapshot in `data/` (ChEMBL 37, OpenADMET revision `146282d2`). If a student gets something different,
something changed — which is worth discussing.

| quantity | value |
|---|---|
| ChEMBL IC50 records for CHEMBL340 | 13 887, in 2 810 assays, 2 275 documents, 1992–2025 |
| … surviving curation as exact, direct-inhibition, nM values | **3 980** (28.7 %) |
| … kept separately as censored (`>`, `<`) | 3 710, of which 3 552 are `>` |
| ChEMBL unique standardised structures (the reference set) | **11 237** |
| OpenADMET rows / with a CYP3A4 value | 4 905 / **2 335** |
| OpenADMET compounds with pIC50 < 4 | 944 (40 %) |
| median credible-interval width, pIC50 < 4 vs ≥ 4 | **2.15** vs **0.22** log units |
| exact structure overlap | **52** (2.2 % of OpenADMET) |
| … of those, with an exact curated ChEMBL value to compare | only **7** |
| median max Tanimoto to ChEMBL (self-matches excluded) | **0.325** |
| OpenADMET compounds with no ChEMBL neighbour at T ≥ 0.4 | 2 024 (**86.7 %**) |
| distinct scaffolds, ChEMBL / OpenADMET | 6 119 / 2 095 |
| shared scaffolds | **70** |
| OpenADMET ring compounds whose scaffold ChEMBL has never seen | 2 188 (**93.9 %**) |
| scaffolds in a ChEMBL sample of matched size (2 329) | 1 763 ± 19 — fewer than OpenADMET's 2 095 |
| "blind spots" (T < 0.30 **and** novel scaffold) | **660** (28.3 %) |
| PCA explained variance, PC1 + PC2 | 48 % + 21 % = 69 % |

**Runtime** (Colab CPU): setup 1–2 min · standardisation ~11 s · 26 million Tanimoto comparisons ~3 s ·
UMAP on 1 997 fingerprints ~9 s. Nothing in the notebook takes longer than fifteen seconds.

---

## The discussion points worth protecting

**1. Why 29 % survives curation (§3).** Students assume curation means "removing bad data". Here it mostly means
*separating* data: 3 710 of the discarded records are censored measurements that carry real information, just not
the kind you can put in a regression. Ask: *what would happen if we treated "IC50 > 20 000 nM" as "IC50 = 20 000 nM"?*

**2. Do not require `assay_type == "B"` (§3).** The EGFR workflow in the parent tutorial does, because kinase
inhibition is a binding assay. Here, 11 582 of 13 887 CYP3A4 records are type **`A` (ADME)** — exactly where CYP
inhibition belongs. Copying a filter from one target to another would silently destroy five sixths of the data.
This is the single most transferable warning in the session.

**2b. Confidence 9 does not mean "this experiment measures CYP3A4" (§3).** The notebook's §3 now shows the top ten
most potent records as a sanity check. Make students look. The best entry is **ketoconazole at 0.9 nM** — the right
answer, and reassuring. But point out what the text records: before the `mentions_target` rule was added, the four
most potent "CYP3A4 inhibitors" came from `CHEMBL3705484`, an **MDM2–p53 HTRF binding assay** that ChEMBL links to
CHEMBL340. It had confidence 9, assay type B, human organism and no TDI keywords. The confidence score describes
the *target assignment*, not your interpretation of the experiment. This is the best argument in the session for
reading assay descriptions and for sorting by your endpoint and looking at the extremes.

**3. Confidence 9 versus 8 (§3, Exercise 3.1).** Score 8 means the target was assigned *by homology*. For a family
of close isoforms, that is precisely the ambiguity you cannot afford. Students who relax the filter gain records and
lose the right to say "human CYP3A4".

**4. Why the pIC50 distributions differ by a log unit (§4).** Three causes, confounded and not separable from the
plot: selection (people publish what they found interesting), censoring (ChEMBL's weak compounds moved to the
censored table), assay differences. If a student says "ChEMBL compounds are more potent", ask which of the three
they mean.

**5. Coverage ≠ agreement (§3, §5).** Everything we conclude is about *structural* coverage. We never claim the
two sources agree on potency — with 7 comparable molecules we could not, and Landrum & Riniker (2024) found about
one log unit of noise when IC50s from different sources are combined.

**6. Novel scaffold ≠ no similar molecule (§6, Exercise 6.1).** Scaffold identity is exact and brittle: one extra
ring and the match is lost. Tanimoto is graded. 94 % of compounds have a novel scaffold but the median max Tanimoto
is still 0.33, not 0.05. The compounds worth worrying about are novel on **both** counts.

**7. Two kinds of outlier (§8).** Unusual *properties* (isolation forest: macrolides, peptides, a C18 quaternary
ammonium surfactant) versus unusual *relative to a reference* (low Tanimoto, unseen scaffold). The red points on the
PCA/UMAP figure sit in the middle of the cloud, not at the edges — which is the whole lesson. Neither kind justifies
deleting a compound.

**8. Why the result is counter-intuitive (§10).** ChEMBL is bigger and older and still covers the challenge set
poorly, because the two collections were built for different purposes: deep-and-narrow analogue series versus
broad-and-shallow diversity screening. "We have more data" is not the same as "we have the right data".

---

## Exercises and when to set them

| # | section | asks | the answer hinges on |
|---|---|---|---|
| 1.1 | §1, ~min 18 | missing value vs censored measurement | a missing value carries no information; a `>` carries real information and is usable for classification, never for regression |
| 2.1 | §2, ~min 29 | query a second CYP isoform from the API | verifying a target before trusting a download; `page_meta.total_count` to size a query without downloading it |
| 3.1 | §3, ~min 43 | rebuild the funnel with confidence ≥ 8 | homologous target assignment is fatal for a CYP isoform |
| 4.1 | §4, ~min 57 | explain a source difference in a descriptor distribution | purpose of the collection — optimised leads vs diversity library; and HBD as the descriptor that *cannot* tell them apart |
| 5.1 | §5, ~min 77 | read a nearest-neighbour pair at T ≈ 0.5 | the similarity principle is a tendency, not a law; and the two numbers come from different assays |
| 6.1 | §6, ~min 93 | find a scaffold absent from ChEMBL | exact-and-brittle vs graded similarity |
| 8.1 | §8, ~min 110 | defend a flagged outlier | statistically unusual ≠ wrong; macrolides are both extreme *and* the known potent CYP3A4 inhibitors |

Solutions are in `<details>` blocks directly under each exercise, so students can self-check. They render collapsed
in Colab.

---

## Deliberately out of scope

Model training, docking, activity cliffs, t-SNE/TMAP, and the OpenADMET `tdi` / `single_concentration` / `emax`
configurations. The blinded test set (`cyp-challenge-TEST-BLINDED.csv`, 750 compounds, SMILES only) is a reasonable
*optional* extension for structure-only exploration — it has no labels, so it cannot be used for anything else.

The natural sequel is a QSAR model with a declared applicability domain (session 05 of the parent
[chemoinformatics tutorials](https://github.com/AxelRolov/chemoinformatics_tutorials)) — and this workshop has
already told the student which compounds fall outside it.

## Troubleshooting

| symptom | cause and fix |
|---|---|
| Widgets show as static text in Colab | The setup cell calls `output.enable_custom_widget_manager()`. If a student skipped it, re-run the setup cell and then the widget cell. |
| `mols2grid` grid is blank | Same cause; the §6 cell has a `Draw.MolsToGridImage` fallback in its `except` branch. |
| The live ChEMBL pull hangs | EMBL-EBI rate-limits bursts. Set `MAX_PAGES = 1`, or skip the cell — nothing downstream needs it. |
| UMAP takes minutes | First-call numba compilation. Warm it up during the break. |
| Different numbers from the table above | Check `data/provenance.json`: a student may be pulling a newer ChEMBL release live. |
