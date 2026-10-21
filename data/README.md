# Cached teaching data

Everything here is derived from two public sources and can be rebuilt with:

```bash
python utils/prepare_cached_data.py            # re-download and re-derive
python utils/prepare_cached_data.py --offline  # re-derive from the cached raw files only
```

[`provenance.json`](provenance.json) is the authoritative record: ChEMBL release and retrieval date, every query
URL, the pinned Hugging Face revision, row counts, and a SHA-256 checksum for every file in this directory.

## Files

| file | rows | content |
|---|---|---|
| `chembl_cyp3a4_ic50_raw.csv.gz` | 13 887 | every ChEMBL IC50 activity record for `CHEMBL340`, exactly as the REST API returned it (22 columns kept). The input to all curation. |
| `chembl_cyp3a4_assays.csv.gz` | 2 810 | the assay record of every assay appearing above: description, type, confidence score, BAO format, organism, subcellular fraction. |
| `chembl_cyp3a4_assay_review.csv` | 2 810 | one row per assay: the automatic context flags (`mentions_target`, `time_dependent`, `induction`, `activation`), the detected `probe_substrate` and `enzyme_system`, and the review columns `instructor_excluded` / `excluded_reason`. The keyword rules and the manual exclusion list both live in `utils/prepare_cached_data.py`; edit either and re-run the script. 10 assays are excluded, including an MDM2–p53 HTRF assay that ChEMBL links to CHEMBL340 and that otherwise supplied the four most potent values in the curated set. |
| `chembl_cyp3a4_curated.csv` | 3 980 | the core quantitative subset: human CYP3A4, assay confidence 9, direct-inhibition context, `standard_relation` `=`, finite positive nM value, no validity flag, parsable structure. One row per **measurement**, with `pIC50`, the standardised structure and its InChIKey. |
| `chembl_cyp3a4_censored.csv` | 3 710 | the records that pass every rule **except** the relation test — `>`, `<`, `>=`, `<=`. Kept so they can be used as inactives in a classification setting, and so they are never mistaken for exact values. |
| `chembl_cyp3a4_structures.csv` | 11 237 | one row per unique standardised structure with **any** CYP3A4 IC50 record, censored included, plus nine descriptors and its Bemis–Murcko scaffold. This is the **similarity reference set** — "the chemical space ChEMBL has looked at for this target". |
| `chembl_cyp3a4_funnel.csv` | 8 | the curation funnel, step by step, as printed in the notebook. |
| `chembl_ui_export_sample.csv` | 200 | the first 200 rows of a ChEMBL **website** CSV export of the same query — semicolon-separated, every field quoted, `None` for missing. Used by section 1 so the manual-download lesson works even if a student's own export fails. |
| `openadmet_cyp_train_inhibition.csv` | 4 905 | byte-for-byte copy of the pinned OpenADMET direct-inhibition training file. Checksum verified on download. |
| `openadmet_cyp3a4_curated.csv` | 2 335 | the rows with a CYP3A4 value, standardised, with the **uncertainty columns preserved** (`pIC50_conf_low`, `pIC50_conf_high`, `pIC50_std`, `pIC50_ci_width`), a `weak_potency_flag` for pIC50 < 4, nine descriptors and the scaffold. |

## Sources and licences

| source | what | licence |
|---|---|---|
| [ChEMBL](https://www.ebi.ac.uk/chembl/) (EMBL-EBI), release recorded in `provenance.json` | target `CHEMBL340` (human CYP3A4, UniProt `P08684`), its IC50 activities and their assays | **CC BY-SA 3.0** |
| [OpenADMET CYP challenge](https://huggingface.co/datasets/openadmet/cyp-challenge-train-test), revision `146282d2ce1532f1803679a834a9bec9dac1e482` | `cyp-challenge-TRAIN_inhibition.csv` — 4 905 compounds, pIC50 with credible intervals for CYP1A2/2C9/2D6/3A4 | **Apache-2.0** |

Derived files (everything with `curated`, `structures`, `review` or `funnel` in the name) inherit the licence of the
source they are derived from.

## The curation rules, in one place

Applied in `curate_chembl()` — identically in `utils/prepare_cached_data.py` and in the notebook, so students see
the same funnel they could reproduce themselves:

1. target organism is *Homo sapiens*;
2. assay **confidence score 9** — "direct single protein target assigned" (8 means assigned by homology, which for a
   CYP isoform family is a real risk of confusion);
3. assay context is **direct inhibition** — no pre-incubation/TDI, no induction, no activation;
4. the record has a structure;
5. no ChEMBL `data_validity_comment`;
6. finite, positive value in **nM**;
7. relation `=` → the quantitative set; `>`, `<`, `>=`, `<=` → the censored table.

`assay_type` is **not** filtered: 11 582 of the 13 887 records are type `A` (ADME), which is where CYP inhibition
data belongs. Requiring type `B` would discard five sixths of the data.

`potential_duplicate` is reported but not used as a filter — ChEMBL sets it when the same value appears twice in one
document, which is often a legitimate replicate. Deduplication happens on **structure** instead.

## The standardisation policy

One policy, applied identically to both sources:

1. parse the SMILES (failures counted, never silently dropped);
2. `rdMolStandardize.Cleanup()` — sanitise, normalise functional groups, reionise;
3. `LargestFragmentChooser(preferOrganic=True)` — remove counter-ions and solvents;
4. `Uncharger()` — neutralise where chemistry allows;
5. canonical **isomeric** SMILES — **stereochemistry is kept**.

**No tautomer canonicalisation**: it is slow, it is a modelling choice rather than a truth, and the ChEMBL structure
pipeline does not do it either. The original SMILES is preserved in every table so the policy can be changed and the
analysis redone.

Exact cross-source overlap is decided on the **standardised isomeric SMILES**, not on a fingerprint similarity of
1.0 — different molecules can share a fingerprint.
