# Cached teaching data

These files are prepared **before class** with `python scripts/prepare_data.py`. Run
`python scripts/prepare_data.py --check` to verify them against the SHA-256 checksums in
[`provenance.json`](provenance.json). The notebook uses these files unless a student downloads live data. Whenever
the notebook falls back to a cached file, it says so on screen.

| file | content | how it was obtained | licence |
|---|---|---|---|
| `openadmet/cyp-challenge-TRAIN_inhibition.csv` | OpenADMET direct-inhibition training set: 4,905 rows covering four isoforms, 2,335 of them with a CYP3A4 pIC50 together with its 95 % interval and standard deviation | [`openadmet/cyp-challenge-train-test`](https://huggingface.co/datasets/openadmet/cyp-challenge-train-test), config `default`, split `train`, revision `146282d2ce1532f1803679a834a9bec9dac1e482` | Apache-2.0 |
| `openadmet/cyp-challenge-TEST-BLINDED.csv` | 750 blinded test structures without labels, used only for the optional structure check | same revision | Apache-2.0 |
| `chembl/api/CHEMBL340_IC50_activities.jsonl.gz` | every `IC50` activity of target `CHEMBL340`: one raw API record per line, unmodified | `activity.json?target_chembl_id=CHEMBL340&standard_type=IC50`, all pages of 1,000 | CC BY-SA 3.0 |
| `chembl/api/CHEMBL340_IC50_assays.jsonl.gz` | metadata of every assay referenced by those records (confidence score, description, format…) | `assay.json?assay_chembl_id__in=…`, batches of 100 | CC BY-SA 3.0 |
| `chembl/api/CHEMBL340_target.json` | the target record (organism, UniProt accession, target type) | `target/CHEMBL340.json` | CC BY-SA 3.0 |
| `chembl/manual/CHEMBL340_IC50_activities_web_export.zip` | the **website** export of the same activities: a `;`-delimited CSV, kept exactly as ChEMBL delivers it | the website's CSV button, applied to the activity browser filtered by `target_chembl_id:CHEMBL340 AND standard_type:"IC50"` | CC BY-SA 3.0 |
| `chembl/manual/CHEMBL340_assays_web_export.zip` | the website export of all CHEMBL340 assays, which supplies the confidence score that the activity export lacks | the website's CSV button in the assay browser filtered by `target_chembl_id:CHEMBL340` | CC BY-SA 3.0 |
| `chembl/assay_review.csv` | every confidence-9 assay with an IC50 record: keyword-rule decision, enzyme source, probe, and the hand-reviewed decisions (`reviewed = True`) | `scripts/prepare_data.py` (rules) plus review before class (see `INSTRUCTORS.md`) | CC BY-SA 3.0 (descriptions) |
| `provenance.json` | ChEMBL release and date, retrieval time, queries, URLs, record counts and SHA-256 of every file | `scripts/prepare_data.py` | — |

## Notes on the two ChEMBL routes

- The API and the website export come from the **same release**, so the notebook can show that they contain the
  same records. It compares a composite key (molecule · assay · document · relation · value · units) because the
  website export has **no activity ID**.
- The website export quotes relations inside the cell (`'='`) and writes missing values either as empty strings or
  as the text `None`. The assay export repeats the column name `Compounds`. The manual route in the notebook
  handles each of these quirks explicitly.
- Edits to `assay_review.csv` survive `prepare_data.py`: rows marked `reviewed = True` are never overwritten.
