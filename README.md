# Exploring CYP3A4 inhibitors: ChEMBL versus OpenADMET

A **two-hour, self-contained chemoinformatics workshop** that runs start to finish in Google Colab.
Click the badge, run the cells, no installation.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AxelRolov/chimie_lia/blob/colab-workshop/notebooks/01_cyp3a4_chembl_vs_openadmet.ipynb)

## The question

> **How well does ChEMBL cover the chemical space of the OpenADMET compounds?**

Two public datasets measure inhibition of **the same human enzyme, CYP3A4** — the cytochrome P450 responsible for
clearing roughly half of all marketed drugs, and therefore the enzyme behind most drug–drug interactions.

| | **ChEMBL** | **OpenADMET CYP challenge** |
|---|---|---|
| what it is | ~30 years of medicinal-chemistry literature, manually curated | one industrial screening campaign, released for a blind challenge |
| target | [`CHEMBL340`](https://www.ebi.ac.uk/chembl/explore/target/CHEMBL340), UniProt [`P08684`](https://www.uniprot.org/uniprotkb/P08684) | CYP3A4 (plus 1A2, 2C9, 2D6) |
| size in this snapshot | 13 887 IC50 records · 11 237 unique structures | 4 905 compounds, **2 335** with a CYP3A4 value |
| endpoint | IC50 (nM), from 2 810 different assays | pIC50 from **one** assay, with a credible interval |
| why the compounds are in it | someone found them worth publishing | they were in the plate |

Students answer the question with **four independent measurements** — exact structure overlap, nearest-neighbour
Tanimoto, Bemis–Murcko scaffolds and a joint UMAP — and reach a result that is genuinely interesting:

> ChEMBL has nearly **five times** as many distinct CYP3A4-tested structures as the challenge set, and still
> **87 %** of the challenge compounds have no ChEMBL molecule within Tanimoto 0.4, while **94 %** sit on a ring
> skeleton ChEMBL has never tested against this enzyme. **More data does not mean more coverage.**

No predictive model is trained. The workshop is about *data*: where it comes from, what it means, and what you are
allowed to conclude from it.

## The 120 minutes

| time | section | what the student ends up with |
|---|---|---|
| 0–10 | CYP3A4, the two datasets, SMILES, IC50 and pIC50 | an interactive IC50 ⇄ pIC50 converter; the target and endpoint understood |
| 10–20 | **ChEMBL route A: the website** | a CSV exported by hand, parsed, and mapped onto a common schema |
| 20–30 | **ChEMBL route B: the API** | the same records by URL, with paging, retries and caching |
| 30–45 | Load, inspect and curate both sources | an assay browser, a curation funnel, one standardisation policy, censored data kept apart |
| 45–60 | Descriptor-based exploratory analysis | distributions compared source by source, with effect sizes and uncertainty |
| 60–80 | Morgan fingerprints and Tanimoto similarity | the nearest ChEMBL neighbour of every challenge compound, browsable |
| 80–95 | Bemis–Murcko scaffolds | common, shared and novel scaffolds; a fair matched-size diversity comparison |
| 95–110 | PCA and UMAP | one joint map of both sources, coloured four ways |
| 105–115 | Outlier detection and interpretation | two kinds of "unusual", kept apart and never auto-deleted |
| 115–120 | The per-compound dashboard and the answers | every result for one molecule at a time, then the evidence-based conclusions |

Seven exercises with hidden instructor solutions are distributed through the notebook (§1.1, §2.1, §3.1, §4.1,
§5.1, §6.1, §8.1) — the five the brief called for, plus one on querying a second target and one on the curation
rules themselves.

## It is interactive

The notebook is built to be *driven*, not read. Sliders, dropdowns and hover-plots throughout:

- an **IC50 ⇄ pIC50 converter** on a slider;
- an **assay browser** — pick one of the 25 largest ChEMBL assays and read what it actually measured, with its own
  pIC50 histogram;
- a **descriptor comparator** — choose any of nine descriptors, get overlaid histograms, medians and Cohen's *d*;
- a **two-molecule Tanimoto explorer** with structure drawings;
- a **coverage slider** — move the Tanimoto threshold and watch the covered fraction;
- a **nearest-neighbour browser** — step through all 2 335 compounds sorted by coverage, seeing each pair drawn;
- a **scaffold browser** with `mols2grid` grids;
- a **joint UMAP** with a colour-by dropdown, plus a plotly version you can hover and filter by clicking the legend;
- an **outlier inspector** switching between property outliers and novelty outliers;
- a final **compound dashboard**: one card per molecule with its activity, its uncertainty, its nearest ChEMBL
  neighbour and every flag it earned.

## For students

1. Click the **Open in Colab** badge above.
2. Run the first cell (⚙️ Setup) and wait for the installation to finish (1–2 min).
3. Work through with `Shift + Enter`. **Move every slider.** Change things, break them, fix them.
4. To keep your work: *File → Save a copy in Drive*.

Everything runs on **CPU**; no GPU and no API key are needed. The whole notebook executes in about **one minute** of
compute — the rest of the two hours is thinking, reading and discussion.

## For instructors

See [`INSTRUCTORS.md`](INSTRUCTORS.md) for the minute-by-minute plan, the discussion points, the expected numbers
and what to prepare before class.

Notebooks are generated from a [jupytext](https://jupytext.readthedocs.io) percent-format Python source in
`src_nb/`, which is what you edit (it diffs cleanly in git):

```bash
pip install jupytext nbformat nbclient
python src_nb/build.py                 # rebuild the notebook (clean, no outputs)
python src_nb/build.py 01 --execute    # rebuild and run it, reporting any error
```

To refresh the cached teaching data from the live sources:

```bash
python utils/prepare_cached_data.py            # re-download and re-derive everything
python utils/prepare_cached_data.py --offline  # re-derive only, from the cached raw files
```

This workshop currently lives on the **`colab-workshop`** branch, and `BRANCH` in `src_nb/build.py` plus `REPO_RAW`
in the notebook's setup cell point there. If it is merged to `main`, set both back to `main` and rebuild. If you
fork the repository, change `REPO` in `src_nb/build.py` too.

```
notebooks/     the .ipynb the students open (generated — do not edit by hand)
src_nb/        the jupytext source + build.py
data/          the cached snapshot with full provenance (see data/README.md)
utils/         prepare_cached_data.py — rebuilds data/ from the live sources
               export_figures.py      — writes figures/ (8 PNGs + 4 CSV tables)
               selftest.py            — offline checks: structures, row counts, curation invariants
figures/       exported figures and summary tables, for slides
INSTRUCTORS.md the teaching guide
CREDITS.md     full attribution of every adapted source
```

## Data and provenance

Everything in `data/` carries its origin in [`data/provenance.json`](data/provenance.json): ChEMBL release and
retrieval date, every query URL, the pinned Hugging Face commit, row counts and SHA-256 checksums of every file.
The OpenADMET training file is pinned to commit
[`146282d2`](https://huggingface.co/datasets/openadmet/cyp-challenge-train-test/tree/146282d2ce1532f1803679a834a9bec9dac1e482)
and its checksum is verified on download, so the numbers quoted in the notebook stay true.

The notebook prefers **live** downloads where they are instructive (the target check, the API paging demonstration)
and falls back to the cached snapshot when a service is unreachable — saying so explicitly when it does. It never
silently substitutes a different target or synthetic data.

See [`data/README.md`](data/README.md) for the file-by-file table.

## Where this comes from

This workshop is built on the [chemoinformatics tutorials](https://github.com/AxelRolov/chemoinformatics_tutorials)
collection, reusing its molecular-representations (session 02), chemical-databases (03) and exploratory-data-analysis
(04) material, and replacing their EGFR / ExpansionRx examples with a **matched pair of datasets measured on the same
protein**. That collection is itself a compilation of open teaching resources published by the chemoinformatics
community — **TeachOpenCADD** (Volkamer lab), **Practical Cheminformatics Tutorials** (Pat Walters), **AI for
Chemistry** (Schwaller group, EPFL), the **MolSSI** workshop, and **RDKit**.

**The credit for the substance belongs to those authors.** [`CREDITS.md`](CREDITS.md) lists every source with its
licence. If you reuse this material, please cite the **original authors**, not this repository.

## Licence

The compilation (sequencing, code, exercises, connective text): **CC BY 4.0** — see [`LICENSE`](LICENSE).
Code snippets: **MIT**. Adapted third-party material keeps its original licence, as recorded in `CREDITS.md`;
in particular, content derived from TeachOpenCADD is CC BY 4.0, data derived from ChEMBL is **CC BY-SA 3.0**, and the
OpenADMET dataset is distributed under **Apache-2.0**.
