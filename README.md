# Exploring CYP3A4 inhibitors: ChEMBL versus OpenADMET

A two-hour, hands-on chemoinformatics workshop in a reactive [marimo](https://marimo.io) notebook.

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/AxelRolov/chimie_lia/blob/main/notebooks/cyp3a4_chembl_vs_openadmet.py)

> **Central question.** *How well does ChEMBL cover the chemical space of the OpenADMET compounds?*

Both datasets concern the **same protein**: human cytochrome P450 3A4 (ChEMBL `CHEMBL340`, UniProt `P08684`).
ChEMBL contributes curated IC50 values from the literature and patents, measured with many different protocols.
The [OpenADMET CYP inhibition challenge](https://huggingface.co/datasets/openadmet/cyp-challenge-train-test)
contributes 2,335 CYP3A4 pIC50 values from a single biochemical protocol. Students compare the two sources'
**structures** with similarity, scaffolds, PCA/UMAP and outlier detection. No predictive model is trained.

Throughout the workshop, the **same protein does not mean interchangeable measurements**. Activities are kept
separate per source and per assay and are never averaged across them.

## For students

1. Click **Open in molab** above. The notebook runs in your browser on a cloud machine, with nothing to install.
   molab installs the packages listed at the top of the notebook.
2. Work from top to bottom. The notebook is *reactive*: moving a slider or clicking a table re-runs every cell
   that depends on it.
3. The five ✏️ exercises have hidden 🔑 solutions. Try each exercise before you open its solution.

To run the notebook locally instead, install [uv](https://docs.astral.sh/uv/) and run:

```bash
uvx marimo edit --sandbox notebooks/cyp3a4_chembl_vs_openadmet.py
```

You can also use `pip install -r requirements.txt` followed by `marimo edit notebooks/cyp3a4_chembl_vs_openadmet.py`.

## Schedule (120 minutes)

| time | part | student result |
|---|---|---|
| 0–10′ | introduction: CYP3A4, the datasets, SMILES, IC50 and pIC50 | understand the target and the endpoint |
| 10–20′ | ChEMBL option 1: manual download from the website | export a CSV and read it with pandas |
| 20–30′ | ChEMBL option 2: API download | retrieve the same target programmatically and show that both routes agree |
| 30–45′ | load OpenADMET; inspect and clean both sources | valid structures, plus a missingness and duplicate report |
| 45–60′ | descriptor-based exploratory analysis | property and activity distributions compared |
| 60–80′ | Morgan fingerprints and Tanimoto similarity | the nearest ChEMBL neighbour of every OpenADMET compound |
| 80–95′ | Bemis–Murcko scaffolds | common, shared and novel scaffolds |
| 95–110′ | PCA and UMAP | one joint map of both sources |
| 110–120′ | outlier detection and discussion | unusual compounds, explained; evidence-based answers |

## Repository layout

```
notebooks/cyp3a4_chembl_vs_openadmet.py   the workshop (marimo notebook; also a plain Python file)
scripts/prepare_data.py                   rebuilds the cached data and data/provenance.json before class
scripts/verify_workshop.py                runs the notebook headless (default and simulated clicks) and checks it
data/                                     cached teaching data with provenance (see data/README.md)
outputs/                                  tables, figures (HTML/PNG) and molecule grids (SVG) from a clean run
INSTRUCTORS.md                            preparation checklist, timing, talking points, solutions
CREDITS.md                                sources and licences
```

The notebook is a plain Python file, so git diffs stay readable without a separate build step. Its functions
can also be imported, and `scripts/prepare_data.py` uses this to share one implementation of the ChEMBL download
and curation code with the notebook.

## Credits and licence

This workshop is built on the [chemoinformatics tutorials](https://github.com/AxelRolov/chemoinformatics_tutorials)
(sessions 02–04). Those tutorials compile open teaching material from TeachOpenCADD, Pat Walters' *Practical
Cheminformatics Tutorials*, the Schwaller group's *AI for Chemistry* course and the MolSSI workshop. The data come
from ChEMBL (CC BY-SA 3.0) and OpenADMET (Apache-2.0). [`CREDITS.md`](CREDITS.md) lists every source. The workshop
was assembled with Claude (Anthropic) and reviewed by the instructor.

The compilation (text, exercises and sequencing) is licensed **CC BY 4.0**, and the code is licensed **MIT** (see
[`LICENSE`](LICENSE)). Data files keep their original licences.
