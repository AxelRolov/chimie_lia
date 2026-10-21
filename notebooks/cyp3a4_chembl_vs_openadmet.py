# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "altair>=5.4",
#     "marimo>=0.23",
#     "numpy>=1.26",
#     "pandas>=2.2",
#     "rdkit>=2024.9.1",
#     "requests>=2.31",
#     "scikit-learn>=1.5",
#     "umap-learn>=0.5.7",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="CYP3A4: ChEMBL vs OpenADMET")

with app.setup(hide_code=True):
    import csv
    import gzip
    import hashlib
    import io
    import json
    import math
    import os
    import re
    import threading
    import time
    import warnings
    import zipfile
    from datetime import datetime, timezone
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import numpy as np
    import pandas as pd
    import requests
    import umap
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import Descriptors, rdFMCS, rdFingerprintGenerator, rdMolDescriptors
    from rdkit.Chem.Draw import rdMolDraw2D
    from rdkit.Chem.MolStandardize import rdMolStandardize
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from sklearn.decomposition import PCA
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    RDLogger.DisableLog("rdApp.*")
    warnings.filterwarnings("ignore", module="umap")
    warnings.filterwarnings("ignore", message=".*n_jobs value.*")
    alt.data_transformers.disable_max_rows()

    # Where is this notebook running? A checkout of the repository (data/ next to notebooks/), molab, or a
    # script that imports the functions below. Cached teaching data come from the checkout when it exists,
    # otherwise from the repository on GitHub.
    try:
        NOTEBOOK_DIR = Path(__file__).resolve().parent     # `python notebook.py`, `marimo edit`, imports
    except NameError:
        NOTEBOOK_DIR = Path(mo.notebook_dir() or Path.cwd())
    REPO_ROOT = NOTEBOOK_DIR.parent
    IN_REPO = (REPO_ROOT / "data" / "provenance.json").is_file()
    WORK_DIR = REPO_ROOT if IN_REPO else NOTEBOOK_DIR
    OUTPUT_DIR = WORK_DIR / "outputs"
    LIVE_CACHE_DIR = WORK_DIR / "workshop_cache"

    REPO = "AxelRolov/chimie_lia"
    REPO_RAW = f"https://raw.githubusercontent.com/{REPO}/main"

    # ---- the target ---------------------------------------------------------------------------------
    TARGET_ID = "CHEMBL340"          # human cytochrome P450 3A4 in ChEMBL
    UNIPROT_ID = "P08684"            # ... and in UniProt
    CHEMBL_BASE = "https://www.ebi.ac.uk"
    CHEMBL_API = f"{CHEMBL_BASE}/chembl/api/data"
    ACTIVITY_QUERY = {"target_chembl_id": TARGET_ID, "standard_type": "IC50"}
    ACTIVITY_URL = f"{CHEMBL_API}/activity.json?target_chembl_id={TARGET_ID}&standard_type=IC50&limit=1000"
    UI_ACTIVITIES_URL = (
        "https://www.ebi.ac.uk/chembl/explore/activities/"
        "QUERYSTRING:target_chembl_id%3ACHEMBL340%20AND%20standard_type%3A%22IC50%22"
    )
    UI_ASSAYS_URL = "https://www.ebi.ac.uk/chembl/explore/assays/QUERYSTRING:target_chembl_id%3ACHEMBL340"

    # ---- OpenADMET CYP challenge, pinned to one revision of the dataset --------------------------------
    HF_DATASET = "openadmet/cyp-challenge-train-test"
    HF_REVISION = "146282d2ce1532f1803679a834a9bec9dac1e482"
    _HF_FILES = f"https://huggingface.co/datasets/{HF_DATASET}/resolve/{HF_REVISION}"
    OPENADMET_TRAIN_URL = f"{_HF_FILES}/cyp-challenge-TRAIN_inhibition.csv"
    OPENADMET_TEST_URL = f"{_HF_FILES}/cyp-challenge-TEST-BLINDED.csv"
    CYP3A4 = "CYP3A4_pIC50_direct_inhibition"
    OPENADMET_EXPECTED = {"rows": 4905, "cyp3a4_values": 2335}

    # ---- cached teaching data (prepared before class with scripts/prepare_data.py) ----------------------
    CACHED = {
        "openadmet_train": "data/openadmet/cyp-challenge-TRAIN_inhibition.csv",
        "openadmet_test": "data/openadmet/cyp-challenge-TEST-BLINDED.csv",
        "api_activities": "data/chembl/api/CHEMBL340_IC50_activities.jsonl.gz",
        "api_assays": "data/chembl/api/CHEMBL340_IC50_assays.jsonl.gz",
        "api_target": "data/chembl/api/CHEMBL340_target.json",
        "ui_activities": "data/chembl/manual/CHEMBL340_IC50_activities_web_export.zip",
        "ui_assays": "data/chembl/manual/CHEMBL340_assays_web_export.zip",
        "assay_review": "data/chembl/assay_review.csv",
        "provenance": "data/provenance.json",
    }

    # ---- analysis settings ---------------------------------------------------------------------------
    SEED = 42
    MAX_PER_SOURCE = 1000            # at most 1,000 unique structures per source on the 2D maps
    WEAK_PIC50 = 4.0                 # below this, potency estimates are extrapolated and imprecise
    FP_RADIUS, FP_BITS = 2, 2048     # binary Morgan fingerprint (ECFP4-like)
    FP_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)
    DESCRIPTORS = ["MW", "cLogP", "TPSA", "HBD", "HBA", "RotB", "AromRings", "FracSP3", "HeavyAtoms"]
    DESCRIPTOR_NAMES = {
        "MW": "molecular weight (Da)", "cLogP": "calculated logP (Crippen)", "TPSA": "topological polar surface area (Å²)",
        "HBD": "H-bond donors", "HBA": "H-bond acceptors", "RotB": "rotatable bonds", "AromRings": "aromatic rings",
        "FracSP3": "fraction of sp³ carbons", "HeavyAtoms": "heavy atoms",
    }
    SOURCES = ["ChEMBL", "OpenADMET", "both"]
    SOURCE_COLORS = ["#3b6fb6", "#e07b39", "#7d4fb8"]
    LFC = rdMolStandardize.LargestFragmentChooser(preferOrganic=True)
    UNCHARGER = rdMolStandardize.Uncharger()
    # numba (used by UMAP) crashes if two threads of one process run it at once, which happens when several
    # sessions share a `marimo run` server. One lock per process, stored on the umap module, serialises them.
    UMAP_LOCK = umap.__dict__.setdefault("_workshop_lock", threading.Lock())

    # ---- the schema both ChEMBL routes are mapped to --------------------------------------------------
    COMMON_COLUMNS = [
        "route", "activity_id", "record_key", "molecule_chembl_id", "smiles", "assay_chembl_id",
        "assay_description", "assay_type", "bao_label", "assay_organism", "confidence_score",
        "target_chembl_id", "target_organism", "target_type", "document_chembl_id", "document_year",
        "standard_type", "standard_relation", "standard_value", "standard_units", "pchembl_value",
        "data_validity_comment", "potential_duplicate", "activity_comment",
    ]
    UI_ACTIVITY_COLUMNS = {   # ChEMBL web-interface header -> common schema
        "Molecule ChEMBL ID": "molecule_chembl_id", "Smiles": "smiles", "Assay ChEMBL ID": "assay_chembl_id",
        "Assay Description": "assay_description", "Assay Type": "assay_type", "BAO Label": "bao_label",
        "Assay Organism": "assay_organism", "Target ChEMBL ID": "target_chembl_id",
        "Target Organism": "target_organism", "Target Type": "target_type",
        "Document ChEMBL ID": "document_chembl_id", "Document Year": "document_year",
        "Standard Type": "standard_type", "Standard Relation": "standard_relation",
        "Standard Value": "standard_value", "Standard Units": "standard_units", "pChEMBL Value": "pchembl_value",
        "Data Validity Comment": "data_validity_comment", "Potential Duplicate": "potential_duplicate",
        "Comment": "activity_comment",
    }
    UI_ASSAY_COLUMNS = {
        "Assay ChEMBL ID": "assay_chembl_id", "Description": "description", "BAO Format": "bao_label",
        "Confidence Score": "confidence_score", "Organism": "assay_organism",
    }

    # ---- curation policy (applied identically to every route), in the order the rules are checked -------
    CURATION_RULES = {
        "target": "target is not human CYP3A4 (CHEMBL340, single protein)",
        "confidence": "assay confidence score is not 9",
        "endpoint": "endpoint is not IC50",
        "assay_context": "assay context is not direct inhibition",
        "validity": "ChEMBL data-validity flag",
        "potential_duplicate": "ChEMBL potential duplicate (value re-reported from an earlier source)",
        "units": "units are not nM",
        "value": "no positive, finite value",
        "structure": "no structure (SMILES)",
        "relation": "relation missing or approximate (~)",
        "exact_duplicate": "exact duplicate record (same molecule, assay, document and value)",
    }
    FLIP_RELATION = {">": "<", ">=": "<=", "<": ">", "<=": ">="}   # IC50 > x  <=>  pIC50 < 9 - log10(x)

    # ---- assay-description keywords used to pre-classify ChEMBL assays (reviewed by the instructor) -----
    TDI_PATTERN = re.compile(
        r"time[- ]?dependent|mechanism[- ]based|irreversib|inactivat|\bTDI\b|IC50 shift|\bkinact\b"
        r"|pre-?incubat\w*[^.;]{0,80}?(?:with|in (?:the )?presence (?:or absence )?of)\s+(?:an? )?NADP"
        r"|pre-?treated with NADP|pre-?incubation of enzyme in the presence of NADP",
        re.I,
    )
    INDUCTION_PATTERN = re.compile(r"\binduc|\bactivation\b|mRNA|expression level|\bPXR\b", re.I)
    OTHER_ENDPOINT_PATTERN = re.compile(
        r"binding affinity|displacement|spectral|type II binding|metabolic stability|half[- ]life|clearance", re.I
    )
    CELLULAR_PATTERN = re.compile(r"hepatocyte(?!s? microsom)|\bTC5 cells|AsPC1|CHO cells|HepG2|HepaRG", re.I)
    INHIBITION_PATTERN = re.compile(r"inhib|IC50", re.I)
    PROBES = [
        ("midazolam", r"midazolam"), ("testosterone", r"testosteron"), ("nifedipine", r"nifedipin"),
        ("BFC", r"\bBFC\b|benzyloxy-? ?4-? ?\(?trifluoromethyl\)?-? ?coumarin|benzyloxy-trifluoromethylcoumarin"
                r"|benzoyloxy-4-trifluoromethyl"),
        ("BQ", r"\bBQ\b|\b7BQ\b|benzyloxyquinoline"), ("DBF", r"\bDBF\b|dibenzylfluorescein"),
        ("DEF", r"\bDEF\b|diethoxyfluorescein"), ("DBOMF", r"DBOMF"), ("BOMCC", r"BOMCC"),
        ("luciferin", r"luciferin|P450-Glo|CYP450-Glo|\bIPA\b"), ("benzyloxyresorufin", r"resorufin"),
        ("erythromycin", r"erythromycin"), ("atorvastatin", r"atorvastatin"), ("terfenadine", r"terfenadine"),
        ("felodipine", r"felodipine"), ("triazolam", r"triazolam"), ("alprazolam", r"alprazolam"),
        ("sorafenib", r"sorafenib"), ("quinine", r"quinine"), ("dextromethorphan", r"dextromethorphan"),
    ]
    FLUOROGENIC = {"BFC", "BQ", "DBF", "DEF", "DBOMF", "BOMCC", "benzyloxyresorufin"}

    # ---- molecules used in the demonstrations (isomeric SMILES from PubChem) ---------------------------
    REFERENCE_DRUGS = {
        "ketoconazole": ("CC(=O)N1CCN(CC1)C2=CC=C(C=C2)OC[C@@H]3CO[C@@](O3)(CN4C=CN=C4)C5=C(C=C(C=C5)Cl)Cl",
                         "strong CYP3A4 inhibitor (azole antifungal); the classic DDI perpetrator"),
        "itraconazole": ("CCC(C)N1C(=O)N(C=N1)C2=CC=C(C=C2)N3CCN(CC3)C4=CC=C(C=C4)OC[C@H]5CO[C@](O5)(CN6C=NC=N6)"
                         "C7=C(C=C(C=C7)Cl)Cl", "strong CYP3A4 inhibitor (triazole antifungal)"),
        "miconazole": ("C1=CC(=C(C=C1Cl)Cl)COC(CN2C=CN=C2)C3=C(C=C(C=C3)Cl)Cl", "azole antifungal, CYP inhibitor"),
        "ritonavir": ("CC(C)C1=NC(=CS1)CN(C)C(=O)N[C@@H](C(C)C)C(=O)N[C@@H](CC2=CC=CC=C2)C[C@@H]([C@H]"
                      "(CC3=CC=CC=C3)NC(=O)OCC4=CN=CS4)O", "strong CYP3A4 inhibitor, used as a pharmacokinetic booster"),
        "cobicistat": ("CC(C)C1=NC(=CS1)CN(C)C(=O)N[C@@H](CCN2CCOCC2)C(=O)N[C@H](CC[C@H](CC3=CC=CC=C3)NC(=O)"
                       "OCC4=CN=CS4)CC5=CC=CC=C5", "booster designed from ritonavir"),
        "verapamil": ("CC(C)C(CCCN(C)CCC1=CC(=C(C=C1)OC)OC)(C#N)C2=CC(=C(C=C2)OC)OC", "moderate CYP3A4 inhibitor"),
        "erythromycin": ("CC[C@@H]1[C@@]([C@@H]([C@H](C(=O)[C@@H](C[C@@]([C@@H]([C@H]([C@@H]([C@H](C(=O)O1)C)"
                         "O[C@H]2C[C@@]([C@H]([C@@H](O2)C)O)(C)OC)C)O[C@H]3[C@@H]([C@H](C[C@H](O3)C)N(C)C)O)(C)O)C)C)O)"
                         "(C)O", "macrolide; time-dependent CYP3A4 inhibitor and substrate"),
        "midazolam": ("CC1=NC=C2N1C3=C(C=C(C=C3)Cl)C(=NC2)C4=CC=CC=C4F", "sensitive CYP3A4 substrate: the clinical probe"),
        "alprazolam": ("CC1=NN=C2N1C3=C(C=C(C=C3)Cl)C(=NC2)C4=CC=CC=C4", "CYP3A4 substrate (triazolobenzodiazepine)"),
        "triazolam": ("CC1=NN=C2N1C3=C(C=C(C=C3)Cl)C(=NC2)C4=CC=CC=C4Cl", "CYP3A4 substrate"),
        "testosterone": ("C[C@]12CC[C@H]3[C@H]([C@@H]1CC[C@@H]2O)CCC4=CC(=O)CC[C@]34C",
                         "CYP3A4 probe substrate (6β-hydroxylation)"),
        "epitestosterone": ("C[C@]12CC[C@H]3[C@H]([C@@H]1CC[C@H]2O)CCC4=CC(=O)CC[C@]34C", "17-epimer of testosterone"),
        "progesterone": ("CC(=O)[C@H]1CC[C@@H]2[C@@]1(CC[C@H]3[C@H]2CCC4=CC(=O)CC[C@]34C)C", "steroid, CYP3A4 substrate"),
        "nifedipine": ("CC1=C(C(C(=C(N1)C)C(=O)OC)C2=CC=CC=C2[N+](=O)[O-])C(=O)OC", "CYP3A4 probe substrate"),
        "BFC": ("C1=CC=C(C=C1)COC2=CC3=C(C=C2)C(=CC(=O)O3)C(F)(F)F",
                "7-benzyloxy-4-trifluoromethylcoumarin, a fluorogenic CYP3A4 probe"),
    }


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/AxelRolov/chimie_lia/blob/main/notebooks/cyp3a4_chembl_vs_openadmet.py)

    # Exploring CYP3A4 inhibitors: ChEMBL versus OpenADMET

    **A two-hour hands-on chemoinformatics workshop** — RDKit, pandas, scikit-learn and UMAP in a reactive
    [marimo](https://marimo.io) notebook.

    > **The question of the day.** *How well does ChEMBL cover the chemical space of the OpenADMET compounds?*
    >
    > We answer it without training a single model: by cleaning two datasets with the same rules and comparing
    > them through **similarity**, **scaffolds**, **2D maps** and **outliers**.

    > **Where this comes from.** This workshop is built on the
    > [chemoinformatics tutorials](https://github.com/AxelRolov/chemoinformatics_tutorials) (sessions 02–04), which
    > compile and update open teaching material from **TeachOpenCADD** (Volkamer lab, CC BY 4.0), **Pat Walters'**
    > *Practical Cheminformatics Tutorials* (MIT), the **Schwaller group**'s *AI for Chemistry* course (MIT) and the
    > **MolSSI** cheminformatics workshop (MIT). Data: **ChEMBL** (EMBL-EBI, CC BY-SA 3.0) and the **OpenADMET CYP
    > inhibition challenge** (Octant / OpenADMET, Apache-2.0). Full credits are in
    > [`CREDITS.md`](https://github.com/AxelRolov/chimie_lia/blob/main/CREDITS.md). The workshop was assembled with
    > **Claude** (Anthropic) and reviewed by the instructor; errors are ours, not the original authors'.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Plan of the session

    | time | part | what you will have at the end |
    |---|---|---|
    | 0–10′ | [0 · Introduction](#intro) | CYP3A4, the two datasets, SMILES, IC50 and pIC50 |
    | 10–20′ | [1 · ChEMBL by hand](#manual) | a CSV exported from the ChEMBL website, read with pandas |
    | 20–30′ | [2 · ChEMBL by API](#api) | the same records retrieved programmatically — and proof that both routes agree |
    | 30–45′ | [3 · Load and clean](#clean) | OpenADMET loaded; valid, standardised structures; a missingness and duplicate report |
    | 45–60′ | [4 · Descriptors](#eda) | property and activity distributions compared between sources |
    | 60–80′ | [5 · Similarity](#similarity) | the nearest ChEMBL neighbour of every OpenADMET compound |
    | 80–95′ | [6 · Scaffolds](#scaffolds) | common, shared and novel Bemis–Murcko scaffolds |
    | 95–110′ | [7 · PCA and UMAP](#maps) | one joint map of both sources |
    | 110–120′ | [8 · Outliers](#outliers) and [9 · Answers](#answers) | unusual compounds, explained; evidence-based answers |

    There are **five short exercises** (✏️). Each has a hidden solution (🔑) — try first, then peek.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.callout(
        mo.md(r"""
    **New to marimo?** It is a *reactive* notebook: when you move a slider or edit a cell, every cell that depends
    on it re-runs automatically, so what you see is always consistent — no hidden state, no "run all cells again".

    - Run a cell with **Shift + Enter** (or the ▶ button). Code of the text cells is hidden; click a cell to see it.
    - A global variable can be defined in **one cell only**. In the exercise cells, prefix your own variables with
      an underscore (`_df`, `_n`) so they stay local to the cell.
    - Widgets (sliders, dropdowns, tables you can click) drive the analysis. Charts with a ⬚ cursor support
      **drag-to-select**: the molecules you select are drawn below the chart.
    """),
        kind="info",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.sidebar(
        [
            mo.md("### 🧪 CYP3A4 workshop\n*ChEMBL vs OpenADMET*"),
            mo.nav_menu(
                {
                    "#intro": "0 · Introduction",
                    "#manual": "1 · ChEMBL by hand",
                    "#api": "2 · ChEMBL by API",
                    "#clean": "3 · Load and clean",
                    "#eda": "4 · Descriptors",
                    "#similarity": "5 · Similarity",
                    "#scaffolds": "6 · Scaffolds",
                    "#maps": "7 · PCA and UMAP",
                    "#outliers": "8 · Outliers",
                    "#answers": "9 · Answers and export",
                },
                orientation="vertical",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    <a id="intro"></a>
    ## 0 · Introduction (0–10′)

    **CYP3A4** is the most important drug-metabolising enzyme of the human liver and gut: it oxidises a large share of
    marketed drugs. A compound that **inhibits** CYP3A4 slows down the clearance of every co-administered CYP3A4
    substrate — ketoconazole, for example, raises the exposure of the sedative midazolam more than tenfold. That is why
    CYP3A4 inhibition is measured early in every drug-discovery project.

    | | **ChEMBL** — `CHEMBL340` | **OpenADMET CYP challenge** |
    |---|---|---|
    | what | curated literature & patent data, many labs | one blind-challenge dataset, one lab (Octant) |
    | assays | hundreds of protocols: liver microsomes or recombinant enzyme, midazolam, testosterone, fluorogenic probes… | one protocol: recombinant CYP3A4, fluorogenic probe **DBOMF**, 12-point dose–response, Bayesian fit with credible intervals |
    | chemistry | medicinal-chemistry series chosen by many projects | Enamine diversity set + FDA-approved drugs |
    | endpoint | `IC50` with a relation (`=`, `>`, `<`) and units | `pIC50` with a 95 % interval and a standard deviation |

    > ⚠️ **Same protein ≠ interchangeable measurements.** An IC50 depends on the enzyme source, the probe substrate,
    > its concentration, the pre-incubation and the read-out. OpenADMET also separates **direct** inhibition from
    > **time-dependent** inhibition (TDI, pre-incubation with NADPH). In this workshop we compare **structures**
    > jointly, but we always keep **activities** separate per source and per assay. We never average them across sources.

    **SMILES** encodes a molecule as text (`c1ccccc1O` is phenol). **IC50** is the concentration that halves enzyme
    activity; because it spans many orders of magnitude we use

    $$\mathrm{pIC50} = -\log_{10}\big(\mathrm{IC50}\,[\mathrm{M}]\big) = 9 - \log_{10}\big(\mathrm{IC50}\,[\mathrm{nM}]\big).$$

    Higher pIC50 = more potent. Try the converter:
    """)
    return


@app.cell
def _():
    ic50_value = mo.ui.number(start=0.001, stop=1_000_000, step=0.001, value=1.0, label="IC50")
    ic50_unit = mo.ui.dropdown({"nM": 1.0, "µM": 1e3, "mM": 1e6}, value="µM", label="unit")
    mo.hstack([ic50_value, ic50_unit], justify="start", wrap=True, gap=1)
    return ic50_unit, ic50_value


@app.cell(hide_code=True)
def _(ic50_unit, ic50_value):
    _nM = float(ic50_value.value or 1) * ic50_unit.value
    _p = 9 - np.log10(_nM)
    _ticks = pd.DataFrame(
        {"pIC50": [3, 4, 5, 6, 7, 8, 9, 10],
         "label": ["1 mM", "100 µM", "10 µM", "1 µM", "100 nM", "10 nM", "1 nM", "0.1 nM"]}
    )
    _shade = alt.Chart(pd.DataFrame({"a": [2.0], "b": [WEAK_PIC50]})).mark_rect(color="#f2c94c", opacity=0.35).encode(
        x=alt.X("a:Q", scale=alt.Scale(domain=[2, 11]), title="pIC50"), x2="b:Q"
    )
    _tick_marks = alt.Chart(_ticks).mark_tick(color="#555", thickness=1, size=18).encode(x="pIC50:Q")
    _tick_text = alt.Chart(_ticks).mark_text(dy=20, fontSize=10, color="#555").encode(x="pIC50:Q", text="label:N")
    _you = alt.Chart(pd.DataFrame({"pIC50": [float(np.clip(_p, 2, 11))]})).mark_point(
        shape="triangle-down", size=260, filled=True, color="#c0392b", yOffset=-16
    ).encode(x="pIC50:Q")
    _ruler = (_shade + _tick_marks + _tick_text + _you).properties(width=520, height=70)
    mo.hstack(
        [
            mo.stat(f"{_p:.2f}", label="pIC50", caption=f"9 − log₁₀({_nM:,.4g} nM)", bordered=True),
            mo.vstack([_ruler, mo.md("<small>Shaded: pIC50 < 4 (IC50 > 100 µM) — weak, often extrapolated, "
                                     "interpret with caution.</small>")]),
        ],
        align="center",
    )
    return


@app.cell
def _():
    drug_picker = mo.ui.dropdown(options=list(REFERENCE_DRUGS), value="ketoconazole", label="Example molecule",
                                 searchable=True)
    drug_picker
    return (drug_picker,)


@app.cell
def _(drug_picker):
    smiles_box = mo.ui.text(value=REFERENCE_DRUGS[drug_picker.value][0], label="SMILES — edit it!", full_width=True)
    smiles_box
    return (smiles_box,)


@app.cell(hide_code=True)
def _(drug_picker, smiles_box):
    _mol = Chem.MolFromSmiles(smiles_box.value)
    if _mol is None:
        _out = mo.callout("RDKit cannot parse this SMILES — check brackets, ring-closure digits and valences.",
                          kind="danger")
    else:
        _role = REFERENCE_DRUGS[drug_picker.value][1] if smiles_box.value == REFERENCE_DRUGS[drug_picker.value][0] else "your molecule"
        _out = mo.hstack(
            [
                mol_grid([smiles_box.value], [drug_picker.value if _role != "your molecule" else "your molecule"],
                         per_row=1, size=(330, 250)),
                mo.md(
                    f"**{_role}**\n\n"
                    f"- formula: {rdMolDescriptors.CalcMolFormula(_mol)}\n"
                    f"- MW {Descriptors.MolWt(_mol):.1f} Da · cLogP {Descriptors.MolLogP(_mol):.2f}\n"
                    f"- {_mol.GetNumHeavyAtoms()} heavy atoms · "
                    f"{len(Chem.FindMolChiralCenters(_mol, includeUnassigned=True, useLegacyImplementation=False))} stereocentres\n"
                    f"- canonical SMILES (RDKit):\n\n  <small>`{Chem.MolToSmiles(_mol)}`</small>"
                ),
            ],
            justify="start", wrap=True,
            align="center",
            widths=[1, 1.3],
            gap=2,
        )
    _out
    return


@app.function
def mol_grid(smiles, legends=None, per_row=4, size=(210, 165), highlights=None):
    """Draw a list of SMILES as an SVG grid (RDKit). `highlights` is an optional list of atom-index lists."""
    svg = mol_grid_svg(smiles, legends, per_row, size, highlights)
    if svg is None:
        return mo.md("_No structures to draw._")
    svg = svg.replace("<svg ", "<svg style='max-width:100%;height:auto' ", 1)     # shrink to fit narrow screens
    return mo.Html(f'<div style="background:white;display:inline-block;border-radius:6px;max-width:100%">{svg}</div>')


@app.function
def mol_grid_svg(smiles, legends=None, per_row=4, size=(210, 165), highlights=None):
    """The SVG text behind `mol_grid` (None if no SMILES can be parsed)."""
    mols, labels, marks = [], [], []
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if mol is None:
            continue
        mols.append(mol)
        labels.append("" if legends is None else str(legends[i])[:42])
        marks.append(list(highlights[i]) if highlights is not None else [])
    if not mols:
        return None
    n_cols = min(per_row, len(mols))
    n_rows = math.ceil(len(mols) / n_cols)
    width, height = size
    drawer = rdMolDraw2D.MolDraw2DSVG(n_cols * width, n_rows * height, width, height)
    drawer.drawOptions().legendFontSize = 14
    drawer.drawOptions().drawMolsSameScale = False          # each molecule fills its own panel
    drawer.DrawMolecules(mols, legends=labels, highlightAtoms=marks if highlights is not None else None)
    drawer.FinishDrawing()
    return drawer.GetDrawingText().replace("<?xml version='1.0' encoding='iso-8859-1'?>", "")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Is `CHEMBL340` really human CYP3A4?

    Before downloading anything, verify the target: organism, UniProt accession and target type. A *protein family*
    or *protein complex* target would mix other enzymes into our data.
    """)
    return


@app.function
def chembl_get(url, params=None, retries=5, timeout=60):
    """GET a ChEMBL API URL and return the JSON, retrying on timeouts, rate limits and server errors."""
    problem = None
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            if response.status_code == 200:
                return response.json()
            problem = f"HTTP {response.status_code}"
            if response.status_code < 500 and response.status_code != 429:
                break                                   # a client error does not go away by retrying
        except requests.RequestException as exc:
            problem = type(exc).__name__
        time.sleep(min(2**attempt, 30))                 # 1, 2, 4, 8 … s back-off
    raise RuntimeError(f"ChEMBL request failed ({problem}) after {attempt + 1} attempt(s): {url}")


@app.function
def verify_target(target):
    """Check a ChEMBL target record against what we expect for human CYP3A4."""
    accessions = [c.get("accession") for c in target.get("target_components", [])]
    checks = [
        ("ChEMBL ID", target.get("target_chembl_id"), TARGET_ID),
        ("preferred name", target.get("pref_name"), "Cytochrome P450 3A4"),
        ("organism", target.get("organism"), "Homo sapiens"),
        ("target type", target.get("target_type"), "SINGLE PROTEIN"),
        ("UniProt accession(s)", ", ".join(map(str, accessions)), UNIPROT_ID),
    ]
    return pd.DataFrame(
        [{"check": name, "found": found, "expected": expected, "ok": "✅" if found == expected else "❌"}
         for name, found, expected in checks]
    )


@app.cell
def _():
    try:
        target_record = chembl_get(f"{CHEMBL_API}/target/{TARGET_ID}.json", retries=2, timeout=30)
        target_origin = "live ChEMBL API"
    except Exception as _exc:
        _bytes, _where = read_cached("api_target")
        target_record = json.loads(_bytes)
        target_origin = f"cached snapshot ({_where}) — the live API was unreachable: {_exc}"
    target_checks = verify_target(target_record)
    mo.vstack(
        [
            mo.md(f"Target record from: **{target_origin}**"),
            mo.ui.table(target_checks, selection=None, pagination=False),
            mo.callout("All checks passed — CHEMBL340 is the single human protein CYP3A4 (UniProt P08684).", kind="success")
            if (target_checks["ok"] == "✅").all()
            else mo.callout("A target check failed: stop and inspect the record before going on.", kind="danger"),
        ]
    )
    return


@app.function
def read_cached(key):
    """Bytes of a cached teaching file and a note saying where they came from (local checkout first, else GitHub)."""
    relative = CACHED[key]
    local = REPO_ROOT / relative
    if local.is_file():
        return local.read_bytes(), f"local copy `{relative}`"
    url = f"{REPO_RAW}/{relative}"
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    return response.content, f"repository snapshot `{relative}` (fetched from GitHub)"


@app.function
def sha256_hex(data):
    return hashlib.sha256(data).hexdigest()


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    <a id="manual"></a>
    ## 1 · ChEMBL option 1 — download by hand (10–20′)

    1. Open [ChEMBL](https://www.ebi.ac.uk/chembl/), type **CHEMBL340** in the search box and open the target
       report card.
    2. **Verify the target**: *Cytochrome P450 3A4*, type *SINGLE PROTEIN*, organism *Homo sapiens*, UniProt
       accession *P08684* (cross-references section).
    3. In *Associated Bioactivities*, open the activities of this target. In the activity browser, tick
       **Standard Type → IC50** in the left-hand filters. (Shortcut: [this link]({UI_ACTIVITIES_URL}) opens the
       filtered table directly — 13,887 records in ChEMBL 37.)
    4. Click **CSV** above the table. ChEMBL prepares the file on its server (≈ 30 s), then offers a **.zip**
       containing one `.csv`.
    5. The activity export lists the assay description but **not the assay confidence score**, which we need. Open
       the [assays of CHEMBL340]({UI_ASSAYS_URL}) and export them as CSV too: we will join the two tables on
       `Assay ChEMBL ID`.
    6. Drop both files below (zip or csv). No time, or no network? The notebook continues with a **cached export made
       the same way** — always labelled as such.

    *If a column you need is missing from an export (structures, for instance), export the related table — compounds
    from the "Related compounds" button, assays from the assay browser — and join it on `Molecule ChEMBL ID` or
    `Assay ChEMBL ID`, exactly as we do for confidence scores here.*
    """)
    return


@app.cell
def _():
    ui_activity_upload = mo.ui.file(filetypes=[".zip", ".csv", ".tsv", ".txt"], kind="area",
                                    label="Your ChEMBL **activity** export (zip or csv)")
    ui_assay_upload = mo.ui.file(filetypes=[".zip", ".csv", ".tsv", ".txt"], kind="area",
                                 label="Your ChEMBL **assay** export (optional)")
    mo.hstack([ui_activity_upload, ui_assay_upload], widths="equal", gap=1)
    return ui_activity_upload, ui_assay_upload


@app.function
def read_ui_export(data, filename=""):
    """Read a table exported from the ChEMBL web interface (csv/tsv, possibly zipped) as text, sniffing the delimiter.

    Returns the DataFrame (every cell a string, nothing converted) and a dict describing the file."""
    inner = filename
    if data[:2] == b"PK":                                          # zip archive, as ChEMBL delivers it
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            tables = [n for n in archive.namelist() if n.lower().endswith((".csv", ".tsv", ".txt"))]
            if len(tables) != 1:
                raise ValueError(f"expected one table inside the zip archive, found {tables}")
            inner = tables[0]
            data = archive.read(inner)
    text = data.decode("utf-8-sig")
    lines = text.splitlines()
    try:
        delimiter = csv.Sniffer().sniff(text[:20000], delimiters=";,\t").delimiter
    except csv.Error:                                              # fall back to the most frequent candidate
        delimiter = max(";,\t", key=lines[0].count)
    header = next(csv.reader([lines[0]], delimiter=delimiter))
    table = pd.read_csv(io.StringIO(text), sep=delimiter, dtype=str, keep_default_na=False)
    info = {
        "file": inner,
        "delimiter": {";": "semicolon ;", ",": "comma ,", "\t": "tab"}[delimiter],
        "rows": len(table),
        "columns": len(header),
        "duplicated headers": sorted({h for h in header if header.count(h) > 1}),
        "first lines": "\n".join(line[:300] for line in lines[:2]),
    }
    return table, info


@app.cell
def _():
    _b, ui_cached_origin = read_cached("ui_activities")
    ui_cached_activities, ui_cached_activities_info = read_ui_export(_b, Path(CACHED["ui_activities"]).name)
    _b2, _ = read_cached("ui_assays")
    ui_cached_assays, ui_cached_assays_info = read_ui_export(_b2, Path(CACHED["ui_assays"]).name)
    return (
        ui_cached_activities,
        ui_cached_activities_info,
        ui_cached_assays,
        ui_cached_assays_info,
        ui_cached_origin,
    )


@app.cell
def _(ui_activity_upload, ui_assay_upload):
    ui_upload_error = None
    ui_uploaded_activities = ui_uploaded_activities_info = None
    ui_uploaded_assays = ui_uploaded_assays_info = None
    try:
        if ui_activity_upload.value:
            ui_uploaded_activities, ui_uploaded_activities_info = read_ui_export(
                ui_activity_upload.contents(), ui_activity_upload.name()
            )
        if ui_assay_upload.value:
            ui_uploaded_assays, ui_uploaded_assays_info = read_ui_export(ui_assay_upload.contents(), ui_assay_upload.name())
    except Exception as _exc:
        ui_upload_error = f"Could not read the uploaded file: {_exc}"
    return (
        ui_upload_error,
        ui_uploaded_activities,
        ui_uploaded_activities_info,
        ui_uploaded_assays,
        ui_uploaded_assays_info,
    )


@app.cell(hide_code=True)
def _(
    ui_cached_activities,
    ui_cached_activities_info,
    ui_cached_assays,
    ui_cached_assays_info,
    ui_cached_origin,
    ui_upload_error,
    ui_uploaded_activities,
    ui_uploaded_activities_info,
    ui_uploaded_assays,
    ui_uploaded_assays_info,
):
    def _inspect(table, info, title, origin):
        _blanks = (table == "").sum()
        _nones = (table == "None").sum()
        _missing = (
            pd.DataFrame({"empty strings": _blanks, "the text 'None'": _nones})
            .query("`empty strings` > 0 or `the text 'None'` > 0")
            .sort_values("empty strings", ascending=False)
            .reset_index(names="column")
        )
        _relations = (
            table["Standard Relation"].replace("", "(empty)").value_counts().rename_axis("raw value").reset_index()
            if "Standard Relation" in table.columns
            else None
        )
        _dup = info["duplicated headers"]
        return mo.vstack(
            [
                mo.md(f"**{title}** — {origin}: `{info['file']}`"),
                mo.hstack(
                    [
                        mo.stat(info["delimiter"], label="delimiter", bordered=True),
                        mo.stat(f"{info['rows']:,}", label="rows", bordered=True),
                        mo.stat(str(info["columns"]), label="columns", bordered=True),
                    ],
                    justify="start", wrap=True,
                ),
                mo.md("The first two lines, exactly as they are in the file:"),
                mo.md(f"```text\n{info['first lines']}\n```"),
                mo.callout(
                    mo.md(f"Duplicated column names in the header: {', '.join(f'`{d}`' for d in _dup)}. "
                          "pandas silently renames the second one (`Compounds.1`) — check which is which before using it."),
                    kind="warn",
                )
                if _dup
                else mo.md(""),
                mo.md("**Missing values come in two disguises** — empty strings *and* the text `None`:"),
                mo.ui.table(_missing, selection=None, page_size=6),
                mo.md("**`Standard Relation` is quoted inside the cell** (`'='`, not `=`): it must be cleaned before "
                      "comparing it with the API.") if _relations is not None else mo.md(""),
                mo.ui.table(_relations, selection=None, pagination=False) if _relations is not None else mo.md(""),
                mo.accordion({f"All {info['columns']} column names": mo.md(", ".join(f"`{c}`" for c in table.columns))}),
            ]
        )

    _acts = (ui_uploaded_activities, ui_uploaded_activities_info, "your upload") if ui_uploaded_activities is not None else (
        ui_cached_activities, ui_cached_activities_info, f"cached snapshot, {ui_cached_origin}")
    _assays = (ui_uploaded_assays, ui_uploaded_assays_info, "your upload") if ui_uploaded_assays is not None else (
        ui_cached_assays, ui_cached_assays_info, "cached snapshot")
    mo.vstack(
        [
            mo.callout(ui_upload_error, kind="danger") if ui_upload_error else mo.md(""),
            mo.ui.tabs(
                {
                    "Activity export": _inspect(_acts[0], _acts[1], "Activity export", _acts[2]),
                    "Assay export": _inspect(_assays[0], _assays[1], "Assay export", _assays[2]),
                }
            ),
        ]
    )
    return


@app.function
def record_key(df):
    """Composite identity of an activity record, available in both routes (the web export has no activity ID)."""
    value = df["standard_value"].map(lambda v: "" if pd.isna(v) else repr(float(v)))
    parts = [df["molecule_chembl_id"], df["assay_chembl_id"], df["document_chembl_id"],
             df["standard_relation"].fillna(""), value, df["standard_units"].fillna("")]
    return parts[0].astype(str).str.cat([p.astype(str) for p in parts[1:]], sep="|")


@app.function
def normalize_ui(activities, assays):
    """Map a ChEMBL web export (activities + assays) to the common schema."""
    missing = [c for c in UI_ACTIVITY_COLUMNS if c not in activities.columns]
    if missing:
        raise ValueError("the activity export lacks " + ", ".join(missing)
                         + " — export all columns, or join the related compound/assay export by ChEMBL ID")
    if not {"Assay ChEMBL ID", "Confidence Score"} <= set(assays.columns):
        raise ValueError("the assay export needs 'Assay ChEMBL ID' and 'Confidence Score'")
    df = activities.rename(columns=UI_ACTIVITY_COLUMNS)[list(UI_ACTIVITY_COLUMNS.values())]
    df = df.replace({"": np.nan, "None": np.nan})                       # both disguises of "missing"
    df["standard_relation"] = df["standard_relation"].str.strip("'")    # "'='"  ->  "="
    for col in ("standard_value", "pchembl_value", "potential_duplicate", "document_year"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    confidence = assays.drop_duplicates("Assay ChEMBL ID").set_index("Assay ChEMBL ID")["Confidence Score"]
    df["confidence_score"] = pd.to_numeric(df["assay_chembl_id"].map(confidence), errors="coerce")  # join by ID
    df["activity_id"] = pd.array([pd.NA] * len(df), dtype="Int64")      # not present in the web export
    df["route"] = "manual"
    df["record_key"] = record_key(df)
    return df[COMMON_COLUMNS].reset_index(drop=True)


@app.function
def assay_meta_from_ui(assays):
    a = assays.replace({"": np.nan, "None": np.nan}).rename(columns=UI_ASSAY_COLUMNS)
    a["confidence_score"] = pd.to_numeric(a["confidence_score"], errors="coerce")
    a["description"] = a["description"].fillna("")
    return a[list(UI_ASSAY_COLUMNS.values())].drop_duplicates("assay_chembl_id").reset_index(drop=True)


@app.cell
def _(
    ui_cached_activities,
    ui_cached_assays,
    ui_uploaded_activities,
    ui_uploaded_assays,
):
    manual_cached_common = normalize_ui(ui_cached_activities, ui_cached_assays)
    manual_cached_assay_meta = assay_meta_from_ui(ui_cached_assays)

    manual_upload_common = manual_upload_assay_meta = None
    manual_upload_note = None
    if ui_uploaded_activities is not None:
        _assays = ui_uploaded_assays if ui_uploaded_assays is not None else ui_cached_assays
        try:
            manual_upload_common = normalize_ui(ui_uploaded_activities, _assays)
            manual_upload_assay_meta = assay_meta_from_ui(_assays)
            if ui_uploaded_assays is None:
                manual_upload_note = ("Your activity export was joined with the **cached** assay export — fine if both "
                                      "come from the same ChEMBL release.")
        except ValueError as _exc:
            manual_upload_note = f"Your upload could not be mapped to the common schema: {_exc}"

    _example = manual_cached_common.dropna(subset=["smiles"]).iloc[0]
    _mapping = pd.DataFrame(
        [{"web-export column": k, "common column": v, "example value": str(_example[v])[:60]}
         for k, v in UI_ACTIVITY_COLUMNS.items()]
        + [{"web-export column": "Confidence Score (assay export, joined on Assay ChEMBL ID)",
            "common column": "confidence_score", "example value": str(_example["confidence_score"])},
           {"web-export column": "— (not exported)", "common column": "activity_id", "example value": "<NA>"}]
    )
    mo.vstack(
        [
            mo.md("### Mapping the web export to a common schema\nBoth routes end up with the same columns, so "
                  "everything downstream is shared. Note the cleaned relation and the joined confidence score:"),
            mo.callout(mo.md(manual_upload_note), kind="info") if manual_upload_note else mo.md(""),
            mo.ui.table(_mapping, selection=None, page_size=8),
        ]
    )
    return (
        manual_cached_assay_meta,
        manual_cached_common,
        manual_upload_assay_meta,
        manual_upload_common,
    )


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    <a id="api"></a>
    ## 2 · ChEMBL option 2 — download with the API (20–30′)

    The same records are available from the ChEMBL REST API. One URL is one query:

    ```
    {ACTIVITY_URL}
    ```

    The answer is a JSON *page* of at most 1,000 records plus `page_meta` with the `total_count` and the URL of the
    `next` page. We follow `next` until it is empty, retry on timeouts, then fetch the **assay metadata** (confidence
    score, description, format) in batches of 100 assay IDs. The official
    [`chembl_webresource_client`](https://github.com/chembl/chembl_webresource_client) (used in session 03 of the
    tutorials) does the same paging for you; here we use plain `requests` so nothing is hidden.
    ([API documentation](https://chembl.gitbook.io/chembl-interface-documentation/web-services/chembl-data-web-services))
    """)
    return


@app.cell(hide_code=True)
def _():
    try:
        _page = chembl_get(f"{CHEMBL_API}/activity.json", {**ACTIVITY_QUERY, "limit": 2}, retries=2, timeout=20)
        _keys = ["activity_id", "molecule_chembl_id", "canonical_smiles", "standard_type", "standard_relation",
                 "standard_value", "standard_units", "pchembl_value", "assay_chembl_id", "assay_description",
                 "target_chembl_id", "target_organism", "document_chembl_id"]
        _api_peek = mo.vstack(
            [
                mo.md(f"**Live peek** (`limit=2`): ChEMBL reports **{_page['page_meta']['total_count']:,}** CYP3A4 IC50 "
                      "records. Here is `page_meta` and the first record:"),
                mo.hstack([mo.tree(_page["page_meta"]), mo.json({k: _page["activities"][0].get(k) for k in _keys})],
                          widths=[1, 2]),
            ]
        )
    except Exception as _exc:
        _api_peek = mo.callout(f"The live API is not reachable right now ({_exc}). The notebook continues with the "
                               "cached snapshot.", kind="warn")
    _api_peek
    return


@app.function
def fetch_activity_pages(query=None, page_size=1000, on_page=None):
    """Download every page of the ChEMBL activity endpoint for `query`; return the list of raw records."""
    query = query or ACTIVITY_QUERY
    first = chembl_get(f"{CHEMBL_API}/activity.json", {**query, "limit": page_size, "offset": 0})
    records, total = list(first["activities"]), first["page_meta"]["total_count"]
    next_url = first["page_meta"]["next"]
    if on_page:
        on_page(len(records), total)
    while next_url:
        page = chembl_get(CHEMBL_BASE + next_url)
        records.extend(page["activities"])
        next_url = page["page_meta"]["next"]
        if on_page:
            on_page(len(records), total)
    if len(records) != total:
        raise RuntimeError(f"ChEMBL announced {total} records but returned {len(records)}")
    return records


@app.function
def fetch_assays(assay_ids, batch_size=100, on_batch=None):
    """Assay metadata for a list of assay ChEMBL IDs, fetched in batches."""
    ids = sorted(set(assay_ids))
    assays = []
    for start in range(0, len(ids), batch_size):
        chunk = ids[start:start + batch_size]
        page = chembl_get(f"{CHEMBL_API}/assay.json", {"assay_chembl_id__in": ",".join(chunk), "limit": 1000})
        assays.extend(page["assays"])
        if on_batch:
            on_batch(len(assays), len(ids))
    missing = set(ids) - {a["assay_chembl_id"] for a in assays}
    if missing:
        raise RuntimeError(f"{len(missing)} assays were not returned, e.g. {sorted(missing)[:3]}")
    return assays


@app.function
def write_jsonl_gz(records, path):
    """Save raw API records as gzipped JSON lines (reproducible bytes: no timestamp in the gzip header)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(r, sort_keys=True) + "\n" for r in records).encode("utf-8")
    with open(path, "wb") as fh, gzip.GzipFile(fileobj=fh, mode="wb", mtime=0, filename="") as gz:
        gz.write(payload)
    return sha256_hex(path.read_bytes())


@app.function
def read_jsonl(data):
    """Raw records from (gzipped) JSON lines, as a DataFrame with the values exactly as the API sent them."""
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    return pd.DataFrame.from_records([json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()])


@app.cell
def _():
    api_fetch_button = mo.ui.run_button(
        label="⬇️ Download all CYP3A4 IC50 records + assay metadata from the ChEMBL API now (≈ 1–2 min)",
        kind="success",
    )
    api_fetch_button
    return (api_fetch_button,)


@app.cell
def _(api_fetch_button):
    live_api = None
    if not api_fetch_button.value:
        _msg = mo.md("_No live download in this session yet — the rest of the notebook uses the **cached API "
                     "snapshot** below. Click the button to fetch the current data yourself._")
    else:
        try:
            _status = chembl_get(f"{CHEMBL_API}/status.json", retries=2, timeout=30)
            _total = chembl_get(f"{CHEMBL_API}/activity.json", {**ACTIVITY_QUERY, "limit": 1})["page_meta"]["total_count"]
            with mo.status.progress_bar(total=math.ceil(_total / 1000), title="ChEMBL activity pages") as _bar:
                _records = fetch_activity_pages(on_page=lambda done, total: _bar.update())
            _ids = sorted({r["assay_chembl_id"] for r in _records})
            with mo.status.progress_bar(total=math.ceil(len(_ids) / 100), title="Assay metadata (batches of 100)") as _bar2:
                _assays = fetch_assays(_ids, on_batch=lambda done, total: _bar2.update())
            _target = chembl_get(f"{CHEMBL_API}/target/{TARGET_ID}.json")
            _stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            _sha_acts = write_jsonl_gz(_records, LIVE_CACHE_DIR / f"live_{_stamp}_activities.jsonl.gz")
            _sha_assays = write_jsonl_gz(_assays, LIVE_CACHE_DIR / f"live_{_stamp}_assays.jsonl.gz")
            live_api = {
                "activities": pd.DataFrame.from_records(_records),
                "assays": pd.DataFrame.from_records(_assays),
                "target": _target,
                "release": _status.get("chembl_db_version"),
                "retrieved": _stamp,
                "sha256": {"activities": _sha_acts, "assays": _sha_assays},
            }
            _msg = mo.callout(
                mo.md(f"Downloaded **{len(_records):,}** activity records and **{len(_assays):,}** assays from "
                      f"**{live_api['release']}** at {_stamp} (UTC). Raw files saved in `{LIVE_CACHE_DIR.name}/`. "
                      "Select *API · live download* below to use them."),
                kind="success",
            )
        except Exception as _exc:
            _msg = mo.callout(
                mo.md(f"**The live download failed** ({_exc}). Nothing was replaced: the notebook keeps using the "
                      "cached snapshot, which is identified below."),
                kind="danger",
            )
    _msg
    return (live_api,)


@app.cell
def _():
    _acts_bytes, api_cached_origin = read_cached("api_activities")
    _assay_bytes, _ = read_cached("api_assays")
    _target_bytes, _ = read_cached("api_target")
    _prov_bytes, _ = read_cached("provenance")
    provenance = json.loads(_prov_bytes)
    api_cached = {
        "activities": read_jsonl(_acts_bytes),
        "assays": read_jsonl(_assay_bytes),
        "target": json.loads(_target_bytes),
        "release": provenance["chembl"]["release"],
        "sha256": {"activities": sha256_hex(_acts_bytes), "assays": sha256_hex(_assay_bytes)},
    }
    return api_cached, api_cached_origin, provenance


@app.cell(hide_code=True)
def _(api_cached_origin, provenance):
    _rows = []
    for _key, _info in provenance["files"].items():
        _observed = sha256_hex(read_cached(_key)[0])         # recompute every checksum from the bytes we use
        _rows.append({"file": _info["path"], "rows / records": _info.get("records"),
                      "sha256 (recorded)": _info["sha256"][:16] + "…",
                      "check": "✅ matches" if _observed == _info["sha256"] else "❌ differs"})
    _c = provenance["chembl"]
    mo.vstack(
        [
            mo.md(
                f"### The cached snapshot and its provenance\n"
                f"- ChEMBL release **{_c['release']}** (released {_c['release_date']}), retrieved **{_c['retrieved_utc']}**\n"
                f"- API query: `{_c['activity_query_url']}` → **{_c['n_activity_records']:,}** activity records, "
                f"**{_c['n_assays']:,}** assays\n"
                f"- web exports: activities `{_c['web_export_activity_query']}`; assays `{_c['web_export_assay_query']}`\n"
                f"- OpenADMET: `{provenance['openadmet']['dataset']}` @ `{provenance['openadmet']['revision'][:12]}…`\n"
                f"- loaded from: {api_cached_origin}"
            ),
            mo.ui.table(pd.DataFrame(_rows), selection=None, pagination=False),
        ]
    )
    return


@app.function
def normalize_api(activities, assays, target):
    """Map raw API activity records (+ their assays and the target record) to the common schema."""
    a = assays.drop_duplicates("assay_chembl_id").set_index("assay_chembl_id")
    df = pd.DataFrame(
        {
            "route": "API",
            "activity_id": pd.to_numeric(activities["activity_id"]).astype("Int64"),
            "molecule_chembl_id": activities["molecule_chembl_id"],
            "smiles": activities["canonical_smiles"],
            "assay_chembl_id": activities["assay_chembl_id"],
            "assay_description": activities["assay_description"],
            "assay_type": activities["assay_type"],
            "bao_label": activities["bao_label"],
            "assay_organism": activities["assay_chembl_id"].map(a["assay_organism"]),
            "confidence_score": pd.to_numeric(activities["assay_chembl_id"].map(a["confidence_score"]), errors="coerce"),
            "target_chembl_id": activities["target_chembl_id"],
            "target_organism": activities["target_organism"],
            "target_type": activities["target_chembl_id"].map({target["target_chembl_id"]: target["target_type"]}),
            "document_chembl_id": activities["document_chembl_id"],
            "document_year": pd.to_numeric(activities["document_year"], errors="coerce"),
            "standard_type": activities["standard_type"],
            "standard_relation": activities["standard_relation"],
            "standard_value": pd.to_numeric(activities["standard_value"], errors="coerce"),
            "standard_units": activities["standard_units"],
            "pchembl_value": pd.to_numeric(activities["pchembl_value"], errors="coerce"),
            "data_validity_comment": activities["data_validity_comment"],
            "potential_duplicate": pd.to_numeric(activities["potential_duplicate"], errors="coerce"),
            "activity_comment": activities["activity_comment"],
        }
    )
    df["record_key"] = record_key(df)
    return df[COMMON_COLUMNS].reset_index(drop=True)


@app.function
def assay_meta_from_api(assays):
    return pd.DataFrame(
        {
            "assay_chembl_id": assays["assay_chembl_id"],
            "description": assays["description"].fillna(""),
            "bao_label": assays["bao_label"],
            "confidence_score": pd.to_numeric(assays["confidence_score"], errors="coerce"),
            "assay_organism": assays["assay_organism"],
        }
    ).drop_duplicates("assay_chembl_id").reset_index(drop=True)


@app.cell
def _(api_cached, live_api):
    api_cached_common = normalize_api(api_cached["activities"], api_cached["assays"], api_cached["target"])
    api_cached_assay_meta = assay_meta_from_api(api_cached["assays"])
    api_live_common = api_live_assay_meta = None
    if live_api is not None:
        api_live_common = normalize_api(live_api["activities"], live_api["assays"], live_api["target"])
        api_live_assay_meta = assay_meta_from_api(live_api["assays"])
    _review_bytes, _ = read_cached("assay_review")
    assay_review = pd.read_csv(io.BytesIO(_review_bytes))
    return (
        api_cached_assay_meta,
        api_cached_common,
        api_live_assay_meta,
        api_live_common,
        assay_review,
    )


@app.function
def compare_routes(curated, reference):
    """Compare curated tables from several routes with a reference route, by composite record key.

    `curated` maps a route label to its curated measurement table. Returns one summary row per route."""
    ref = curated[reference]
    ref_ids = ref.drop_duplicates("record_key").set_index("record_key")["activity_id"]
    ref_kept = ref.loc[ref["subset"] != "excluded", "record_key"].value_counts()
    ref_all = ref["record_key"].value_counts()
    rows = []
    for label, table in curated.items():
        kept = table.loc[table["subset"] != "excluded", "record_key"].value_counts()
        all_keys = table["record_key"].value_counts()
        # activity IDs of the retained records: the web export has none, so borrow them from the API through the key
        ids = table.loc[table["subset"] != "excluded", "activity_id"]
        if ids.isna().all():
            ids = table.loc[table["subset"] != "excluded", "record_key"].map(ref_ids)
        ref_kept_ids = set(ref.loc[ref["subset"] != "excluded", "activity_id"].dropna().astype(int))
        rows.append(
            {
                "route": label,
                "records": len(table),
                "quantitative": int((table["subset"] == "quantitative").sum()),
                "censored": int((table["subset"] == "censored").sum()),
                "excluded": int((table["subset"] == "excluded").sum()),
                "same raw records as reference": all_keys.sort_index().equals(ref_all.sort_index()),
                "same retained records": kept.sort_index().equals(ref_kept.sort_index()),
                "same retained activity IDs": set(ids.dropna().astype(int)) == ref_kept_ids,
            }
        )
    return pd.DataFrame(rows)


@app.cell
def _(
    api_cached_assay_meta,
    api_cached_common,
    api_live_assay_meta,
    api_live_common,
    assay_review,
    manual_cached_assay_meta,
    manual_cached_common,
    manual_upload_assay_meta,
    manual_upload_common,
):
    route_tables = {
        "API · cached snapshot": (api_cached_common, api_cached_assay_meta),
        "Manual web export · cached snapshot": (manual_cached_common, manual_cached_assay_meta),
    }
    if api_live_common is not None:
        route_tables["API · live download (this session)"] = (api_live_common, api_live_assay_meta)
    if manual_upload_common is not None:
        route_tables["Manual web export · your upload"] = (manual_upload_common, manual_upload_assay_meta)
    _curated = {k: curate_chembl(c, classify_assays(m, assay_review)) for k, (c, m) in route_tables.items()}
    route_comparison = compare_routes(_curated, "API · cached snapshot")
    _same = route_comparison.iloc[1][["same raw records as reference", "same retained records",
                                       "same retained activity IDs"]].all()
    mo.vstack(
        [
            mo.md("### Do both routes give the same data?\nEach route goes through **the same** normalisation and "
                  "curation functions. We compare records by their composite key "
                  "(molecule · assay · document · relation · value · units), because the web export has no activity ID; "
                  "retained activity IDs are then recovered through that key."),
            mo.ui.table(route_comparison.set_index("route").T.astype(str).reset_index(names="check"), selection=None,
                        pagination=False),
            mo.callout(
                "Manual export and API snapshot (same release, same query) agree record for record — after curation too."
                if _same else "The two cached routes differ — check the release and the export scope.",
                kind="success" if _same else "danger",
            ),
        ]
    )
    return route_comparison, route_tables


@app.cell
def _(route_tables):
    chembl_route = mo.ui.radio(
        list(route_tables),
        value="API · cached snapshot",
        label="**ChEMBL records used in the rest of the notebook** (any choice goes through the same pipeline):",
    )
    chembl_route
    return (chembl_route,)


@app.cell
def _(assay_review, chembl_route, route_tables):
    chembl_common, _meta = route_tables[chembl_route.value]
    chembl_assays = classify_assays(_meta, assay_review)
    chembl_route_label = chembl_route.value
    return chembl_assays, chembl_common, chembl_route_label


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    <a id="clean"></a>
    ## 3 · Load OpenADMET, then inspect and clean both sources (30–45′)

    We read the OpenADMET training file from a **pinned revision** on Hugging Face (dataset
    `openadmet/cyp-challenge-train-test`, config `default`, split `train`) and check its size *before* touching any
    structure.
    """)
    return


@app.cell
def _(provenance):
    try:
        _response = requests.get(OPENADMET_TRAIN_URL, timeout=60)
        _response.raise_for_status()
        _oa_bytes = _response.content
        openadmet_origin = f"pinned Hugging Face file, revision `{HF_REVISION[:12]}…`"
    except Exception as _exc:
        _oa_bytes, _where = read_cached("openadmet_train")
        openadmet_origin = f"⚠️ cached snapshot ({_where}) — the live download failed: {type(_exc).__name__}"
    openadmet_sha256 = sha256_hex(_oa_bytes)
    openadmet_raw = pd.read_csv(io.BytesIO(_oa_bytes))

    _checks = pd.DataFrame(
        [
            {"check": "rows (all four isoforms)", "found": len(openadmet_raw), "expected": OPENADMET_EXPECTED["rows"]},
            {"check": "rows with a CYP3A4 value", "found": int(openadmet_raw[CYP3A4].notna().sum()),
             "expected": OPENADMET_EXPECTED["cyp3a4_values"]},
            {"check": "sha256 = recorded snapshot", "found": openadmet_sha256[:16] + "…",
             "expected": provenance["files"]["openadmet_train"]["sha256"][:16] + "…"},
        ]
    )
    _checks["ok"] = ["✅" if str(f) == str(e) else "❌" for f, e in zip(_checks["found"], _checks["expected"])]
    mo.stop(
        (_checks["ok"] == "❌").iloc[:2].any(),
        mo.callout("The OpenADMET file does not have the expected size: stop and check the revision.", kind="danger"),
    )
    mo.vstack([mo.md(f"Loaded from: **{openadmet_origin}**"), mo.ui.table(_checks, selection=None, pagination=False)])
    return (openadmet_raw,)


@app.cell(hide_code=True)
def _(chembl_common, chembl_route_label, openadmet_raw):
    def _overview(df, name, id_col, smiles_col):
        _types = df.dtypes.astype(str).value_counts().to_dict()
        return mo.vstack(
            [
                mo.hstack(
                    [
                        mo.stat(f"{df.shape[0]:,} × {df.shape[1]}", label="shape", bordered=True),
                        mo.stat(f"{df[id_col].nunique():,}", label=f"unique {id_col}", bordered=True),
                        mo.stat(f"{df[smiles_col].nunique():,}", label="unique SMILES strings", bordered=True),
                    ],
                    justify="start", wrap=True,
                ),
                mo.md("column types: " + ", ".join(f"{v} × `{k}`" for k, v in _types.items())),
                mo.ui.table(df.head(50), selection=None, page_size=5),
            ]
        )

    mo.ui.tabs(
        {
            "OpenADMET (raw)": _overview(openadmet_raw, "OpenADMET", "Molecule_Name", "SMILES"),
            f"ChEMBL ({chembl_route_label})": _overview(chembl_common, "ChEMBL", "molecule_chembl_id", "smiles"),
        }
    )
    return


@app.cell(hide_code=True)
def _(chembl_common, openadmet_raw):
    _rows = []
    for _name, _df in (("OpenADMET", openadmet_raw), ("ChEMBL", chembl_common)):
        _frac = _df.isna().mean()
        _rows += [{"source": _name, "column": c, "missing": f} for c, f in _frac.items()]
    _miss = pd.DataFrame(_rows)
    _chart = alt.hconcat(
        *[
            alt.Chart(_miss[_miss["source"] == _name], title=_name)
            .mark_bar(color=_colour)
            .encode(
                y=alt.Y("column:N", sort="-x", title=None, axis=alt.Axis(labelLimit=230)),
                x=alt.X("missing:Q", axis=alt.Axis(format="%"), title="fraction missing", scale=alt.Scale(domain=[0, 1])),
                tooltip=["column", alt.Tooltip("missing:Q", format=".1%")],
            )
            .properties(width=110, height=alt.Step(12))
            for _name, _colour in zip(SOURCES[:2], SOURCE_COLORS[:2])
        ],
        spacing=10,
    )
    _iso = openadmet_raw.filter(like="_pIC50_direct_inhibition").filter(regex="^CYP[0-9A-Z]+_pIC50_direct_inhibition$")
    _pattern = (
        _iso.notna()
        .rename(columns=lambda c: c.split("_")[0])
        .apply(lambda r: " + ".join(c for c, v in r.items() if v) or "none", axis=1)
        .value_counts()
        .rename_axis("isoforms measured")
        .reset_index(name="molecules")
    )
    mo.vstack(
        [
            mo.md("### Missingness\nOpenADMET is a **sparse matrix**: each molecule was measured on some isoforms only. "
                  "An empty CYP3A4 cell means *not measured / not reported*, **not inactive**."),
            _chart,
            mo.md("Which isoform combinations were measured (OpenADMET training file)?"),
            mo.ui.table(_pattern, selection=None, page_size=8),
        ]
    )
    return


@app.cell
def _(openadmet_raw):
    oa_meas = (
        openadmet_raw.loc[
            openadmet_raw[CYP3A4].notna(),
            ["Molecule_Name", "SMILES", CYP3A4, f"{CYP3A4}_conf_low", f"{CYP3A4}_conf_high", f"{CYP3A4}_std"],
        ]
        .rename(
            columns={
                "Molecule_Name": "name", "SMILES": "smiles", CYP3A4: "pIC50",
                f"{CYP3A4}_conf_low": "pIC50_ci_low", f"{CYP3A4}_conf_high": "pIC50_ci_high",
                f"{CYP3A4}_std": "pIC50_std",
            }
        )
        .reset_index(drop=True)
    )
    oa_meas["ci_width"] = oa_meas["pIC50_ci_high"] - oa_meas["pIC50_ci_low"]
    oa_meas["weak_flag"] = oa_meas["pIC50"] < WEAK_PIC50       # flagged, never removed or clipped
    oa_meas["source"] = "OpenADMET"
    oa_meas["assay"] = "OpenADMET direct inhibition (recombinant CYP3A4, DBOMF)"
    return (oa_meas,)


@app.cell(hide_code=True)
def _(oa_meas):
    _d = oa_meas[["name", "pIC50", "pIC50_ci_low", "pIC50_ci_high", "ci_width", "weak_flag"]].copy()
    _d["flag"] = np.where(_d["weak_flag"], "pIC50 < 4 (cautious)", "pIC50 ≥ 4")
    fig_uncertainty = (
        alt.Chart(_d)
        .mark_circle(size=22, opacity=0.6)
        .encode(
            x=alt.X("pIC50:Q", title="OpenADMET CYP3A4 pIC50"),
            y=alt.Y("ci_width:Q", title="width of the 95 % interval (log units)"),
            color=alt.Color("flag:N", scale=alt.Scale(domain=["pIC50 ≥ 4", "pIC50 < 4 (cautious)"],
                                                      range=["#3b6fb6", "#d4a017"]), title=None),
            tooltip=["name", alt.Tooltip("pIC50:Q", format=".2f"), alt.Tooltip("pIC50_ci_low:Q", format=".2f"),
                     alt.Tooltip("pIC50_ci_high:Q", format=".2f")],
        )
        .properties(width=520, height=300, title="Uncertainty grows as potency drops")
    )
    oa_uncertainty_chart = mo.ui.altair_chart(fig_uncertainty, chart_selection="interval", legend_selection=True)
    mo.vstack(
        [
            mo.md(
                f"### OpenADMET CYP3A4 values and their uncertainty\n"
                f"**{len(oa_meas):,}** molecules have a CYP3A4 pIC50 (median {oa_meas['pIC50'].median():.2f}). "
                f"**{int(oa_meas['weak_flag'].sum()):,} ({oa_meas['weak_flag'].mean():.0%})** are below 4: those potencies "
                "lie below the lowest tested concentration, so they are extrapolated and their intervals are wide. "
                "We **flag** them — we do not clip, censor or delete them. *Drag over the chart to inspect points.*"
            ),
            oa_uncertainty_chart,
        ]
    )
    return fig_uncertainty, oa_uncertainty_chart


@app.cell(hide_code=True)
def _(oa_uncertainty_chart):
    _sel = oa_uncertainty_chart.value
    mo.ui.table(_sel.drop(columns=["flag"], errors="ignore").reset_index(drop=True), selection=None, page_size=5) if len(_sel) else mo.md(
        "_Select points on the chart to list them here._"
    )
    return


@app.cell(hide_code=True)
def _(chembl_meas, openadmet_raw):
    _no3a4 = openadmet_raw[openadmet_raw[CYP3A4].isna() & openadmet_raw.filter(regex="^CYP.*_direct_inhibition$").notna().any(axis=1)].iloc[0]
    _cens = chembl_meas[(chembl_meas["subset"] == "censored") & (chembl_meas["standard_relation"] == ">")].iloc[0]
    exercise1_q1 = mo.ui.radio(
        {
            "It is inactive on CYP3A4 (pIC50 below detection).": "a",
            "It was not measured / not reported for CYP3A4: we know nothing about it.": "b",
            "Its CYP3A4 pIC50 is 0.": "c",
        },
        label=f"**Q1.** `{_no3a4['Molecule_Name']}` has values for other isoforms but an **empty CYP3A4 cell**. What can you conclude?",
    )
    _bound = 9 - np.log10(_cens["standard_value"])
    exercise1_q2 = mo.ui.radio(
        {
            f"pIC50 = {_bound:.2f}": "a",
            f"pIC50 < {_bound:.2f}": "b",
            f"pIC50 > {_bound:.2f}": "c",
            "The record is useless and should be deleted.": "d",
        },
        label=(f"**Q2.** ChEMBL record `{_cens['molecule_chembl_id']}` in `{_cens['assay_chembl_id']}` says "
               f"**IC50 {_cens['standard_relation']} {_cens['standard_value']:,.0f} nM**. Which statement is correct?"),
    )
    mo.vstack(
        [
            mo.md("### ✏️ Exercise 1 — a missing value versus a censored measurement"),
            exercise1_q1,
            exercise1_q2,
        ]
    )
    return exercise1_q1, exercise1_q2


@app.cell(hide_code=True)
def _(exercise1_q1, exercise1_q2):
    _fb = []
    if exercise1_q1.value:
        _fb.append(
            mo.callout("Q1 ✔ Right: missing means *not measured*. Treating it as inactive would invent data.", kind="success")
            if exercise1_q1.value == "b"
            else mo.callout("Q1 ✘ Not quite — a blank cell carries no information about potency.", kind="danger")
        )
    if exercise1_q2.value:
        _fb.append(
            mo.callout("Q2 ✔ Right: IC50 above a value means pIC50 *below* the corresponding bound. It is a real, "
                       "informative measurement — just not an exact number. We keep it in a separate censored table.",
                       kind="success")
            if exercise1_q2.value == "b"
            else mo.callout("Q2 ✘ Remember pIC50 = 9 − log₁₀(IC50 / nM): a *larger* IC50 gives a *smaller* pIC50, and "
                            "an inequality stays an inequality.", kind="danger")
        )
    mo.vstack(
        _fb
        + [
            mo.md("**Then in code:** count the OpenADMET molecules with no CYP3A4 value but at least one other isoform "
                  "value, and count the ChEMBL censored records per relation (`chembl_meas`, column `subset`)."),
        ]
    )
    return


@app.cell
def _():
    # ✏️ Exercise 1 — your code here (prefix your variables with _ so they stay local to this cell)
    return


@app.cell(hide_code=True)
def _():
    mo.accordion(
        {
            "🔑 Solution — exercise 1": mo.md(r"""
    ```python
    _iso = openadmet_raw.filter(regex="^CYP.*_pIC50_direct_inhibition$")
    _other = _iso.drop(columns=[CYP3A4]).notna().any(axis=1)
    print("no CYP3A4 value but another isoform:", int((openadmet_raw[CYP3A4].isna() & _other).sum()))

    print(chembl_meas.loc[chembl_meas["subset"] == "censored", "standard_relation"].value_counts())
    ```
    About 2,570 OpenADMET molecules lack a CYP3A4 value: they were simply not measured on CYP3A4 (most come from the
    other isoforms' screens). In ChEMBL, the censored records are mostly `>` (e.g. "IC50 > 30 µM": the compound was
    weaker than the top concentration). Both are kept, but neither is turned into a number.
    """)
        }
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Curating ChEMBL: one documented policy

    Every record is checked against the rules below, in this order; the **first** rule it fails becomes its
    `exclusion_reason`. Nothing is deleted from the measurement table — excluded records just get a reason.

    | rule | why |
    |---|---|
    | target = `CHEMBL340`, *SINGLE PROTEIN*, *Homo sapiens* | the verified human enzyme |
    | assay **confidence score 9** | ChEMBL is sure the assay measures this single protein |
    | endpoint **IC50** | one endpoint only |
    | assay context = **direct inhibition** | no time-dependent inhibition, induction, other endpoints, whole cells (see the assay review) |
    | no data-validity flag, not a *potential duplicate* | ChEMBL's own warnings |
    | units **nM**, value positive and finite, a structure | needed for pIC50 and for chemistry |
    | relation `=` → **quantitative**; `>`, `<`, `≥`, `≤` → **censored** (kept separately) | limits are never treated as exact values |
    | an exact duplicate record (same molecule, assay, document, value) is kept once | the same number entered twice is not two measurements |

    We do **not** require `assay_type = B`: most CYP inhibition assays are classified **A** (ADME) in ChEMBL.
    """)
    return


@app.function
def classify_assay(description, bao_label=None):
    """Keyword pre-classification of a ChEMBL CYP3A4 assay: ('include' | 'exclude', reason)."""
    if TDI_PATTERN.search(description):
        return "exclude", "time-dependent inhibition or NADPH pre-incubation"
    if INDUCTION_PATTERN.search(description):
        return "exclude", "induction / activation / expression"
    if OTHER_ENDPOINT_PATTERN.search(description):
        return "exclude", "other endpoint (binding, stability)"
    if CELLULAR_PATTERN.search(description) or bao_label == "organism-based format":
        return "exclude", "ambiguous: whole-cell or in vivo context"
    if not INHIBITION_PATTERN.search(description):
        return "exclude", "ambiguous: no inhibition endpoint stated"
    return "include", "direct inhibition"


@app.function
def enzyme_source(description):
    """Enzyme preparation named in an assay description."""
    if re.search(r"recombinant|supersome|baculo|expressed in|cDNA|insect|E\. ?coli|Escherichia|yeast|bactosome", description, re.I):
        return "recombinant enzyme"
    if re.search(r"microsom", description, re.I) and re.search(r"liver|hepat", description, re.I) or re.search(r"\bHLM\b", description):
        return "human liver microsomes"
    if re.search(r"hepatocyte", description, re.I):
        return "hepatocytes"
    return "not stated"


@app.function
def probe_substrate(description):
    """Probe substrate named in an assay description ('multiple' if several, 'not stated' if none)."""
    hits = [name for name, pattern in PROBES if re.search(pattern, description, re.I)]
    return hits[0] if len(hits) == 1 else ("multiple" if hits else "not stated")


@app.function
def probe_family(probe):
    if probe in FLUOROGENIC:
        return "fluorogenic probe"
    if probe == "luciferin":
        return "luminogenic probe"
    if probe in ("midazolam", "testosterone", "multiple", "not stated"):
        return {"multiple": "several probes", "not stated": "probe not stated"}.get(probe, probe)
    return "other drug probe"


@app.function
def classify_assays(meta, review=None):
    """Classify every assay with the keyword rules, then apply the instructor's reviewed decisions."""
    out = meta.drop_duplicates("assay_chembl_id").copy()
    rules = [classify_assay(d, b) for d, b in zip(out["description"], out["bao_label"])]
    out["rule_decision"] = [r[0] for r in rules]
    out["rule_reason"] = [r[1] for r in rules]
    out["enzyme_source"] = out["description"].map(enzyme_source)
    out["probe_substrate"] = out["description"].map(probe_substrate)
    out["probe_family"] = out["probe_substrate"].map(probe_family)
    out["decision"], out["decision_reason"] = out["rule_decision"], out["rule_reason"]
    out["decision_basis"] = "keyword rules"
    if review is not None and len(review):
        reviewed = review[review["reviewed"].astype(str).str.lower() == "true"].set_index("assay_chembl_id")
        hit = out["assay_chembl_id"].isin(reviewed.index)
        out.loc[hit, "decision"] = out.loc[hit, "assay_chembl_id"].map(reviewed["review_decision"])
        out.loc[hit, "decision_reason"] = out.loc[hit, "assay_chembl_id"].map(reviewed["review_reason"])
        out.loc[hit, "decision_basis"] = "instructor review"
    return out.reset_index(drop=True)


@app.function
def curate_chembl(common, assays, skip=()):
    """Apply the curation policy to common-schema records. Adds `exclusion_reason`, `subset` and pIC50 columns.

    `skip` lists rule keys (see CURATION_RULES) to switch off — only for the what-if explorer."""
    df = common.copy()
    a = assays.set_index("assay_chembl_id")
    for col in ("enzyme_source", "probe_substrate", "probe_family", "decision", "decision_reason", "decision_basis"):
        df[col] = df["assay_chembl_id"].map(a[col])
    value = df["standard_value"]
    failed = {
        "target": ~(df["target_chembl_id"].eq(TARGET_ID) & df["target_type"].eq("SINGLE PROTEIN")
                    & df["target_organism"].eq("Homo sapiens")),
        "confidence": df["confidence_score"].ne(9),
        "endpoint": df["standard_type"].ne("IC50"),
        "assay_context": df["decision"].ne("include"),
        "validity": df["data_validity_comment"].notna(),
        "potential_duplicate": df["potential_duplicate"].eq(1),
        "units": df["standard_units"].ne("nM"),
        "value": ~(value.gt(0) & np.isfinite(value)),
        "structure": df["smiles"].isna(),
        "relation": ~df["standard_relation"].isin(["=", ">", ">=", "<", "<="]),
    }
    reason = pd.Series(None, index=df.index, dtype=object)
    for key, mask in failed.items():
        if key in skip:
            continue
        new = mask.fillna(True).astype(bool) & reason.isna()
        if key == "assay_context":
            reason[new] = "assay context: " + df.loc[new, "decision_reason"].fillna("assay not classified")
        elif key == "validity":
            reason[new] = "ChEMBL data-validity flag: " + df.loc[new, "data_validity_comment"].astype(str)
        else:
            reason[new] = CURATION_RULES[key]
    if "exact_duplicate" not in skip:
        reason[reason.isna() & df["record_key"].duplicated()] = CURATION_RULES["exact_duplicate"]
    df["exclusion_reason"] = reason
    quantitative = reason.isna() & df["standard_relation"].eq("=")
    censored = reason.isna() & ~df["standard_relation"].eq("=")
    df["subset"] = np.select([quantitative, censored], ["quantitative", "censored"], default="excluded")
    p = 9 - np.log10(value.where(value > 0))                    # pIC50 = 9 - log10(IC50 / nM)
    df["pIC50"] = p.where(quantitative)
    df["pIC50_bound"] = p.where(censored)                      # a limit, never an exact value
    df["pIC50_relation"] = np.select(
        [quantitative, censored], ["=", df["standard_relation"].map(FLIP_RELATION).fillna("?")], default=""
    )
    df["weak_flag"] = df["pIC50"].lt(WEAK_PIC50)
    return df


@app.function
def curation_funnel(measurements):
    """Records removed by each rule (first failing rule), in policy order."""
    reasons = measurements["exclusion_reason"].fillna("")
    rows, remaining = [{"step": "records in the download", "removed": "", "remaining": len(measurements)}], len(measurements)
    for key, label in CURATION_RULES.items():
        if key == "assay_context":
            removed = int(reasons.str.startswith("assay context").sum())
        elif key == "validity":
            removed = int(reasons.str.startswith("ChEMBL data-validity flag").sum())
        else:
            removed = int((reasons == label).sum())
        remaining -= removed
        rows.append({"step": f"− {label}", "removed": f"{removed:,}", "remaining": remaining})
    rows.append({"step": "= quantitative (relation '=')", "removed": "",
                 "remaining": int((measurements["subset"] == "quantitative").sum())})
    rows.append({"step": "= censored (>, <, ≥, ≤), kept separately", "removed": "",
                 "remaining": int((measurements["subset"] == "censored").sum())})
    return pd.DataFrame(rows)


@app.cell
def _(chembl_assays, chembl_common):
    chembl_meas = curate_chembl(chembl_common, chembl_assays)
    return (chembl_meas,)


@app.cell(hide_code=True)
def _(chembl_meas, chembl_route_label):
    _q = chembl_meas[chembl_meas["subset"] == "quantitative"]
    _c = chembl_meas[chembl_meas["subset"] == "censored"]
    mo.vstack(
        [
            mo.md(f"#### Result of the policy on *{chembl_route_label}*"),
            mo.hstack(
                [
                    mo.stat(f"{len(chembl_meas):,}", label="records", bordered=True),
                    mo.stat(f"{len(_q):,}", label="quantitative", caption=f"{_q['molecule_chembl_id'].nunique():,} molecules · {_q['assay_chembl_id'].nunique():,} assays", bordered=True),
                    mo.stat(f"{len(_c):,}", label="censored (kept apart)", bordered=True),
                    mo.stat(f"{int(_q['weak_flag'].sum()):,}", label="quantitative with pIC50 < 4", bordered=True),
                ],
                justify="start", wrap=True,
            ),
            mo.ui.table(curation_funnel(chembl_meas), selection=None, pagination=False),
        ]
    )
    return


@app.cell
def _():
    assay_view_filter = mo.ui.dropdown(
        ["all assays", "included", "excluded", "instructor-reviewed"], value="all assays", label="Show"
    )
    assay_view_filter
    return (assay_view_filter,)


@app.cell(hide_code=True)
def _(assay_view_filter, chembl_assays, chembl_meas):
    _n = chembl_meas.groupby("assay_chembl_id").size().rename("IC50 records")
    _v = chembl_assays.join(_n, on="assay_chembl_id").dropna(subset=["IC50 records"])
    _v = _v[_v["confidence_score"] == 9]
    if assay_view_filter.value == "included":
        _v = _v[_v["decision"] == "include"]
    elif assay_view_filter.value == "excluded":
        _v = _v[_v["decision"] == "exclude"]
    elif assay_view_filter.value == "instructor-reviewed":
        _v = _v[_v["decision_basis"] == "instructor review"]
    _v = _v.sort_values("IC50 records", ascending=False).reset_index(drop=True)[
        ["assay_chembl_id", "IC50 records", "decision", "decision_reason", "decision_basis", "enzyme_source",
         "probe_substrate", "description"]
    ]
    _reasons = chembl_assays[chembl_assays["confidence_score"] == 9].groupby(["decision", "decision_reason"]).size().rename("assays").reset_index()
    mo.vstack(
        [
            mo.md("#### The assay review (confidence-9 assays with IC50 records)\nKeyword rules pre-classify every "
                  "assay description; the instructor read the largest assays before class (`decision_basis`). "
                  "Use the search box and the column filters to find, e.g., *time dependent* or *hepatocyte*."),
            mo.ui.table(_reasons, selection=None, pagination=False),
            mo.ui.table(_v, selection=None, page_size=8),
        ]
    )
    return


@app.cell
def _():
    whatif_rules = mo.ui.multiselect(
        options={label: key for key, label in CURATION_RULES.items()},
        value=list(CURATION_RULES.values()),
        label="**What if?** Rules switched on (remove one to see what it does):",
        full_width=True,
    )
    whatif_rules
    return (whatif_rules,)


@app.cell(hide_code=True)
def _(chembl_assays, chembl_common, chembl_meas, whatif_rules):
    _skip = set(CURATION_RULES) - set(whatif_rules.value)
    _alt = curate_chembl(chembl_common, chembl_assays, skip=_skip)

    def _summary(df):
        return {s: int((df["subset"] == s).sum()) for s in ("quantitative", "censored", "excluded")} | {
            "molecules (quantitative)": int(df.loc[df["subset"] == "quantitative", "molecule_chembl_id"].nunique())}

    _tab = pd.DataFrame({"workshop policy": _summary(chembl_meas), "your what-if": _summary(_alt)})
    _tab["difference"] = _tab["your what-if"] - _tab["workshop policy"]
    mo.vstack(
        [
            mo.md(f"Rules switched **off**: {', '.join(CURATION_RULES[k] for k in _skip) if _skip else '_none_'}. "
                  "The workshop policy itself never changes — the rest of the notebook keeps using it."),
            mo.ui.table(_tab.reset_index(names="count"), selection=None, pagination=False),
        ]
    )
    return


@app.cell(hide_code=True)
def _(chembl_meas):
    _q = chembl_meas[chembl_meas["subset"] == "quantitative"]
    _with_pchembl = _q.dropna(subset=["pchembl_value"])
    _max_diff = float((_with_pchembl["pIC50"] - _with_pchembl["pchembl_value"]).abs().max())
    _examples = pd.DataFrame({"IC50": ["1 nM", "1 µM = 1,000 nM", "100 µM = 100,000 nM"], "nM": [1, 1e3, 1e5]})
    _examples["pIC50 = 9 − log10(nM)"] = 9 - np.log10(_examples["nM"])
    _ok = _q["standard_units"].eq("nM").all() and _max_diff < 0.01
    mo.vstack(
        [
            mo.md("#### Unit-conversion checks"),
            mo.ui.table(_examples, selection=None, pagination=False),
            mo.callout(
                mo.md(f"All {len(_q):,} quantitative values are in **nM**, and our pIC50 agrees with ChEMBL's own "
                      f"`pchembl_value` on the {len(_with_pchembl):,} records that have one (largest difference "
                      f"{_max_diff:.3f}, i.e. rounding to two decimals)."),
                kind="success" if _ok else "danger",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### One structure-standardisation policy for both sources

    Same RDKit steps for ChEMBL and OpenADMET (as in session 04 of the tutorials):
    `Cleanup` (normalise functional groups, reionise) → keep the **largest organic fragment** (drops counter-ions and
    solvents) → **neutralise** → canonical **isomeric** SMILES. Stereochemistry is **kept**; tautomers are **not**
    canonicalised (slow, and a modelling choice we do not need today). The original SMILES are preserved next to the
    standardised ones, and every change is recorded.

    **Identity** of two structures = identical standardised isomeric SMILES — *not* a fingerprint similarity of 1.
    """)
    return


@app.function
def standardize_smiles(smiles):
    """Standardise one SMILES with the workshop policy; returns the new SMILES, a status and what changed."""
    if not isinstance(smiles, str) or not smiles.strip():
        return {"smiles_std": None, "std_status": "missing", "std_changes": ""}
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"smiles_std": None, "std_status": "invalid SMILES", "std_changes": ""}
    before = Chem.MolToSmiles(mol)
    n_fragments = len(Chem.GetMolFrags(mol))
    try:
        clean = rdMolStandardize.Cleanup(mol)
        parent = LFC.choose(clean)
        neutral = UNCHARGER.uncharge(parent)
    except Exception as exc:
        return {"smiles_std": None, "std_status": "standardisation failed", "std_changes": str(exc)[:80]}
    if neutral.GetNumAtoms() == 0:
        return {"smiles_std": None, "std_status": "no organic fragment", "std_changes": ""}
    std = Chem.MolToSmiles(neutral)
    changes = []
    if n_fragments > 1:
        changes.append(f"kept largest of {n_fragments} fragments")
    if Chem.MolToSmiles(clean) != before:
        changes.append("normalised groups")
    if Chem.MolToSmiles(neutral) != Chem.MolToSmiles(parent):
        changes.append("neutralised")
    return {"smiles_std": std, "std_status": "changed" if std != before else "unchanged", "std_changes": "; ".join(changes)}


@app.function
def standardize_many(smiles, title="Standardising structures"):
    """Standardise every distinct SMILES once; returns a lookup table indexed by the original SMILES."""
    unique = pd.unique(pd.Series(smiles).dropna())
    rows = [{"smiles": s, **standardize_smiles(s)} for s in mo.status.progress_bar(unique, title=title, remove_on_exit=True)]
    return pd.DataFrame(rows).set_index("smiles")


@app.cell
def _(chembl_meas, oa_meas):
    std_lookup = standardize_many(pd.concat([chembl_meas["smiles"], oa_meas["smiles"]]))
    chembl_std = chembl_meas.join(std_lookup, on="smiles")
    oa_std = oa_meas.join(std_lookup, on="smiles")
    return chembl_std, oa_std


@app.cell(hide_code=True)
def _(chembl_std, oa_std):
    def _report(df, name):
        _s = df.drop_duplicates("smiles")
        _changes = _s["std_changes"].fillna("").str.split("; ").explode()
        return {
            "source": name,
            "records with a SMILES": int(df["smiles"].notna().sum()),
            "distinct input SMILES": int(df["smiles"].nunique()),
            "invalid / failed": int(_s["smiles_std"].isna().sum() - _s["smiles"].isna().sum()),
            "fragments removed": int((_changes.str.startswith("kept largest")).sum()),
            "groups normalised": int((_changes == "normalised groups").sum()),
            "neutralised": int((_changes == "neutralised").sum()),
            "unique standardised structures": int(df["smiles_std"].nunique()),
        }

    _rep = pd.DataFrame([_report(chembl_std, "ChEMBL (all records)"), _report(oa_std, "OpenADMET (CYP3A4)")]).set_index("source").T
    mo.vstack([mo.md("#### What standardisation did"), mo.ui.table(_rep.reset_index(names="count"), selection=None, pagination=False)])
    return


@app.cell
def _(chembl_std, oa_std):
    _changed = pd.concat(
        [
            chembl_std.loc[chembl_std["std_status"] == "changed", ["molecule_chembl_id", "smiles", "smiles_std", "std_changes"]]
            .rename(columns={"molecule_chembl_id": "id"}).assign(source="ChEMBL"),
            oa_std.loc[oa_std["std_status"] == "changed", ["name", "smiles", "smiles_std", "std_changes"]]
            .rename(columns={"name": "id"}).assign(source="OpenADMET"),
        ]
    ).drop_duplicates("smiles")
    std_examples_table = mo.ui.table(_changed.reset_index(drop=True), selection="single", page_size=5,
                                     label="Structures changed by standardisation — click a row to see before/after")
    std_examples_table
    return (std_examples_table,)


@app.cell(hide_code=True)
def _(std_examples_table):
    _sel = std_examples_table.value
    if len(_sel):
        _r = _sel.iloc[0]
        _out = mol_grid([_r["smiles"], _r["smiles_std"]], ["original", "standardised"], per_row=2, size=(300, 220))
    else:
        _out = mo.md("_Click a row above to compare the original and the standardised structure._")
    _out
    return


@app.function
def build_structure_table(chembl, openadmet):
    """One row per unique standardised structure, with source membership and source-specific activity summaries.

    ChEMBL reference sets, from strict to loose: `ref_quantitative` (≥ 1 quantitative record), `ref_curated`
    (quantitative or censored — the workshop default), `ref_broad` (any confidence-9 human CYP3A4 IC50 record,
    whatever the assay context) and `ref_any` (any CHEMBL340 IC50 record, including lower assay confidence)."""
    ok = chembl[chembl["smiles_std"].notna()]
    first_three = {CURATION_RULES[k] for k in ("target", "confidence", "endpoint")}
    broad = ok[~ok["exclusion_reason"].isin(first_three)]
    curated = ok[ok["subset"].isin(["quantitative", "censored"])]
    quant = ok[ok["subset"] == "quantitative"]
    chem = ok.groupby("smiles_std").agg(chembl_ids=("molecule_chembl_id", lambda s: ", ".join(sorted(set(s)))))
    chem["ref_any"] = True
    chem["ref_broad"] = chem.index.isin(broad["smiles_std"])
    chem["ref_curated"] = chem.index.isin(curated["smiles_std"])
    chem["ref_quantitative"] = chem.index.isin(quant["smiles_std"])
    q = quant.groupby("smiles_std")
    chem["chembl_median_pIC50"] = q["pIC50"].median()
    chem["chembl_n_quant"] = q.size()
    chem["chembl_n_assays"] = q["assay_chembl_id"].nunique()
    chem["chembl_n_censored"] = curated[curated["subset"] == "censored"].groupby("smiles_std").size()
    oa = openadmet[openadmet["smiles_std"].notna()].groupby("smiles_std").agg(
        openadmet_names=("name", ", ".join), openadmet_pIC50=("pIC50", "median"), openadmet_n=("pIC50", "size"),
        openadmet_weak=("weak_flag", "any"), openadmet_ci_width=("ci_width", "median"),
    )
    st = chem.join(oa, how="outer")
    st["in_openadmet"] = st["openadmet_n"].notna()
    for col in ("ref_any", "ref_broad", "ref_curated", "ref_quantitative"):
        st[col] = st[col].astype("boolean").fillna(False).astype(bool)
    st["membership"] = np.select(
        [st["in_openadmet"] & st["ref_curated"], st["in_openadmet"], st["ref_curated"]], ["both", "OpenADMET", "ChEMBL"],
        default="",
    )
    st.index.name = "smiles_std"
    st = st.reset_index()
    st["label"] = np.where(st["in_openadmet"], st["openadmet_names"].str.split(", ").str[0],
                           st["chembl_ids"].str.split(", ").str[0])
    return st


@app.cell
def _(chembl_std, oa_std):
    struct_all = build_structure_table(chembl_std, oa_std)
    structures = struct_all[struct_all["membership"] != ""].reset_index(drop=True)
    return struct_all, structures


@app.cell(hide_code=True)
def _(chembl_std, oa_std, openadmet_raw, structures):
    _q = chembl_std["subset"] == "quantitative"
    _c = chembl_std["subset"] == "censored"
    _rows = [
        ("rows in the downloaded file", len(chembl_std), len(openadmet_raw)),
        ("rows with a CYP3A4 IC50 / pIC50", len(chembl_std), len(oa_std)),
        ("quantitative records after curation", int(_q.sum()), len(oa_std)),
        ("censored records (kept separately)", int(_c.sum()), 0),
        ("… of which with a valid standardised structure", int((_q | _c).sum() - chembl_std.loc[_q | _c, "smiles_std"].isna().sum()),
         int(oa_std["smiles_std"].notna().sum())),
        ("OpenADMET values with pIC50 < 4 (flagged, kept)", int(chembl_std.loc[_q, "weak_flag"].sum()), int(oa_std["weak_flag"].sum())),
        ("unique standardised structures", int(structures["membership"].isin(["ChEMBL", "both"]).sum()),
         int(structures["membership"].isin(["OpenADMET", "both"]).sum())),
        ("… present in both sources (exact identity)", int((structures["membership"] == "both").sum()),
         int((structures["membership"] == "both").sum())),
    ]
    before_after = pd.DataFrame(_rows, columns=["", "ChEMBL", "OpenADMET"])
    mo.vstack(
        [
            mo.md("### Before / after\nThe structural analyses below use **one row per unique structure**; the "
                  "measurements stay in their own tables, per source and per assay."),
            mo.ui.table(before_after, selection=None, pagination=False),
        ]
    )
    return (before_after,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    <a id="eda"></a>
    ## 4 · Descriptor-based exploratory analysis (45–60′)

    Nine classic descriptors (RDKit) for every unique structure. A structure in **both** sources counts in both
    distributions. Remember: differences between sources can come from **which chemistry each source selected**
    (medicinal-chemistry series vs a diversity library + approved drugs) *and* from **how activity was measured**.
    """)
    return


@app.function
def descriptor_row(mol):
    return {
        "MW": Descriptors.MolWt(mol), "cLogP": Descriptors.MolLogP(mol), "TPSA": rdMolDescriptors.CalcTPSA(mol),
        "HBD": rdMolDescriptors.CalcNumHBD(mol), "HBA": rdMolDescriptors.CalcNumHBA(mol),
        "RotB": rdMolDescriptors.CalcNumRotatableBonds(mol), "AromRings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "FracSP3": rdMolDescriptors.CalcFractionCSP3(mol), "HeavyAtoms": mol.GetNumHeavyAtoms(),
    }


@app.function
def murcko_scaffold(mol):
    """Bemis–Murcko scaffold as canonical SMILES without stereochemistry; '' for an acyclic molecule."""
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold, isomericSmiles=False) if scaffold.GetNumAtoms() else ""


@app.cell
def _(structures):
    _mols = [Chem.MolFromSmiles(s) for s in mo.status.progress_bar(structures["smiles_std"], title="Descriptors & scaffolds", remove_on_exit=True)]
    features = pd.concat([structures, pd.DataFrame([descriptor_row(m) for m in _mols])], axis=1)
    features["scaffold"] = [murcko_scaffold(m) for m in _mols]
    by_source = pd.concat(
        [
            features[features["membership"].isin(["ChEMBL", "both"])].assign(source="ChEMBL"),
            features[features["membership"].isin(["OpenADMET", "both"])].assign(source="OpenADMET"),
        ],
        ignore_index=True,
    )
    return by_source, features


@app.function
def normalized_histogram(df, column, group="source", bins=30):
    """Histogram per group on shared bins, as fractions of each group (so groups of different size compare)."""
    values = df[column].dropna()
    if np.allclose(values, values.round()) and values.max() - values.min() <= 40:
        edges = np.arange(values.min() - 0.5, values.max() + 1.5)
    else:
        edges = np.histogram_bin_edges(values, bins=bins)
    rows = []
    for name, sub in df.groupby(group):
        counts, _ = np.histogram(sub[column].dropna(), bins=edges)
        total = max(int(counts.sum()), 1)
        rows += [{group: name, "start": a, "end": b, "fraction": c / total, "count": int(c)}
                 for a, b, c in zip(edges[:-1], edges[1:], counts)]
    return pd.DataFrame(rows)


@app.cell
def _():
    descriptor_picker = mo.ui.dropdown(DESCRIPTORS, value="MW", label="Descriptor")
    descriptor_bins = mo.ui.slider(10, 80, step=5, value=35, label="bins (continuous descriptors)", show_value=True)
    mo.hstack([descriptor_picker, descriptor_bins], justify="start", wrap=True, gap=2)
    return descriptor_bins, descriptor_picker


@app.cell(hide_code=True)
def _(by_source, descriptor_bins, descriptor_picker):
    _col = descriptor_picker.value
    _h = normalized_histogram(by_source, _col, bins=descriptor_bins.value)
    fig_descriptor = (
        alt.Chart(_h)
        .mark_bar(opacity=0.55, binSpacing=0)
        .encode(
            x=alt.X("start:Q", bin="binned", title=DESCRIPTOR_NAMES[_col]),
            x2="end:Q",
            y=alt.Y("fraction:Q", stack=None, axis=alt.Axis(format="%"), title="fraction of the source"),
            color=alt.Color("source:N", scale=alt.Scale(domain=SOURCES[:2], range=SOURCE_COLORS[:2]),
                            legend=alt.Legend(orient="top", title=None)),
            tooltip=["source", alt.Tooltip("start:Q", format=".2f"), alt.Tooltip("end:Q", format=".2f"), "count",
                     alt.Tooltip("fraction:Q", format=".1%")],
        )
        .properties(width=520, height=240)
    )
    _stats = by_source.groupby("source")[DESCRIPTORS].describe(percentiles=[0.25, 0.5, 0.75])
    _summary = pd.DataFrame(
        {
            s: {
                "n structures": int(_stats.loc[s, (_col, "count")]),
                "median": round(_stats.loc[s, (_col, "50%")], 2),
                "IQR": f"{_stats.loc[s, (_col, '25%')]:.2f} – {_stats.loc[s, (_col, '75%')]:.2f}",
                "mean": round(_stats.loc[s, (_col, "mean")], 2),
            }
            for s in ("ChEMBL", "OpenADMET")
        }
    )
    _medians = by_source.groupby("source")[DESCRIPTORS].median().T
    _medians["OpenADMET − ChEMBL"] = _medians["OpenADMET"] - _medians["ChEMBL"]
    _medians["in ChEMBL IQRs"] = (_medians["OpenADMET − ChEMBL"] / (by_source[by_source.source == "ChEMBL"][DESCRIPTORS].quantile(0.75) - by_source[by_source.source == "ChEMBL"][DESCRIPTORS].quantile(0.25))).round(2)
    mo.vstack(
        [
            fig_descriptor,
            mo.ui.table(_summary.reset_index(names=_col), selection=None, pagination=False),
            mo.accordion(
                {
                    "Medians of all nine descriptors, and the shift in units of ChEMBL's interquartile range": mo.ui.table(
                        _medians.round(2).reset_index(names="descriptor"), selection=None, pagination=False
                    )
                },
                expanded=True,
            ),
        ]
    )
    return (fig_descriptor,)


@app.cell
def _():
    corr_scope = mo.ui.radio(["ChEMBL", "OpenADMET", "difference (OpenADMET − ChEMBL)"], value="ChEMBL", inline=True,
                             label="Spearman correlation of descriptors in:")
    corr_scope
    return (corr_scope,)


@app.cell(hide_code=True)
def _(by_source, corr_scope):
    _cc = by_source[by_source.source == "ChEMBL"][DESCRIPTORS].corr(method="spearman")
    _co = by_source[by_source.source == "OpenADMET"][DESCRIPTORS].corr(method="spearman")
    _m = {"ChEMBL": _cc, "OpenADMET": _co}.get(corr_scope.value, _co - _cc)
    _long = _m.reset_index(names="a").melt(id_vars="a", var_name="b", value_name="rho")
    _base = alt.Chart(_long).encode(x=alt.X("a:N", sort=DESCRIPTORS, title=None), y=alt.Y("b:N", sort=DESCRIPTORS, title=None))
    _heat = _base.mark_rect().encode(
        color=alt.Color("rho:Q", scale=alt.Scale(scheme="redblue", domain=[-1, 1], reverse=True), title="ρ"),
        tooltip=["a", "b", alt.Tooltip("rho:Q", format=".2f")],
    )
    _text = _base.mark_text(fontSize=10).encode(
        text=alt.Text("rho:Q", format=".2f"),
        color=alt.condition("abs(datum.rho) > 0.6", alt.value("white"), alt.value("black")),
    )
    (_heat + _text).properties(width=380, height=380, title=f"Spearman ρ — {corr_scope.value}")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Activity distributions — per source and per assay, never pooled

    ChEMBL values come from many protocols; OpenADMET from one. Grouping ChEMBL records by **probe** or **enzyme
    source** shows how much the "same" endpoint moves with the assay. There is **no active/inactive cut-off** here:
    we look at whole distributions. Values below 4 (shaded) are imprecise in both sources.
    """)
    return


@app.cell
def _():
    pic50_group_by = mo.ui.radio({"probe family": "probe_family", "enzyme source": "enzyme_source"}, value="probe family",
                                 inline=True, label="Group ChEMBL records by")
    pic50_group_by
    return (pic50_group_by,)


@app.cell(hide_code=True)
def _(chembl_std, oa_std, pic50_group_by):
    _OA = "OpenADMET · recombinant, DBOMF"
    _CENS = "ChEMBL · censored 'IC50 >' bounds (true pIC50 is lower)"
    _q = chembl_std[chembl_std["subset"] == "quantitative"]
    _c = chembl_std[(chembl_std["subset"] == "censored") & chembl_std["standard_relation"].isin([">", ">="])]
    _d = pd.concat(
        [
            pd.DataFrame({"group": _OA, "pIC50": oa_std["pIC50"], "kind": "OpenADMET"}),
            pd.DataFrame({"group": "ChEMBL · " + _q[pic50_group_by.value].astype(str), "pIC50": _q["pIC50"], "kind": "ChEMBL"}),
            pd.DataFrame({"group": _CENS, "pIC50": _c["pIC50_bound"], "kind": "bound"}),
        ],
        ignore_index=True,
    )
    _stats = _d.groupby("group")["pIC50"].agg(n="size", median="median", below_4=lambda s: (s < WEAK_PIC50).mean())
    _order = [_OA] + _stats.drop([_OA, _CENS]).sort_values("n", ascending=False).index.tolist() + [_CENS]
    _shade = alt.Chart(pd.DataFrame({"a": [1.5], "b": [WEAK_PIC50]})).mark_rect(color="#f2c94c", opacity=0.25).encode(
        x=alt.X("a:Q", scale=alt.Scale(domain=[1.5, 10.5], nice=False)), x2="b:Q")
    _box = (
        alt.Chart(_d)
        .mark_boxplot(extent="min-max", size=13)
        .encode(
            y=alt.Y("group:N", sort=_order, title=None, axis=alt.Axis(labelLimit=320)),
            x=alt.X("pIC50:Q", title="pIC50", scale=alt.Scale(domain=[1.5, 10.5], nice=False)),
            color=alt.Color("kind:N", scale=alt.Scale(domain=["OpenADMET", "ChEMBL", "bound"],
                                                      range=[SOURCE_COLORS[1], SOURCE_COLORS[0], "#9a9a9a"]), legend=None),
        )
    )
    fig_pic50 = (_shade + _box).properties(width=330, height=alt.Step(24))
    mo.vstack(
        [
            fig_pic50,
            mo.callout(
                mo.md(f"ChEMBL's quantitative medians look **higher** than OpenADMET's. Part of that is reporting: in "
                      f"ChEMBL, weak compounds are mostly recorded as **'IC50 > x'** ({len(_c):,} censored records, grey "
                      "row: the true pIC50 lies *below* each bound), while OpenADMET fitted a number — often below 4 — "
                      "for every compound. Never compare the grey row as if it were values."),
                kind="warn",
            ),
            mo.ui.table(
                _stats.loc[_order].reset_index().assign(median=lambda t: t["median"].round(2),
                                                        below_4=lambda t: (100 * t["below_4"]).round(1)).rename(columns={"below_4": "% below 4"}),
                selection=None, pagination=False,
            ),
        ]
    )
    return (fig_pic50,)


@app.cell(hide_code=True)
def _(chembl_std, oa_std, structures):
    _both = structures.loc[structures["membership"] == "both", "smiles_std"]
    _oa = oa_std[oa_std["smiles_std"].isin(_both)][["smiles_std", "name", "pIC50"]].rename(columns={"pIC50": "OpenADMET pIC50"})
    _ch = chembl_std[(chembl_std["subset"] == "quantitative") & chembl_std["smiles_std"].isin(_both)][
        ["smiles_std", "molecule_chembl_id", "assay_chembl_id", "probe_family", "enzyme_source", "pIC50"]
    ].rename(columns={"pIC50": "ChEMBL pIC50"})
    _pairs = _oa.merge(_ch, on="smiles_std")
    _line = alt.Chart(pd.DataFrame({"x": [2, 10], "y": [2, 10]})).mark_line(color="#999", strokeDash=[4, 4]).encode(x="x:Q", y="y:Q")
    _pts = (
        alt.Chart(_pairs)
        .mark_circle(size=50, opacity=0.75)
        .encode(
            x=alt.X("ChEMBL pIC50:Q", scale=alt.Scale(domain=[2, 10])),
            y=alt.Y("OpenADMET pIC50:Q", scale=alt.Scale(domain=[1.5, 8])),
            color=alt.Color("probe_family:N", title="ChEMBL probe"),
            tooltip=["name", "molecule_chembl_id", "assay_chembl_id", "probe_family", "enzyme_source",
                     alt.Tooltip("ChEMBL pIC50:Q", format=".2f"), alt.Tooltip("OpenADMET pIC50:Q", format=".2f")],
        )
    )
    _rho = _pairs[["ChEMBL pIC50", "OpenADMET pIC50"]].corr(method="spearman").iloc[0, 1] if len(_pairs) > 2 else float("nan")
    mo.vstack(
        [
            mo.md(
                f"#### The same molecules, measured twice\n**{_both.size}** structures are in both sources; "
                f"**{_pairs['smiles_std'].nunique()}** of them have quantitative ChEMBL records ({len(_pairs)} ChEMBL "
                f"measurements). One point per ChEMBL measurement — nothing averaged. Spearman ρ = {_rho:.2f}. "
                "Points far from the diagonal are not errors: they are different assays."
            ),
            (_line + _pts).properties(width=480, height=380),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### ✏️ Exercise 2 — explain a source difference

    Pick a descriptor whose distribution differs clearly between the sources (look at the "shift in ChEMBL IQRs"
    column). Write two or three sentences: is the difference more likely due to **chemical selection** (which
    molecules each source contains) or to **assay differences**? Can a descriptor of the *structure* even be affected
    by the assay? Then check one hypothesis in code, e.g. compare the 50 most potent ChEMBL compounds with the rest.
    """)
    return


@app.cell
def _():
    exercise2_answer = mo.ui.text_area(placeholder="Your explanation…", rows=4, full_width=True)
    exercise2_answer
    return


@app.cell
def _():
    # ✏️ Exercise 2 — your code here (use _underscored names)
    return


@app.cell(hide_code=True)
def _():
    mo.accordion(
        {
            "🔑 Solution — exercise 2": mo.md(r"""
    Structure descriptors are computed from the structure alone, so a difference in MW, cLogP or aromatic rings can
    only come from **which molecules are in each set**. OpenADMET's CYP3A4 set comes from a diversity library and
    approved drugs: smaller, less lipophilic, fewer aromatic rings than the typical ChEMBL compound, which was
    synthesised in a medicinal-chemistry project (often an optimised, lipophilic lead) and then tested on CYP3A4.
    The assay can still matter **indirectly**: ChEMBL papers tend to report CYP3A4 IC50s for compounds that are
    interesting (potent on the main target, or a CYP liability), so the *selection* of what gets published depends on
    the assay outcome.

    ```python
    _q = chembl_std[chembl_std["subset"] == "quantitative"]
    _per_structure = _q.groupby("smiles_std")["pIC50"].median()
    _f = features.set_index("smiles_std").loc[_per_structure.index, DESCRIPTORS]
    _top = _per_structure.nlargest(50).index
    print(_f.loc[_top].median().round(2).to_frame("top-50 potent").join(_f.drop(_top).median().round(2).rename("rest")))
    ```
    The most potent CYP3A4 inhibitors in ChEMBL are larger and more lipophilic than the rest — CYP3A4's large,
    hydrophobic active site favours such molecules, which is one reason lipophilicity tracks CYP3A4 inhibition.
    """)
        }
    )
    return


@app.cell(hide_code=True)
def _(by_source):
    mo.accordion(
        {
            "🔍 Explore freely (drag-and-drop chart builder)": mo.ui.data_explorer(
                by_source[["source", "label"] + DESCRIPTORS + ["openadmet_pIC50", "chembl_median_pIC50"]]
            )
        }
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    <a id="similarity"></a>
    ## 5 · Morgan fingerprints and Tanimoto similarity (60–80′)

    A **Morgan fingerprint** (radius 2, 2,048 bits — ECFP4-like) sets one bit for every circular atom environment
    up to two bonds wide. The **Tanimoto** similarity of two binary fingerprints $A$ and $B$ is

    $$T(A,B) = \frac{|A \cap B|}{|A \cup B|} \in [0, 1].$$

    Pick a pair, or type your own SMILES. Atoms **highlighted** lie outside the maximum common substructure.
    """)
    return


@app.cell
def _():
    pair_presets = {
        "midazolam vs alprazolam (close analogues)": ("midazolam", "alprazolam"),
        "alprazolam vs triazolam (one chlorine)": ("alprazolam", "triazolam"),
        "ritonavir vs cobicistat (designed booster)": ("ritonavir", "cobicistat"),
        "ketoconazole vs itraconazole (azoles)": ("ketoconazole", "itraconazole"),
        "testosterone vs progesterone (steroids)": ("testosterone", "progesterone"),
        "testosterone vs epitestosterone (T = 1, not identical!)": ("testosterone", "epitestosterone"),
        "ketoconazole vs midazolam (unrelated)": ("ketoconazole", "midazolam"),
    }
    pair_picker = mo.ui.dropdown(list(pair_presets), value="midazolam vs alprazolam (close analogues)", label="Pair",
                                 full_width=True)
    pair_picker
    return pair_picker, pair_presets


@app.cell
def _(pair_picker, pair_presets):
    _a, _b = pair_presets[pair_picker.value]
    pair_a = mo.ui.text(value=REFERENCE_DRUGS[_a][0], label=f"A ({_a})", full_width=True)
    pair_b = mo.ui.text(value=REFERENCE_DRUGS[_b][0], label=f"B ({_b})", full_width=True)
    mo.vstack([pair_a, pair_b])
    return pair_a, pair_b


@app.function
def pair_with_differences(smiles_a, smiles_b, legends, size=(300, 230)):
    """Draw two molecules side by side, highlighting atoms outside their maximum common substructure."""
    mols = [Chem.MolFromSmiles(smiles_a), Chem.MolFromSmiles(smiles_b)]
    if any(m is None for m in mols):
        return mol_grid([smiles_a, smiles_b], legends, per_row=2, size=size)
    mcs = rdFMCS.FindMCS(mols, timeout=2, ringMatchesRingOnly=True, completeRingsOnly=True)
    core = Chem.MolFromSmarts(mcs.smartsString) if mcs.numAtoms else None
    marks = []
    for mol in mols:
        matched = set(mol.GetSubstructMatch(core)) if core is not None else set()
        marks.append([a.GetIdx() for a in mol.GetAtoms() if a.GetIdx() not in matched])
    return mol_grid([smiles_a, smiles_b], legends, per_row=2, size=size, highlights=marks)


@app.cell(hide_code=True)
def _(pair_a, pair_b):
    _ma, _mb = Chem.MolFromSmiles(pair_a.value), Chem.MolFromSmiles(pair_b.value)
    if _ma is None or _mb is None:
        _out = mo.callout("One of the SMILES cannot be parsed.", kind="danger")
    else:
        _fa, _fb = FP_GEN.GetFingerprint(_ma), FP_GEN.GetFingerprint(_mb)
        _t = DataStructs.TanimotoSimilarity(_fa, _fb)
        _shared = len(set(_fa.GetOnBits()) & set(_fb.GetOnBits()))
        _identical = Chem.MolToSmiles(_ma) == Chem.MolToSmiles(_mb)
        _out = mo.vstack(
            [
                mo.hstack(
                    [
                        mo.stat(f"{_t:.3f}", label="Tanimoto", bordered=True),
                        mo.stat(f"{_fa.GetNumOnBits()} / {_fb.GetNumOnBits()}", label="bits on (A / B)", bordered=True),
                        mo.stat(str(_shared), label="shared bits", bordered=True),
                        mo.stat("yes" if _identical else "no", label="identical structures?", bordered=True),
                    ],
                    justify="start", wrap=True,
                ),
                pair_with_differences(pair_a.value, pair_b.value, ["A", "B"]),
                mo.callout("Tanimoto = 1 but the structures differ: this fingerprint ignores stereochemistry. That is why "
                           "exact overlap is decided by standardised isomeric SMILES, never by T = 1.", kind="warn")
                if _t == 1 and not _identical else mo.md(""),
            ]
        )
    _out
    return


@app.cell(hide_code=True)
def _():
    _names = ["ketoconazole", "itraconazole", "miconazole", "ritonavir", "cobicistat", "verapamil", "midazolam",
              "alprazolam", "triazolam", "testosterone", "progesterone", "nifedipine"]
    _fps = [FP_GEN.GetFingerprint(Chem.MolFromSmiles(REFERENCE_DRUGS[n][0])) for n in _names]
    _rows = [{"a": a, "b": b, "T": DataStructs.TanimotoSimilarity(fa, fb)} for a, fa in zip(_names, _fps) for b, fb in zip(_names, _fps)]
    _base = alt.Chart(pd.DataFrame(_rows)).encode(x=alt.X("a:N", sort=_names, title=None), y=alt.Y("b:N", sort=_names, title=None))
    mo.vstack(
        [
            mo.md("A few CYP3A4 drugs compared all-against-all. Close analogues (one atom or ring changed) reach "
                  "T ≈ 0.6–0.8 with this fingerprint; unrelated drugs sit around 0.05–0.2."),
            (_base.mark_rect().encode(color=alt.Color("T:Q", scale=alt.Scale(scheme="blues", domain=[0, 1])),
                                      tooltip=["a", "b", alt.Tooltip("T:Q", format=".2f")])
             + _base.mark_text(fontSize=9).encode(text=alt.Text("T:Q", format=".2f"),
                                                  color=alt.condition("datum.T > 0.6", alt.value("white"), alt.value("black"))))
            .properties(width=420, height=420),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### The nearest ChEMBL neighbour of every OpenADMET compound

    For each OpenADMET structure we compute its Tanimoto similarity to **every** structure of the ChEMBL reference
    set at once (RDKit's `BulkTanimotoSimilarity`), then keep the best **non-identical** match. Exact structure
    overlap is reported separately — a molecule is not its own neighbour.
    """)
    return


@app.function
def similarity_matrix(query_fps, reference_fps):
    """Tanimoto similarity of every query fingerprint to every reference fingerprint (float32 matrix)."""
    out = np.empty((len(query_fps), len(reference_fps)), dtype=np.float32)
    for i, fp in enumerate(mo.status.progress_bar(query_fps, title="Bulk Tanimoto", remove_on_exit=True)):
        out[i] = DataStructs.BulkTanimotoSimilarity(fp, reference_fps)
    return out


@app.function
def nearest_neighbours(sim, query_keys, reference_keys, reference_mask):
    """Best non-identical reference for every query row of `sim`, among the columns allowed by `reference_mask`.

    Identity = same standardised isomeric SMILES; such a match is returned as `exact` and excluded from the search."""
    s = np.where(reference_mask[None, :], sim, -1.0)
    position = {k: j for j, k in enumerate(reference_keys)}
    exact = np.zeros(len(query_keys), dtype=bool)
    for i, key in enumerate(query_keys):
        j = position.get(key)
        if j is not None and reference_mask[j]:
            exact[i] = True
            s[i, j] = -1.0                                   # no self-match
    best = s.argmax(axis=1)
    return exact, best, s[np.arange(len(best)), best]


@app.cell
def _(struct_all):
    sim_refs = struct_all[struct_all["ref_any"]].reset_index(drop=True)
    sim_queries = struct_all[struct_all["in_openadmet"]].reset_index(drop=True)
    _ref_fps = [FP_GEN.GetFingerprint(Chem.MolFromSmiles(s)) for s in sim_refs["smiles_std"]]
    _query_fps = [FP_GEN.GetFingerprint(Chem.MolFromSmiles(s)) for s in sim_queries["smiles_std"]]
    sim_matrix = similarity_matrix(_query_fps, _ref_fps)
    return sim_matrix, sim_queries, sim_refs


@app.cell
def _():
    reference_choice = mo.ui.radio(
        {
            "quantitative + censored records (workshop default)": "ref_curated",
            "quantitative records only": "ref_quantitative",
            "any confidence-9 human CYP3A4 IC50 record (incl. TDI, other assay contexts)": "ref_broad",
            "any CHEMBL340 IC50 record, even with assay confidence < 9": "ref_any",
        },
        value="quantitative + censored records (workshop default)",
        label="**ChEMBL reference set**",
    )
    coverage_threshold = mo.ui.slider(0.2, 0.9, step=0.05, value=0.4, label="'covered' if nearest neighbour T ≥",
                                      show_value=True)
    mo.vstack([reference_choice, coverage_threshold])
    return coverage_threshold, reference_choice


@app.cell
def _(reference_choice, sim_matrix, sim_queries, sim_refs):
    _mask = sim_refs[reference_choice.value].to_numpy()
    _exact, _best, _t = nearest_neighbours(sim_matrix, sim_queries["smiles_std"].tolist(), sim_refs["smiles_std"].tolist(), _mask)
    nn_table = pd.DataFrame(
        {
            "name": sim_queries["label"],
            "openadmet_pIC50": sim_queries["openadmet_pIC50"].round(2),
            "exact_in_chembl": _exact,
            "nn_tanimoto": _t.round(3),
            "nn_chembl_id": sim_refs["chembl_ids"].to_numpy()[_best],
            "nn_chembl_median_pIC50": sim_refs["chembl_median_pIC50"].to_numpy()[_best].round(2),
            "smiles_std": sim_queries["smiles_std"],
            "nn_smiles": sim_refs["smiles_std"].to_numpy()[_best],
            "query_row": np.arange(len(sim_queries)),
        }
    )
    nn_reference_size = int(_mask.sum())
    return nn_reference_size, nn_table


@app.cell(hide_code=True)
def _(coverage_threshold, nn_reference_size, nn_table):
    _thr = coverage_threshold.value
    _h = nn_table.assign(bin=(nn_table["nn_tanimoto"] // 0.025) * 0.025, covered=nn_table["nn_tanimoto"] >= _thr)
    _h = _h.groupby(["bin", "covered"]).size().rename("molecules").reset_index()
    _hist = (
        alt.Chart(_h)
        .mark_bar(width=9)
        .encode(
            x=alt.X("bin:Q", title="Tanimoto to the nearest non-identical ChEMBL structure",
                    scale=alt.Scale(domain=[0, 1], nice=False)),
            y=alt.Y("molecules:Q", stack=True),
            color=alt.Color("covered:N", scale=alt.Scale(domain=[True, False], range=["#3b6fb6", "#e07b39"]),
                            legend=alt.Legend(title=f"T ≥ {_thr:.2f}")),
            tooltip=["bin", "molecules"],
        )
    )
    _rule = alt.Chart(pd.DataFrame({"t": [_thr]})).mark_rule(color="black", strokeDash=[4, 3]).encode(x="t:Q")
    fig_nn_similarity = (_hist + _rule).properties(width=520, height=250)
    mo.vstack(
        [
            mo.hstack(
                [
                    mo.stat(f"{nn_reference_size:,}", label="ChEMBL reference structures", bordered=True),
                    mo.stat(f"{len(nn_table):,}", label="OpenADMET structures", bordered=True),
                    mo.stat(f"{int(nn_table['exact_in_chembl'].sum())} ({nn_table['exact_in_chembl'].mean():.1%})",
                            label="exact overlap", caption="same standardised structure", bordered=True),
                    mo.stat(f"{nn_table['nn_tanimoto'].median():.2f}", label="median nearest-neighbour T", bordered=True),
                    mo.stat(f"{(nn_table['nn_tanimoto'] >= _thr).mean():.1%}", label=f"with a neighbour T ≥ {_thr:.2f}", bordered=True),
                ],
                justify="start",
                wrap=True,
            ),
            fig_nn_similarity,
        ]
    )
    return (fig_nn_similarity,)


@app.cell
def _(nn_table):
    nn_browser = mo.ui.table(
        nn_table.drop(columns=["query_row"]).sort_values("nn_tanimoto").reset_index(drop=True),
        selection="single",
        page_size=6,
        label="Click a compound (sorted from least to most covered; use the search box for a name)",
    )
    nn_browser
    return (nn_browser,)


@app.cell(hide_code=True)
def _(
    chembl_std,
    nn_browser,
    reference_choice,
    sim_matrix,
    sim_queries,
    sim_refs,
):
    _sel = nn_browser.value
    if not len(_sel):
        _out = mo.md("_Select a row above to see the compound next to its three nearest ChEMBL neighbours._")
    else:
        _r = _sel.iloc[0]
        _i = int(sim_queries.index[sim_queries["smiles_std"] == _r["smiles_std"]][0])
        _row = np.where(sim_refs[reference_choice.value].to_numpy(), sim_matrix[_i], -1.0)
        _row[(sim_refs["smiles_std"] == _r["smiles_std"]).to_numpy()] = -1.0
        _top = np.argsort(-_row)[:3]
        _nbrs = sim_refs.iloc[_top]
        _legends = [f"{_r['name']} (pIC50 {_r['openadmet_pIC50']})"] + [
            f"{ids.split(', ')[0]}  T={_row[j]:.2f}" for ids, j in zip(_nbrs["chembl_ids"], _top)
        ]
        _records = chembl_std[chembl_std["smiles_std"] == _nbrs.iloc[0]["smiles_std"]][
            ["molecule_chembl_id", "assay_chembl_id", "subset", "standard_relation", "standard_value", "pIC50",
             "pIC50_bound", "probe_family", "enzyme_source", "exclusion_reason"]
        ]
        _out = mo.vstack(
            [
                mo.md(f"**{_r['name']}** — exact structure in ChEMBL: **{'yes' if _r['exact_in_chembl'] else 'no'}**. "
                      "Nearest non-identical neighbours (atoms outside the common substructure with the first "
                      "neighbour are highlighted):"),
                pair_with_differences(_r["smiles_std"], _nbrs.iloc[0]["smiles_std"], _legends[:2]),
                mol_grid(_nbrs["smiles_std"].tolist()[1:], _legends[2:], per_row=2, size=(260, 200)),
                mo.md("ChEMBL records of the nearest neighbour (activities stay per assay):"),
                mo.ui.table(_records.reset_index(drop=True), selection=None, page_size=5),
            ]
        )
    _out
    return


@app.cell(hide_code=True)
def _(chembl_std, coverage_threshold, sim_matrix, sim_queries, sim_refs):
    _rows, _exact_sets = [], {}
    for _label, _col in [("quantitative only", "ref_quantitative"), ("quantitative + censored (default)", "ref_curated"),
                         ("any confidence-9 IC50 record", "ref_broad"), ("any CHEMBL340 IC50 record", "ref_any")]:
        _mask = sim_refs[_col].to_numpy()
        _exact, _best, _t = nearest_neighbours(sim_matrix, sim_queries["smiles_std"].tolist(), sim_refs["smiles_std"].tolist(), _mask)
        _exact_sets[_col] = set(sim_queries.loc[_exact, "smiles_std"])
        _rows.append({"ChEMBL reference": _label, "structures": int(_mask.sum()), "exact overlap": int(_exact.sum()),
                      "median NN T": round(float(np.median(_t)), 3),
                      f"% with NN T ≥ {coverage_threshold.value:.2f}": round(100 * float((_t >= coverage_threshold.value).mean()), 1),
                      "% with NN T < 0.30": round(100 * float((_t < 0.30).mean()), 1)})
    _extra = _exact_sets["ref_any"] - _exact_sets["ref_curated"]
    _why = (
        chembl_std[chembl_std["smiles_std"].isin(_extra)]
        .assign(reason=lambda d: d["exclusion_reason"].fillna("kept"), assay=lambda d: d["assay_description"].str.slice(0, 70))
        .groupby(["reason", "assay"]).agg(records=("smiles_std", "size"), structures=("smiles_std", "nunique"))
        .sort_values("structures", ascending=False).reset_index()
    )
    mo.vstack(
        [
            mo.md("#### How much does the answer depend on the definition of the reference set?\n"
                  "The curation policy "
                  "is part of the answer. Most OpenADMET molecules that ChEMBL *does* contain — many are approved drugs "
                  "— were tested in assays our policy excludes:"),
            mo.ui.table(pd.DataFrame(_rows), selection=None, pagination=False),
            mo.md(f"Why the **{len(_extra)}** extra exact matches of the loosest definition are not in the default "
                  "reference (first failed rule of their ChEMBL records, and the assay):"),
            mo.ui.table(_why, selection=None, page_size=6),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### ✏️ Exercise 3 — inspect a nearest-neighbour pair

    In the table above, pick one compound with T ≈ 0.4–0.5 and one with T > 0.7.
    1. What exactly differs between each compound and its neighbour (look at the highlighted atoms)?
    2. Would you, as a chemist, call the first pair "analogues"? What does that say about a fixed cut-off?
    3. Does the neighbour have a **quantitative** ChEMBL value, a **censored** one, or only excluded records?
       Could you use its activity to guess the OpenADMET value? (Think assay differences.)
    """)
    return


@app.cell
def _():
    # ✏️ Exercise 3 — your code here, e.g. nn_table[nn_table["nn_tanimoto"].between(0.4, 0.5)].head()
    return


@app.cell(hide_code=True)
def _():
    mo.accordion(
        {
            "🔑 Solution — exercise 3": mo.md(r"""
    ```python
    _mid = nn_table[nn_table["nn_tanimoto"].between(0.4, 0.5)].iloc[0]
    _high = nn_table[(nn_table["nn_tanimoto"] > 0.7) & ~nn_table["exact_in_chembl"]].iloc[0]
    for _r in (_mid, _high):
        print(_r["name"], _r["nn_chembl_id"], _r["nn_tanimoto"])
    pair_with_differences(_mid["smiles_std"], _mid["nn_smiles"], ["OpenADMET", "ChEMBL"])
    ```
    At T ≈ 0.45 the pair usually shares a ring system or a linker but differs in substituents or in one ring —
    a medicinal chemist may or may not call it an analogue. Above 0.7 the change is typically a single substituent.
    Any threshold is a convention, so we report whole distributions and test several cut-offs (slider).
    A neighbour's ChEMBL activity comes from another assay (probe, enzyme source) and, when censored, is only a bound:
    it is a hint, not a substitute for the OpenADMET value.
    """)
        }
    )
    return


@app.cell
def _():
    test_set_button = mo.ui.run_button(label="Optional: check coverage of the 750 blinded test structures")
    test_set_button
    return (test_set_button,)


@app.cell(hide_code=True)
def _(nn_table, sim_refs, test_set_button):
    mo.stop(not test_set_button.value)
    try:
        _r = requests.get(OPENADMET_TEST_URL, timeout=60)
        _r.raise_for_status()
        _bytes, _origin = _r.content, "pinned Hugging Face file"
    except Exception as _exc:
        _bytes, _where = read_cached("openadmet_test")
        _origin = f"cached snapshot ({_where}); live download failed: {type(_exc).__name__}"
    _test = pd.read_csv(io.BytesIO(_bytes))
    _std = standardize_many(_test["SMILES"], title="Standardising test structures")
    _keys = _std["smiles_std"].dropna().unique().tolist()
    _mask = sim_refs["ref_curated"].to_numpy()
    _sim = similarity_matrix([FP_GEN.GetFingerprint(Chem.MolFromSmiles(s)) for s in _keys],
                             [FP_GEN.GetFingerprint(Chem.MolFromSmiles(s)) for s in sim_refs["smiles_std"]])
    _exact, _best, _t = nearest_neighbours(_sim, _keys, sim_refs["smiles_std"].tolist(), _mask)
    _cmp = pd.concat([pd.DataFrame({"set": "training (CYP3A4)", "T": nn_table["nn_tanimoto"]}),
                      pd.DataFrame({"set": "blinded test (structures only)", "T": _t})])
    mo.vstack(
        [
            mo.md(f"Test set from {_origin}: {len(_test)} rows, {len(_keys)} unique structures, {int(_exact.sum())} exactly "
                  f"in ChEMBL, median nearest-neighbour T = {np.median(_t):.2f}. No labels are used."),
            alt.Chart(_cmp).transform_density("T", groupby=["set"], as_=["T", "density"], extent=[0, 1])
            .mark_line().encode(x="T:Q", y="density:Q", color="set:N").properties(width=560, height=220),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    <a id="scaffolds"></a>
    ## 6 · Bemis–Murcko scaffolds (80–95′)

    The **Bemis–Murcko scaffold** keeps ring systems and the linkers between them and drops all side chains
    (stereochemistry removed). **Acyclic** molecules have no scaffold — we count them separately instead of lumping
    them into one fake "empty" scaffold.
    """)
    return


@app.cell
def _(features):
    _cyclic = features[features["scaffold"] != ""]
    scaffold_table = (
        pd.DataFrame(
            {
                "ChEMBL": _cyclic[_cyclic["membership"].isin(["ChEMBL", "both"])]["scaffold"].value_counts(),
                "OpenADMET": _cyclic[_cyclic["membership"].isin(["OpenADMET", "both"])]["scaffold"].value_counts(),
            }
        )
        .fillna(0)
        .astype(int)
        .rename_axis("scaffold")
        .reset_index()
    )
    scaffold_table["status"] = np.select(
        [(scaffold_table["ChEMBL"] > 0) & (scaffold_table["OpenADMET"] > 0), scaffold_table["OpenADMET"] > 0],
        ["shared", "OpenADMET only"], default="ChEMBL only",
    )
    scaffold_table["total"] = scaffold_table["ChEMBL"] + scaffold_table["OpenADMET"]
    scaffold_table = scaffold_table.sort_values("total", ascending=False).reset_index(drop=True)
    return (scaffold_table,)


@app.cell(hide_code=True)
def _(features, scaffold_table):
    def _src(name):
        _f = features[features["membership"].isin([name, "both"])]
        _counts = scaffold_table.loc[scaffold_table[name] > 0, name]
        return _f, _counts

    _cards = []
    for _name in ("ChEMBL", "OpenADMET"):
        _f, _counts = _src(_name)
        _cards.append(
            mo.vstack(
                [
                    mo.md(f"**{_name}**"),
                    mo.hstack(
                        wrap=True,
                        items=[
                            mo.stat(f"{len(_counts):,}", label="scaffolds", caption=f"{len(_counts) / max((_f['scaffold'] != '').sum(), 1):.2f} per cyclic compound", bordered=True),
                            mo.stat(f"{(_counts == 1).mean():.0%}", label="singletons", caption="scaffolds seen once", bordered=True),
                            mo.stat(f"{int((_f['scaffold'] == '').sum())}", label="acyclic compounds", bordered=True),
                        ],
                        justify="start",
                    ),
                ]
            )
        )
    _oa = features[features["membership"].isin(["OpenADMET", "both"]) & (features["scaffold"] != "")]
    _chembl_scaffolds = set(scaffold_table.loc[scaffold_table["ChEMBL"] > 0, "scaffold"])
    _novel = ~_oa["scaffold"].isin(_chembl_scaffolds)
    mo.vstack(
        [
            *_cards,
            mo.hstack(
                [
                    mo.stat(f"{int((scaffold_table['status'] == 'shared').sum()):,}", label="shared scaffolds", bordered=True),
                    mo.stat(f"{_novel.mean():.1%}", label="OpenADMET cyclic compounds with a scaffold absent from ChEMBL",
                            caption=f"{int(_novel.sum()):,} of {len(_oa):,}", bordered=True),
                    mo.stat(f"{int((scaffold_table['status'] == 'OpenADMET only').sum()):,}", label="scaffolds found only in OpenADMET", bordered=True),
                ],
                justify="start",
                wrap=True,
            ),
        ]
    )
    return


@app.cell
def _():
    top_n_scaffolds = mo.ui.slider(4, 24, step=4, value=8, label="most common scaffolds to draw", show_value=True)
    top_n_scaffolds
    return (top_n_scaffolds,)


@app.cell(hide_code=True)
def _(scaffold_table, top_n_scaffolds):
    def _grid(col, n):
        _t = scaffold_table[scaffold_table[col] > 0].sort_values(col, ascending=False).head(n)
        return mol_grid(_t["scaffold"].tolist(), [f"{c} cpds · {s}" for c, s in zip(_t[col], _t["status"])], per_row=4,
                        size=(190, 150))

    _shared = scaffold_table[scaffold_table["status"] == "shared"].assign(m=lambda t: t[["ChEMBL", "OpenADMET"]].min(axis=1))
    _shared = _shared.sort_values("m", ascending=False).head(top_n_scaffolds.value)
    mo.ui.tabs(
        {
            "ChEMBL": _grid("ChEMBL", top_n_scaffolds.value),
            "OpenADMET": _grid("OpenADMET", top_n_scaffolds.value),
            "Shared": mol_grid(_shared["scaffold"].tolist(), [f"ChEMBL {a} · OA {b}" for a, b in zip(_shared["ChEMBL"], _shared["OpenADMET"])],
                               per_row=4, size=(190, 150)),
        }
    )
    return


@app.cell
def _():
    scaffold_status_filter = mo.ui.dropdown(["all", "shared", "OpenADMET only", "ChEMBL only"], value="OpenADMET only",
                                            label="Scaffold table: show")
    scaffold_status_filter
    return (scaffold_status_filter,)


@app.cell
def _(scaffold_status_filter, scaffold_table):
    _t = scaffold_table if scaffold_status_filter.value == "all" else scaffold_table[scaffold_table["status"] == scaffold_status_filter.value]
    scaffold_browser = mo.ui.table(_t.reset_index(drop=True), selection="multi", page_size=6,
                                   label="Select one or more scaffolds: their members are drawn below, and they can colour the maps of part 7")
    scaffold_browser
    return (scaffold_browser,)


@app.cell(hide_code=True)
def _(features, scaffold_browser):
    _sel = scaffold_browser.value
    if not len(_sel):
        _out = mo.md("_Select scaffolds in the table to see the compounds that contain them._")
    else:
        _m = features[features["scaffold"].isin(_sel["scaffold"])].sort_values("membership").head(16)
        _out = mol_grid(_m["smiles_std"].tolist(), [f"{l} · {m}" for l, m in zip(_m["label"], _m["membership"])], per_row=4,
                        size=(200, 160))
    _out
    return


@app.function
def scaffold_rarefaction(scaffolds, sizes, repeats=20, seed=SEED):
    """Unique scaffolds in random subsamples of each size: diversity compared at *matched* sample sizes."""
    rng = np.random.default_rng(seed)
    values = np.asarray(scaffolds)
    rows = []
    for n in sizes:
        if n > len(values):
            continue
        for r in range(repeats):
            counts = pd.Series(values[rng.choice(len(values), size=n, replace=False)]).value_counts()
            rows.append({"n": n, "repeat": r, "unique scaffolds": len(counts),
                         "scaffolds per compound": len(counts) / n, "singleton fraction": float((counts == 1).mean())})
    return pd.DataFrame(rows)


@app.cell(hide_code=True)
def _(features):
    _frames = []
    for _name in ("ChEMBL", "OpenADMET"):
        _s = features.loc[features["membership"].isin([_name, "both"]) & (features["scaffold"] != ""), "scaffold"]
        _sizes = sorted({100, 250, 500, 750, 1000, 1500, 2000, 3000, 5000, 7500, len(_s)})
        _frames.append(scaffold_rarefaction(_s, _sizes).assign(source=_name))
    _rare = pd.concat(_frames)
    _band = alt.Chart(_rare).mark_errorband(extent="stdev", opacity=0.25).encode(
        x=alt.X("n:Q", title="compounds sampled (cyclic only)", scale=alt.Scale(type="log")),
        y=alt.Y("scaffolds per compound:Q", scale=alt.Scale(zero=False)),
        color=alt.Color("source:N", scale=alt.Scale(domain=SOURCES[:2], range=SOURCE_COLORS[:2])),
    )
    _line = alt.Chart(_rare).mark_line(point=True).encode(x="n:Q", y="mean(scaffolds per compound):Q", color="source:N")
    fig_rarefaction = (_band + _line).properties(width=520, height=260)
    _at = _rare[_rare["n"] == 1000].groupby("source")[["scaffolds per compound", "singleton fraction"]].agg(["mean", "std"]).round(3)
    mo.vstack(
        [
            mo.md("#### Diversity at matched sample sizes\nScaffolds-per-compound always *falls* as a set grows, so "
                  "comparing a 2,300-compound set with a 7,000-compound set is unfair. Here both are subsampled to the "
                  "same sizes (20 random draws each, seed 42):"),
            fig_rarefaction,
            mo.md("At n = 1,000:"),
            mo.ui.table(_at.reset_index(), selection=None, pagination=False),
        ]
    )
    return (fig_rarefaction,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### ✏️ Exercise 4 — find a scaffold absent from the ChEMBL reference

    Use the scaffold table (filter *OpenADMET only*) to pick the **most populated** OpenADMET scaffold that never
    occurs in ChEMBL. Draw it and its members. Then check with the nearest-neighbour table: are its members also
    far from ChEMBL in fingerprint space, or do they have close ChEMBL neighbours built on a *different* scaffold?
    """)
    return


@app.cell
def _():
    # ✏️ Exercise 4 — your code here
    return


@app.cell(hide_code=True)
def _():
    mo.accordion(
        {
            "🔑 Solution — exercise 4": mo.md(r"""
    ```python
    _novel = scaffold_table[scaffold_table["status"] == "OpenADMET only"].sort_values("OpenADMET", ascending=False)
    _scaf = _novel.iloc[0]["scaffold"]
    _members = features[features["scaffold"] == _scaf]
    print(_scaf, len(_members), "compounds")
    print(nn_table[nn_table["smiles_std"].isin(_members["smiles_std"])][["name", "nn_tanimoto", "nn_chembl_id"]])
    mol_grid([_scaf] + _members["smiles_std"].tolist()[:7], ["scaffold"] + _members["label"].tolist()[:7])
    ```
    A scaffold can be "new" while its compounds still have fairly similar ChEMBL neighbours (same side chains or a
    ring swapped for a close isostere): scaffold novelty and fingerprint novelty are related but not identical.
    Small scaffolds (one benzene ring) are almost never novel; novel ones are usually fused or linked ring systems.
    """)
        }
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    <a id="maps"></a>
    ## 7 · Chemical space in 2D: PCA and UMAP (95–110′)

    Maps are drawn on a **reproducible random sample**: at most 1,000 unique structures per source (seed 42),
    then the union is de-duplicated; a structure present in both sources keeps its **both** label. The sample is
    *not* enriched in active or unusual compounds.

    - **PCA** of the nine standardised descriptors: a linear map whose axes we can interpret through the loadings.
    - **UMAP** of the binary Morgan fingerprints with the Jaccard (= 1 − Tanimoto) distance: a non-linear map that
      preserves neighbourhoods. We fit **one joint map**; maps fitted separately would have unrelated axes.
    """)
    return


@app.cell
def _(features):
    _rng_parts = []
    for _name in ("ChEMBL", "OpenADMET"):
        _pool = features[features["membership"].isin([_name, "both"])]
        _rng_parts.append(_pool.sample(n=min(MAX_PER_SOURCE, len(_pool)), random_state=SEED))
    map_sample = pd.concat(_rng_parts).drop_duplicates("smiles_std").reset_index(drop=True)
    mo.md(
        f"Map sample: **{len(map_sample):,}** unique structures — "
        + ", ".join(f"{k}: {v}" for k, v in map_sample["membership"].value_counts().items())
    )
    return (map_sample,)


@app.cell
def _(features, map_sample):
    pca_scaler = StandardScaler().fit(map_sample[DESCRIPTORS])
    pca_model = PCA(n_components=5, random_state=SEED).fit(pca_scaler.transform(map_sample[DESCRIPTORS]))
    _xy = pca_model.transform(pca_scaler.transform(map_sample[DESCRIPTORS]))
    map_sample_pca = map_sample.assign(PC1=_xy[:, 0], PC2=_xy[:, 1])
    _all = pca_model.transform(pca_scaler.transform(features[DESCRIPTORS]))
    features_pca = features[["smiles_std"]].assign(PC1=_all[:, 0], PC2=_all[:, 1])
    return features_pca, map_sample_pca, pca_model


@app.cell(hide_code=True)
def _(pca_model):
    _ev = pd.DataFrame({"component": [f"PC{i + 1}" for i in range(5)], "explained": pca_model.explained_variance_ratio_})
    _ev["cumulative"] = _ev["explained"].cumsum()
    _load = pd.DataFrame(pca_model.components_[:3].T, index=DESCRIPTORS, columns=["PC1", "PC2", "PC3"])
    _load_long = _load.reset_index(names="descriptor").melt(id_vars="descriptor", var_name="component", value_name="loading")
    _bars = alt.Chart(_ev).mark_bar(color="#7d8fa6").encode(
        x=alt.X("component:N", title=None), y=alt.Y("explained:Q", axis=alt.Axis(format="%"), title="explained variance"))
    _cum = alt.Chart(_ev).mark_line(point=True, color="black").encode(x="component:N", y="cumulative:Q")
    _heat = alt.Chart(_load_long).mark_rect().encode(
        x=alt.X("component:N", title=None), y=alt.Y("descriptor:N", sort=DESCRIPTORS, title=None),
        color=alt.Color("loading:Q", scale=alt.Scale(scheme="redblue", domain=[-0.7, 0.7], reverse=True)),
        tooltip=["descriptor", "component", alt.Tooltip("loading:Q", format=".2f")])
    _txt = alt.Chart(_load_long).mark_text(fontSize=10).encode(x="component:N", y=alt.Y("descriptor:N", sort=DESCRIPTORS),
                                                               text=alt.Text("loading:Q", format=".2f"))
    mo.hstack(wrap=True, items=[(_bars + _cum).properties(width=240, height=220, title="Explained variance"),
               (_heat + _txt).properties(width=200, height=220, title="Leading loadings")], justify="start", gap=2)
    return


@app.cell
def _():
    _options = ["source", "scaffold (your selection in part 6, else the 8 most common)", "OpenADMET pIC50",
                "ChEMBL median pIC50 (quantitative records)", "OpenADMET pIC50 < 4 (cautious)"]
    map_color = mo.ui.dropdown(_options, value="source", label="Colour the maps by")
    map_color
    return (map_color,)


@app.function
def map_frame(sample, coords, color_mode, selected_scaffolds, top_scaffolds):
    """Data for a 2D map: coordinates plus one `colour` column (categorical or numeric) chosen by `color_mode`."""
    d = sample[["smiles_std", "label", "membership", "scaffold", "openadmet_pIC50", "chembl_median_pIC50",
                "openadmet_weak", "MW", "cLogP"]].copy()
    d[["x", "y"]] = coords
    if color_mode.startswith("scaffold"):
        chosen = list(selected_scaffolds) or list(top_scaffolds)
        names = {s: f"S{i + 1}" for i, s in enumerate(chosen)}
        d["colour"] = d["scaffold"].map(names).fillna("other")
    elif color_mode == "OpenADMET pIC50":
        d["value"] = d["openadmet_pIC50"]
    elif color_mode.startswith("ChEMBL median"):
        d["value"] = d["chembl_median_pIC50"]
    elif color_mode.startswith("OpenADMET pIC50 < 4"):
        d["colour"] = np.where(d["openadmet_weak"].astype("boolean").fillna(False), "pIC50 < 4", np.where(d["membership"] == "ChEMBL", "ChEMBL only", "pIC50 ≥ 4"))
    else:
        d["colour"] = d["membership"]
    if "value" in d:
        d["has_value"] = d["value"].notna()
        d["value"] = d["value"].fillna(d["value"].median())
    return d.astype({"openadmet_weak": "object"})


@app.function
def map_chart(d, x_title, y_title, title):
    base = alt.Chart(d).mark_circle(size=22, opacity=0.75)
    tooltip = ["label", "membership", alt.Tooltip("openadmet_pIC50:Q", format=".2f"),
               alt.Tooltip("chembl_median_pIC50:Q", format=".2f"), alt.Tooltip("MW:Q", format=".0f"),
               alt.Tooltip("cLogP:Q", format=".1f")]
    if "has_value" in d:
        color = alt.condition("datum.has_value", alt.Color("value:Q", scale=alt.Scale(scheme="viridis"), title="pIC50",
                                                       legend=alt.Legend(orient="top")),
                              alt.value("#d9d9d9"))
        enc = {"color": color, "order": alt.Order("has_value:Q")}
    elif set(d["colour"]) <= set(SOURCES):
        enc = {"color": alt.Color("colour:N", scale=alt.Scale(domain=SOURCES, range=SOURCE_COLORS), title=None,
                                  legend=alt.Legend(orient="top"))}
    else:
        enc = {"color": alt.Color("colour:N", scale=alt.Scale(scheme="tableau10"), title=None,
                                  legend=alt.Legend(orient="top", columns=5))}
    return base.encode(x=alt.X("x:Q", title=x_title), y=alt.Y("y:Q", title=y_title), tooltip=tooltip, **enc).properties(
        width=520, height=420, title=title)


@app.cell
def _(features, map_color, map_sample_pca, scaffold_browser):
    _top = features.loc[features["scaffold"] != "", "scaffold"].value_counts().head(8).index
    _sel = scaffold_browser.value["scaffold"].tolist() if len(scaffold_browser.value) else []
    _d = map_frame(map_sample_pca, map_sample_pca[["PC1", "PC2"]].to_numpy(), map_color.value, _sel, _top)
    fig_pca = map_chart(_d, "PC1", "PC2", "PCA of nine standardised descriptors")
    pca_chart = mo.ui.altair_chart(fig_pca, chart_selection="interval", legend_selection=False)
    pca_chart
    return fig_pca, pca_chart


@app.cell(hide_code=True)
def _(pca_chart):
    _sel = pca_chart.value
    mo.vstack([mo.md(f"**{len(_sel)} selected** — first 12 drawn:"),
               mol_grid(_sel["smiles_std"].tolist()[:12], [f"{l} · {m}" for l, m in zip(_sel["label"], _sel["membership"])][:12])]
              ) if len(_sel) else mo.md("_Drag a rectangle on the PCA map to draw the molecules inside it._")
    return


@app.cell
def _():
    umap_params = (
        mo.md("**UMAP settings** (start with 15 and 0.1)  \n{n_neighbors}  \n{min_dist}")
        .batch(
            n_neighbors=mo.ui.slider(5, 50, step=5, value=15, label="n_neighbors", show_value=True),
            min_dist=mo.ui.slider(0.0, 0.8, step=0.05, value=0.1, label="min_dist", show_value=True),
        )
        .form(submit_button_label="Refit UMAP", bordered=True)
    )
    umap_params
    return (umap_params,)


@app.cell
def _(map_sample, umap_params):
    _p = umap_params.value or {"n_neighbors": 15, "min_dist": 0.1}
    map_fps = np.array([FP_GEN.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in map_sample["smiles_std"]], dtype=bool)
    with mo.status.spinner(title=f"Fitting UMAP (n_neighbors={_p['n_neighbors']}, min_dist={_p['min_dist']})…"), UMAP_LOCK:
        umap_model = umap.UMAP(n_neighbors=int(_p["n_neighbors"]), min_dist=float(_p["min_dist"]), metric="jaccard",
                               random_state=SEED).fit(map_fps)
    map_sample_umap = map_sample.assign(UMAP1=umap_model.embedding_[:, 0], UMAP2=umap_model.embedding_[:, 1])
    return map_sample_umap, umap_model


@app.cell
def _(features, map_color, map_sample_umap, scaffold_browser):
    _top = features.loc[features["scaffold"] != "", "scaffold"].value_counts().head(8).index
    _sel = scaffold_browser.value["scaffold"].tolist() if len(scaffold_browser.value) else []
    _d = map_frame(map_sample_umap, map_sample_umap[["UMAP1", "UMAP2"]].to_numpy(), map_color.value, _sel, _top)
    fig_umap = map_chart(_d, "UMAP 1", "UMAP 2", "UMAP of Morgan fingerprints (Jaccard distance)")
    umap_chart = mo.ui.altair_chart(fig_umap, chart_selection="interval", legend_selection=False)
    umap_chart
    return fig_umap, umap_chart


@app.cell(hide_code=True)
def _(umap_chart):
    _sel = umap_chart.value
    mo.vstack([mo.md(f"**{len(_sel)} selected** — first 12 drawn:"),
               mol_grid(_sel["smiles_std"].tolist()[:12], [f"{l} · {m}" for l, m in zip(_sel["label"], _sel["membership"])][:12])]
              ) if len(_sel) else mo.md("_Drag a rectangle on the UMAP to draw the molecules inside it. Try an "
                                        "OpenADMET-only island, then a mixed region._")
    return


@app.cell(hide_code=True)
def _():
    mo.callout(
        mo.md("**Reading the maps.** PCA axes mean something (see the loadings); UMAP axes do not, and distances "
              "*between* UMAP islands are not meaningful. Colour by *ChEMBL median pIC50* and by *OpenADMET pIC50* in "
              "turn: each colouring uses one source's own measurements only, on the same joint map."),
        kind="info",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    <a id="outliers"></a>
    ## 8 · Outliers: unusual properties versus structural novelty (110–120′)

    Two different questions:

    1. **Unusual physicochemical properties** — an *Isolation Forest* on the nine standardised descriptors of all
       unique structures (both sources). Isolation Forest isolates points with random splits; points that are
       isolated after few splits get a high anomaly score.
    2. **Structural novelty relative to ChEMBL** — OpenADMET compounds whose nearest ChEMBL neighbour is far
       (low maximum Tanimoto, part 5).

    A compound can be one, the other, both or neither. Flags are **reasons to look**, never reasons to delete, and
    they are computed in the full descriptor / fingerprint space — never from the 2D pictures.
    """)
    return


@app.cell
def _(features):
    _scaler = StandardScaler().fit(features[DESCRIPTORS])
    _z = _scaler.transform(features[DESCRIPTORS])
    _forest = IsolationForest(n_estimators=500, random_state=SEED).fit(_z)
    outlier_scores = features[["smiles_std", "label", "membership"] + DESCRIPTORS].copy()
    outlier_scores["anomaly_score"] = -_forest.score_samples(_z)          # higher = more unusual
    _zdf = pd.DataFrame(_z, columns=DESCRIPTORS)
    outlier_scores["most extreme descriptors"] = [
        ", ".join(f"{c} {'+' if v > 0 else '−'}{abs(v):.1f}σ" for c, v in row.abs().nlargest(2).index.map(lambda c: (c, row[c])))
        for _, row in _zdf.iterrows()
    ]
    outlier_scores["anomaly_rank_percent"] = outlier_scores["anomaly_score"].rank(pct=True, ascending=False) * 100
    return (outlier_scores,)


@app.cell
def _():
    outlier_scope = mo.ui.radio(["OpenADMET", "ChEMBL", "all"], value="OpenADMET", inline=True, label="Rank compounds from")
    outlier_top_k = mo.ui.slider(4, 40, step=4, value=12, label="how many to inspect", show_value=True)
    unusual_percent = mo.ui.slider(0.5, 10, step=0.5, value=2.0, label="'unusual properties' = top % of anomaly scores",
                                   show_value=True)
    novelty_threshold = mo.ui.slider(0.15, 0.5, step=0.05, value=0.3, label="'structurally novel' = nearest ChEMBL T <",
                                     show_value=True)
    mo.vstack([mo.hstack([outlier_scope, outlier_top_k], justify="start", wrap=True, gap=2),
               mo.hstack([unusual_percent, novelty_threshold], justify="start", wrap=True, gap=2)])
    return novelty_threshold, outlier_scope, outlier_top_k, unusual_percent


@app.cell(hide_code=True)
def _(outlier_scope, outlier_scores, outlier_top_k):
    _pool = outlier_scores if outlier_scope.value == "all" else outlier_scores[
        outlier_scores["membership"].isin([outlier_scope.value, "both"])]
    property_candidates = _pool.nlargest(outlier_top_k.value, "anomaly_score")
    mo.vstack(
        [
            mo.md(f"#### Highest Isolation-Forest scores ({outlier_scope.value})\n`most extreme descriptors` gives the "
                  "two descriptors furthest from the mean (in standard deviations) — a quick *why*, not the forest's "
                  "internal logic."),
            mo.ui.table(property_candidates[["label", "membership", "anomaly_score", "most extreme descriptors", "MW", "cLogP",
                                             "TPSA", "HeavyAtoms", "FracSP3"]].round(2).reset_index(drop=True), selection=None, page_size=6),
            mol_grid(property_candidates["smiles_std"].tolist(),
                     [f"{l} · {e.split(',')[0]}" for l, e in zip(property_candidates["label"], property_candidates["most extreme descriptors"])]),
        ]
    )
    return (property_candidates,)


@app.cell(hide_code=True)
def _(nn_table, novelty_threshold, outlier_top_k):
    novel_candidates = nn_table[~nn_table["exact_in_chembl"]].nsmallest(outlier_top_k.value, "nn_tanimoto")
    _n_novel = int((nn_table["nn_tanimoto"] < novelty_threshold.value).sum())
    mo.vstack(
        [
            mo.md(f"#### Least covered OpenADMET compounds\n**{_n_novel}** OpenADMET structures "
                  f"({_n_novel / len(nn_table):.1%}) have no ChEMBL neighbour with T ≥ {novelty_threshold.value:.2f}. "
                  f"The {outlier_top_k.value} most isolated:"),
            mo.ui.table(novel_candidates[["name", "openadmet_pIC50", "nn_tanimoto", "nn_chembl_id"]].reset_index(drop=True), selection=None, page_size=6),
            mol_grid(novel_candidates["smiles_std"].tolist(),
                     [f"{n} · T={t:.2f}" for n, t in zip(novel_candidates["name"], novel_candidates["nn_tanimoto"])]),
        ]
    )
    return (novel_candidates,)


@app.cell(hide_code=True)
def _(nn_table, novelty_threshold, outlier_scores, unusual_percent):
    _oa = nn_table[["smiles_std", "name", "nn_tanimoto"]].merge(outlier_scores[["smiles_std", "anomaly_rank_percent"]], on="smiles_std")
    _oa["unusual properties"] = np.where(_oa["anomaly_rank_percent"] <= unusual_percent.value, "yes", "no")
    _oa["structurally novel"] = np.where(_oa["nn_tanimoto"] < novelty_threshold.value, "yes", "no")
    _ct = pd.crosstab(_oa["unusual properties"], _oa["structurally novel"], margins=True, margins_name="total")
    outlier_crosstab = _oa
    mo.vstack(
        [
            mo.md(f"#### Two kinds of outliers (OpenADMET compounds)\nRows: in the top {unusual_percent.value}% of anomaly "
                  f"scores. Columns: nearest ChEMBL neighbour T < {novelty_threshold.value:.2f}."),
            mo.ui.table(_ct.reset_index(), selection=None, pagination=False),
        ]
    )
    return (outlier_crosstab,)


@app.cell(hide_code=True)
def _(
    features_pca,
    map_sample_pca,
    map_sample_umap,
    novel_candidates,
    property_candidates,
    umap_model,
):
    _flags = pd.concat(
        [property_candidates[["smiles_std", "label"]].assign(flag="unusual properties"),
         novel_candidates[["smiles_std", "name"]].rename(columns={"name": "label"}).assign(flag="structurally novel")]
    ).drop_duplicates("smiles_std")
    _pca_bg = map_sample_pca[["PC1", "PC2"]].rename(columns={"PC1": "x", "PC2": "y"})
    _pca_fg = _flags.merge(features_pca, on="smiles_std").rename(columns={"PC1": "x", "PC2": "y"})
    _umap_bg = map_sample_umap[["UMAP1", "UMAP2"]].rename(columns={"UMAP1": "x", "UMAP2": "y"})
    _in = _flags["smiles_std"].isin(map_sample_umap["smiles_std"])
    _fg_in = _flags[_in].merge(map_sample_umap[["smiles_std", "UMAP1", "UMAP2"]], on="smiles_std").rename(columns={"UMAP1": "x", "UMAP2": "y"}).assign(placed="in the map sample")
    _out_fps = np.array([FP_GEN.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in _flags.loc[~_in, "smiles_std"]], dtype=bool)
    _fg_out = _flags[~_in].copy()
    if len(_fg_out):
        with UMAP_LOCK:
            _xy = umap_model.transform(_out_fps)
        _fg_out["x"], _fg_out["y"] = _xy[:, 0], _xy[:, 1]
    _fg_out["placed"] = "projected onto the fitted map"
    _umap_fg = pd.concat([_fg_in, _fg_out])

    def _panel(bg, fg, title):
        _b = alt.Chart(bg).mark_circle(size=12, color="#cfcfcf").encode(x=alt.X("x:Q", title=None), y=alt.Y("y:Q", title=None))
        _f = alt.Chart(fg).mark_point(size=90, filled=True).encode(
            x="x:Q", y="y:Q",
            color=alt.Color("flag:N", scale=alt.Scale(domain=["unusual properties", "structurally novel"], range=["#c0392b", "#8e44ad"])),
            shape=alt.Shape("placed:N", legend=alt.Legend(title=None)) if "placed" in fg else alt.value("circle"),
            tooltip=["label", "flag"],
        )
        return (_b + _f).properties(width=300, height=280, title=title)

    fig_outlier_maps = alt.hconcat(_panel(_pca_bg, _pca_fg.assign(placed="PCA coordinates"), "PCA"),
                                   _panel(_umap_bg, _umap_fg, "UMAP")).resolve_scale(shape="independent")
    mo.vstack(
        [
            mo.md("#### Where are the flagged compounds on the maps?\nFlagged compounds outside the map sample are "
                  "*projected* onto the fitted maps (PCA: same scaler and axes; UMAP: `transform`), not added to the fit."),
            fig_outlier_maps,
        ]
    )
    return (fig_outlier_maps,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### ✏️ Exercise 5 — why might a flagged outlier still be valid chemistry?

    Pick one compound from each list above (unusual properties; structurally novel). For each, decide whether the
    flag points to (a) a **data problem** (wrong structure, salt not removed, mixture) or (b) **real but rare
    chemistry** (macrocycle, natural product, peptide, very small fragment, approved drug from an unusual class).
    What would you check before trusting — or discarding — its CYP3A4 value?
    """)
    return


@app.cell
def _():
    exercise5_answer = mo.ui.text_area(placeholder="Compound, flag, your diagnosis…", rows=4, full_width=True)
    exercise5_answer
    return


@app.cell(hide_code=True)
def _():
    mo.accordion(
        {
            "🔑 Solution — exercise 5": mo.md(r"""
    The highest anomaly scores typically go to **very large** molecules (macrolides, peptides, complex natural
    products or drugs such as cyclosporin-like structures: many heavy atoms, high TPSA, many rotatable bonds) and to
    **very small, very polar** ones. Many approved drugs in OpenADMET's FDA library are exactly like that — real
    molecules, measured in the same assay as the rest. The right reaction is to *check*, not to delete:

    - is the structure complete and correctly standardised (no counter-ion left, no fragment of a mixture)?
    - is the measured pIC50 below 4 (then very uncertain), and is its interval wide?
    - for a structurally novel compound: is it novel because it is a new chemotype (interesting for the challenge)
      or because of an odd representation (unusual tautomer, metal complex)?

    A compound flagged by both criteria is the best candidate for a closer look; it is also the kind of compound for
    which any model trained on ChEMBL would be extrapolating.
    """)
        }
    )
    return


@app.cell(hide_code=True)
def _(
    features,
    nn_table,
    novel_candidates,
    outlier_crosstab,
    property_candidates,
    scaffold_table,
    struct_all,
    structures,
):
    _n_oa = int(structures["membership"].isin(["OpenADMET", "both"]).sum())
    _both = int((structures["membership"] == "both").sum())
    _both_any = int((struct_all["in_openadmet"] & struct_all["ref_any"]).sum())
    _t = nn_table["nn_tanimoto"]
    _oa_cyc = features[features["membership"].isin(["OpenADMET", "both"]) & (features["scaffold"] != "")]
    _chembl_scaf = set(scaffold_table.loc[scaffold_table["ChEMBL"] > 0, "scaffold"])
    _novel_scaf = float((~_oa_cyc["scaffold"].isin(_chembl_scaf)).mean())
    _verdict = "small" if (_t.median() < 0.4 and _novel_scaf > 0.5) else "partial"
    _low = nn_table[_t < 0.3]
    _props = features.set_index("smiles_std").loc[nn_table["smiles_std"], ["HeavyAtoms", "MW"]].reset_index(drop=True)
    _is_low = (_t < 0.3).to_numpy()
    _ha_low, _ha_rest = np.median(_props["HeavyAtoms"][_is_low]), np.median(_props["HeavyAtoms"][~_is_low])
    _size_note = (
        f"Their median size is {_ha_low:.0f} heavy atoms, against {_ha_rest:.0f} for the other compounds, so poor "
        "coverage is not just a size effect: they are new chemotypes for ChEMBL's CYP3A4 data."
        if abs(_ha_low - _ha_rest) < 3
        else f"They are {'smaller' if _ha_low < _ha_rest else 'larger'} than the rest (median {_ha_low:.0f} vs "
        f"{_ha_rest:.0f} heavy atoms). Small molecules set few fingerprint bits, which caps the Tanimoto they can reach."
    )
    _both_flags = outlier_crosstab[(outlier_crosstab["unusual properties"] == "yes") & (outlier_crosstab["structurally novel"] == "yes")]
    _prop_examples = ", ".join(f"{l} ({e})" for l, e in zip(property_candidates["label"].head(3), property_candidates["most extreme descriptors"].head(3)))
    mo.md(
        f"""
    <a id="answers"></a>
    ## 9 · Evidence-based answers

    *Every number below is computed live from the choices made above. Unless you changed the ChEMBL reference in
    part 5, it uses the workshop default.*

    **How much chemical space overlaps?**
    Of {_n_oa:,} unique OpenADMET CYP3A4 structures, only **{_both:,} ({_both / _n_oa:.1%})** are exactly present in
    the curated ChEMBL CYP3A4 set. That rises to {_both_any} if lower-confidence ChEMBL assays, such as DrugMatrix's,
    are allowed. The median Tanimoto to the nearest *non-identical* ChEMBL structure is **{_t.median():.2f}**:
    {(_t >= 0.4).mean():.1%} of the compounds have a ChEMBL neighbour at T ≥ 0.4, and {(_t >= 0.6).mean():.1%} at
    T ≥ 0.6. **{_novel_scaf:.0%}** of the cyclic OpenADMET compounds have a Bemis–Murcko scaffold that never occurs in
    ChEMBL, and the two sources share only {int((scaffold_table['status'] == 'shared').sum())} scaffolds.

    The overlap is therefore **{_verdict}**. The joint maps show the same picture: OpenADMET occupies regions that
    ChEMBL's CYP3A4 records barely sample. These are smaller, less lipophilic molecules (median MW
    {features.loc[features['membership'] != 'ChEMBL', 'MW'].median():.0f} vs {features.loc[features['membership'] != 'OpenADMET', 'MW'].median():.0f} Da)
    from a diversity library, set against ChEMBL's optimised medicinal-chemistry series.

    **Which challenge compounds have weak ChEMBL coverage?**
    **{len(_low):,} ({len(_low) / len(nn_table):.0%})** have no ChEMBL neighbour with T ≥ 0.30. The most isolated
    are {", ".join(novel_candidates["name"].head(5))}. {_size_note} For these compounds ChEMBL offers no close
    analogue, and any model learned from ChEMBL would be extrapolating.

    **Which unusual compounds deserve inspection, and why?**
    The highest Isolation-Forest scores go to compounds with extreme descriptors, for example {_prop_examples}: long
    flexible chains, very polar or very small molecules. With the current slider settings, **{len(_both_flags)}**
    OpenADMET compounds are flagged *both* as property outliers and as structurally novel, and they are the first to
    inspect. A flag is not an error. Check the structure, the standardisation and the uncertainty of the value
    (is pIC50 below 4?), then keep the compound.

    > Caveat for every answer: ChEMBL and OpenADMET potencies come from different assays. We compared **structures**,
    > and looked at activities only side by side, per source and per assay.
    """
    )
    return


@app.cell
def _():
    export_button = mo.ui.run_button(label="💾 Write the result tables and figures to the outputs folder")
    export_button
    return (export_button,)


@app.cell(hide_code=True)
def _(
    before_after,
    chembl_std,
    export_button,
    features,
    fig_descriptor,
    fig_nn_similarity,
    fig_outlier_maps,
    fig_pca,
    fig_pic50,
    fig_rarefaction,
    fig_umap,
    fig_uncertainty,
    nn_table,
    novel_candidates,
    oa_std,
    outlier_scores,
    property_candidates,
    route_comparison,
    scaffold_table,
):
    _tables = {
        "before_after_counts.csv": before_after,
        "route_comparison.csv": route_comparison,
        "chembl_cyp3a4_measurements_curated.csv": chembl_std.drop(columns=["record_key"]),
        "openadmet_cyp3a4_measurements.csv": oa_std,
        "structures_with_descriptors.csv": features,
        "nearest_chembl_neighbours.csv": nn_table.drop(columns=["query_row"]),
        "scaffolds.csv": scaffold_table,
        "outlier_scores.csv": outlier_scores,
    }
    _figures = {
        "openadmet_uncertainty": fig_uncertainty, "descriptor_distribution": fig_descriptor, "pic50_by_assay": fig_pic50,
        "nearest_neighbour_similarity": fig_nn_similarity, "scaffold_rarefaction": fig_rarefaction, "pca_map": fig_pca,
        "umap_map": fig_umap, "outliers_on_maps": fig_outlier_maps,
    }
    _grids = {
        "least_covered_openadmet.svg": (novel_candidates["smiles_std"], [f"{n} T={t:.2f}" for n, t in zip(novel_candidates["name"], novel_candidates["nn_tanimoto"])]),
        "property_outliers.svg": (property_candidates["smiles_std"], property_candidates["label"].tolist()),
    }
    _written = None
    if export_button.value or os.environ.get("WORKSHOP_EXPORT") == "1":
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        for _name, _df in _tables.items():                       # large tables are gzipped (pandas reads them as is)
            _big = len(_df) > 5000
            _df.to_csv(OUTPUT_DIR / (_name + ".gz" if _big else _name), index=False,
                       compression={"method": "gzip", "mtime": 0} if _big else None)
        _png = 0
        for _name, _fig in _figures.items():
            _fig.save(OUTPUT_DIR / f"{_name}.html")
            try:                                                  # PNG needs the optional vl-convert-python package
                _fig.save(OUTPUT_DIR / f"{_name}.png", scale_factor=2)
                _png += 1
            except Exception:
                pass
        for _name, (_smiles, _legends) in _grids.items():
            (OUTPUT_DIR / _name).write_text(mol_grid_svg(list(_smiles), _legends) or "")
        _written = mo.callout(mo.md(f"Wrote {len(_tables)} tables, {len(_figures)} interactive figures (HTML"
                                    f"{', ' + str(_png) + ' PNG' if _png else ''}) and {len(_grids)} molecule grids "
                                    f"(SVG) to `{OUTPUT_DIR}`."), kind="success")
    mo.vstack(
        [
            _written or mo.md("Or download single tables:"),
            mo.hstack([mo.download(data=_df.to_csv(index=False).encode(), filename=_name, label=_name)
                       for _name, _df in list(_tables.items())[:4]], justify="start", wrap=True),
            mo.hstack([mo.download(data=_df.to_csv(index=False).encode(), filename=_name, label=_name)
                       for _name, _df in list(_tables.items())[4:]], justify="start", wrap=True),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ---
    ### Going further (outside today's core)
    Model training, docking, activity cliffs and other embeddings (t-SNE, TMAP) are deliberately left out. Natural
    next steps: the classical-ML session of the [tutorials](https://github.com/AxelRolov/chemoinformatics_tutorials)
    with a *scaffold split*, and the [OpenADMET CYP challenge](https://huggingface.co/spaces/openadmet/cyp-challenge)
    itself (open until November 2026).

    **References.** Zdrazil *et al.*, *The ChEMBL database in 2023*, Nucleic Acids Res. 2024 · Bento *et al.*,
    *An open source chemical structure curation pipeline using RDKit*, J. Cheminform. 2020 · Bemis & Murcko,
    *The properties of known drugs. 1. Molecular frameworks*, J. Med. Chem. 1996 · Landrum & Riniker, *Combining IC50 or
    Ki values from different sources is a source of significant noise*, J. Chem. Inf. Model. 2024 · Liu, Ting & Zhou,
    *Isolation Forest*, ICDM 2008 · McInnes, Healy & Melville, *UMAP*, arXiv:1802.03426.
    """)
    return


if __name__ == "__main__":
    app.run()
