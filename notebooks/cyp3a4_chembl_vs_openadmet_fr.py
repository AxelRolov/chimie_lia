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
app = marimo.App(width="medium", app_title="CYP3A4 : ChEMBL vs OpenADMET (FR)")

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

    # Où ce notebook s’exécute-t-il ? Dans une copie du dépôt (data/ à côté de notebooks/), sur molab, ou dans un
    # script qui importe les fonctions ci-dessous. Les données pédagogiques en cache viennent de la copie locale
    # si elle existe, sinon du dépôt sur GitHub.
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

    # ---- la cible -----------------------------------------------------------------------------------
    TARGET_ID = "CHEMBL340"          # cytochrome P450 3A4 humain dans ChEMBL
    UNIPROT_ID = "P08684"            # ... et dans UniProt
    CHEMBL_BASE = "https://www.ebi.ac.uk"
    CHEMBL_API = f"{CHEMBL_BASE}/chembl/api/data"
    ACTIVITY_QUERY = {"target_chembl_id": TARGET_ID, "standard_type": "IC50"}
    ACTIVITY_URL = f"{CHEMBL_API}/activity.json?target_chembl_id={TARGET_ID}&standard_type=IC50&limit=1000"
    UI_ACTIVITIES_URL = (
        "https://www.ebi.ac.uk/chembl/explore/activities/"
        "QUERYSTRING:target_chembl_id%3ACHEMBL340%20AND%20standard_type%3A%22IC50%22"
    )
    UI_ASSAYS_URL = "https://www.ebi.ac.uk/chembl/explore/assays/QUERYSTRING:target_chembl_id%3ACHEMBL340"

    # ---- défi CYP d’OpenADMET, épinglé sur une révision précise du jeu de données ----------------------
    HF_DATASET = "openadmet/cyp-challenge-train-test"
    HF_REVISION = "146282d2ce1532f1803679a834a9bec9dac1e482"
    _HF_FILES = f"https://huggingface.co/datasets/{HF_DATASET}/resolve/{HF_REVISION}"
    OPENADMET_TRAIN_URL = f"{_HF_FILES}/cyp-challenge-TRAIN_inhibition.csv"
    OPENADMET_TEST_URL = f"{_HF_FILES}/cyp-challenge-TEST-BLINDED.csv"
    CYP3A4 = "CYP3A4_pIC50_direct_inhibition"
    OPENADMET_EXPECTED = {"rows": 4905, "cyp3a4_values": 2335}

    # ---- données pédagogiques en cache (préparées avant le cours avec scripts/prepare_data.py) ----------
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

    # ---- paramètres de l’analyse ---------------------------------------------------------------------
    SEED = 42
    MAX_PER_SOURCE = 1000            # au plus 1 000 structures uniques par source sur les cartes 2D
    WEAK_PIC50 = 4.0                 # en dessous, les estimations de puissance sont extrapolées et imprécises
    FP_RADIUS, FP_BITS = 2, 2048     # empreinte de Morgan binaire (type ECFP4)
    FP_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)
    DESCRIPTORS = ["MW", "cLogP", "TPSA", "HBD", "HBA", "RotB", "AromRings", "FracSP3", "HeavyAtoms"]
    DESCRIPTOR_NAMES = {
        "MW": "masse molaire (Da)", "cLogP": "logP calculé (Crippen)", "TPSA": "surface polaire topologique (Å²)",
        "HBD": "donneurs de liaison H", "HBA": "accepteurs de liaison H", "RotB": "liaisons rotatives", "AromRings": "cycles aromatiques",
        "FracSP3": "fraction de carbones sp³", "HeavyAtoms": "atomes lourds",
    }
    SOURCES = ["ChEMBL", "OpenADMET", "both"]
    SOURCE_COLORS = ["#3b6fb6", "#e07b39", "#7d4fb8"]
    LFC = rdMolStandardize.LargestFragmentChooser(preferOrganic=True)
    UNCHARGER = rdMolStandardize.Uncharger()
    # numba (utilisé par UMAP) plante si deux threads d’un même processus l’exécutent en même temps, ce qui arrive
    # quand plusieurs sessions partagent un serveur `marimo run`. Un verrou par processus, stocké sur le module umap, les sérialise.
    UMAP_LOCK = umap.__dict__.setdefault("_workshop_lock", threading.Lock())

    # ---- le schéma commun vers lequel les deux voies ChEMBL sont converties ----------------------------
    COMMON_COLUMNS = [
        "route", "activity_id", "record_key", "molecule_chembl_id", "smiles", "assay_chembl_id",
        "assay_description", "assay_type", "bao_label", "assay_organism", "confidence_score",
        "target_chembl_id", "target_organism", "target_type", "document_chembl_id", "document_year",
        "standard_type", "standard_relation", "standard_value", "standard_units", "pchembl_value",
        "data_validity_comment", "potential_duplicate", "activity_comment",
    ]
    UI_ACTIVITY_COLUMNS = {   # en-tête de l’interface web ChEMBL -> schéma commun
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

    # ---- politique de curation (appliquée à l’identique à chaque voie), dans l’ordre de vérification des règles ---
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

    # ---- mots-clés de description d’essai servant à pré-classer les essais ChEMBL (revus par l’enseignant) ----
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

    # ---- molécules utilisées dans les démonstrations (SMILES isomériques issus de PubChem) ---------------
    REFERENCE_DRUGS = {
        "ketoconazole": ("CC(=O)N1CCN(CC1)C2=CC=C(C=C2)OC[C@@H]3CO[C@@](O3)(CN4C=CN=C4)C5=C(C=C(C=C5)Cl)Cl",
                         "inhibiteur puissant du CYP3A4 (antifongique azolé) ; le « perpétrateur » classique d’interactions médicamenteuses (DDI)"),
        "itraconazole": ("CCC(C)N1C(=O)N(C=N1)C2=CC=C(C=C2)N3CCN(CC3)C4=CC=C(C=C4)OC[C@H]5CO[C@](O5)(CN6C=NC=N6)"
                         "C7=C(C=C(C=C7)Cl)Cl", "inhibiteur puissant du CYP3A4 (antifongique triazolé)"),
        "miconazole": ("C1=CC(=C(C=C1Cl)Cl)COC(CN2C=CN=C2)C3=C(C=C(C=C3)Cl)Cl", "antifongique azolé, inhibiteur des CYP"),
        "ritonavir": ("CC(C)C1=NC(=CS1)CN(C)C(=O)N[C@@H](C(C)C)C(=O)N[C@@H](CC2=CC=CC=C2)C[C@@H]([C@H]"
                      "(CC3=CC=CC=C3)NC(=O)OCC4=CN=CS4)O", "inhibiteur puissant du CYP3A4, utilisé comme booster pharmacocinétique"),
        "cobicistat": ("CC(C)C1=NC(=CS1)CN(C)C(=O)N[C@@H](CCN2CCOCC2)C(=O)N[C@H](CC[C@H](CC3=CC=CC=C3)NC(=O)"
                       "OCC4=CN=CS4)CC5=CC=CC=C5", "booster conçu à partir du ritonavir"),
        "verapamil": ("CC(C)C(CCCN(C)CCC1=CC(=C(C=C1)OC)OC)(C#N)C2=CC(=C(C=C2)OC)OC", "inhibiteur modéré du CYP3A4"),
        "erythromycin": ("CC[C@@H]1[C@@]([C@@H]([C@H](C(=O)[C@@H](C[C@@]([C@@H]([C@H]([C@@H]([C@H](C(=O)O1)C)"
                         "O[C@H]2C[C@@]([C@H]([C@@H](O2)C)O)(C)OC)C)O[C@H]3[C@@H]([C@H](C[C@H](O3)C)N(C)C)O)(C)O)C)C)O)"
                         "(C)O", "macrolide ; inhibiteur temps-dépendant et substrat du CYP3A4"),
        "midazolam": ("CC1=NC=C2N1C3=C(C=C(C=C3)Cl)C(=NC2)C4=CC=CC=C4F", "substrat sensible du CYP3A4 : la sonde clinique"),
        "alprazolam": ("CC1=NN=C2N1C3=C(C=C(C=C3)Cl)C(=NC2)C4=CC=CC=C4", "substrat du CYP3A4 (triazolobenzodiazépine)"),
        "triazolam": ("CC1=NN=C2N1C3=C(C=C(C=C3)Cl)C(=NC2)C4=CC=CC=C4Cl", "substrat du CYP3A4"),
        "testosterone": ("C[C@]12CC[C@H]3[C@H]([C@@H]1CC[C@@H]2O)CCC4=CC(=O)CC[C@]34C",
                         "substrat sonde du CYP3A4 (6β-hydroxylation)"),
        "epitestosterone": ("C[C@]12CC[C@H]3[C@H]([C@@H]1CC[C@H]2O)CCC4=CC(=O)CC[C@]34C", "17-épimère de la testostérone"),
        "progesterone": ("CC(=O)[C@H]1CC[C@@H]2[C@@]1(CC[C@H]3[C@H]2CCC4=CC(=O)CC[C@]34C)C", "stéroïde, substrat du CYP3A4"),
        "nifedipine": ("CC1=C(C(C(=C(N1)C)C(=O)OC)C2=CC=CC=C2[N+](=O)[O-])C(=O)OC", "substrat sonde du CYP3A4"),
        "BFC": ("C1=CC=C(C=C1)COC2=CC3=C(C=C2)C(=CC(=O)O3)C(F)(F)F",
                "7-benzyloxy-4-trifluorométhylcoumarine, une sonde fluorogène du CYP3A4"),
    }


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    [![Ouvrir dans molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/AxelRolov/chimie_lia/blob/main/notebooks/cyp3a4_chembl_vs_openadmet_fr.py)

    # Explorer les inhibiteurs du CYP3A4 : ChEMBL versus OpenADMET

    *Version française — [English version](https://molab.marimo.io/github/AxelRolov/chimie_lia/blob/main/notebooks/cyp3a4_chembl_vs_openadmet.py)*

    **Un atelier pratique de chémoinformatique de deux heures** — RDKit, pandas, scikit-learn et UMAP dans un notebook
    [marimo](https://marimo.io) réactif.

    > **La question du jour.** *Quelle est la couverture, par ChEMBL, de l’espace chimique des composés OpenADMET ?*
    >
    > Nous y répondons sans entraîner le moindre modèle : en nettoyant deux jeux de données avec les mêmes règles et en
    > les comparant par la **similarité**, les **squelettes** (« scaffolds » de Bemis–Murcko), les **cartes 2D** et les
    > **valeurs atypiques** (outliers).

    > **D’où vient cet atelier.** Cet atelier s’appuie sur les
    > [tutoriels de chémoinformatique](https://github.com/AxelRolov/chemoinformatics_tutorials) (séances 02–04), qui
    > compilent et mettent à jour des supports pédagogiques libres issus de **TeachOpenCADD** (laboratoire Volkamer, CC BY 4.0), des
    > *Practical Cheminformatics Tutorials* de **Pat Walters** (MIT), du cours *AI for Chemistry* du **groupe Schwaller** (MIT) et de
    > l’atelier de chémoinformatique de **MolSSI** (MIT). Données : **ChEMBL** (EMBL-EBI, CC BY-SA 3.0) et le **défi OpenADMET
    > d’inhibition des CYP** (Octant / OpenADMET, Apache-2.0). Les crédits complets figurent dans
    > [`CREDITS.md`](https://github.com/AxelRolov/chimie_lia/blob/main/CREDITS.md). L’atelier a été conçu avec
    > **Claude** (Anthropic) et relu par l’enseignant ; les erreurs sont les nôtres, et non celles des auteurs d’origine.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Plan de la séance

    | durée | partie | ce que vous aurez à la fin |
    |---|---|---|
    | 0–10′ | [0 · Introduction](#intro) | le CYP3A4, les deux jeux de données, les SMILES, l’IC50 et le pIC50 |
    | 10–20′ | [1 · ChEMBL à la main](#manual) | un CSV exporté depuis le site de ChEMBL, lu avec pandas |
    | 20–30′ | [2 · ChEMBL par l’API](#api) | les mêmes enregistrements récupérés par programme — et la preuve que les deux voies concordent |
    | 30–45′ | [3 · Chargement et nettoyage](#clean) | OpenADMET chargé ; structures valides et standardisées ; un rapport sur les valeurs manquantes et les doublons |
    | 45–60′ | [4 · Descripteurs](#eda) | distributions des propriétés et des activités comparées entre les sources |
    | 60–80′ | [5 · Similarité](#similarity) | le plus proche voisin ChEMBL de chaque composé OpenADMET |
    | 80–95′ | [6 · Squelettes](#scaffolds) | squelettes de Bemis–Murcko communs, partagés et nouveaux |
    | 95–110′ | [7 · ACP (PCA) et UMAP](#maps) | une carte conjointe des deux sources |
    | 110–120′ | [8 · Valeurs atypiques](#outliers) et [9 · Réponses](#answers) | composés inhabituels, expliqués ; réponses fondées sur les données |

    Il y a **cinq courts exercices** (✏️). Chacun a une solution cachée (🔑) — essayez d’abord, puis regardez.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.callout(
        mo.md(r"""
    **Nouveau sur marimo ?** C’est un notebook *réactif* : quand vous déplacez un curseur ou modifiez une cellule, toutes
    les cellules qui en dépendent se ré-exécutent automatiquement, de sorte que ce que vous voyez est toujours cohérent
    — aucun état caché, pas besoin de « tout ré-exécuter ».

    - Exécutez une cellule avec **Maj + Entrée** (ou le bouton ▶). Le code des cellules de texte est masqué ; cliquez sur
      une cellule pour le voir.
    - Une variable globale ne peut être définie que dans **une seule cellule**. Dans les cellules d’exercice, faites
      commencer vos propres variables par un tiret bas (`_df`, `_n`) pour qu’elles restent locales à la cellule.
    - Les widgets (curseurs, listes déroulantes, tableaux sur lesquels vous pouvez cliquer) pilotent l’analyse. Les
      graphiques dont le curseur est ⬚ permettent de **glisser pour sélectionner** : les molécules sélectionnées sont
      dessinées sous le graphique.
    """),
        kind="info",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.sidebar(
        [
            mo.md("### 🧪 Atelier CYP3A4\n*ChEMBL vs OpenADMET*"),
            mo.nav_menu(
                {
                    "#intro": "0 · Introduction",
                    "#manual": "1 · ChEMBL à la main",
                    "#api": "2 · ChEMBL par l’API",
                    "#clean": "3 · Chargement et nettoyage",
                    "#eda": "4 · Descripteurs",
                    "#similarity": "5 · Similarité",
                    "#scaffolds": "6 · Squelettes",
                    "#maps": "7 · ACP (PCA) et UMAP",
                    "#outliers": "8 · Valeurs atypiques",
                    "#answers": "9 · Réponses et export",
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

    Le **CYP3A4** est l’enzyme du métabolisme des médicaments la plus importante du foie et de l’intestin humains : il
    oxyde une grande partie des médicaments commercialisés. Un composé qui **inhibe** le CYP3A4 ralentit la clairance de
    tous les substrats du CYP3A4 co-administrés — le kétoconazole, par exemple, multiplie par plus de dix l’exposition
    au sédatif midazolam. C’est pourquoi l’inhibition du CYP3A4 est mesurée très tôt dans tout projet de découverte de
    médicaments.

    | | **ChEMBL** — `CHEMBL340` | **Défi CYP d’OpenADMET** |
    |---|---|---|
    | nature | données de la littérature et des brevets, curées, issues de nombreux laboratoires | un seul jeu de données de défi à l’aveugle, un seul laboratoire (Octant) |
    | essais | des centaines de protocoles : microsomes hépatiques ou enzyme recombinante, midazolam, testostérone, substrats sondes fluorogènes… | un seul protocole : CYP3A4 recombinant, substrat sonde fluorogène **DBOMF**, courbe dose–réponse à 12 points, ajustement bayésien avec intervalles de crédibilité |
    | chimie | séries de chimie médicinale choisies par de nombreux projets | jeu de diversité Enamine + médicaments approuvés par la FDA |
    | mesure | `IC50` avec une relation (`=`, `>`, `<`) et des unités | `pIC50` avec un intervalle à 95 % et un écart-type |

    > ⚠️ **Même protéine ≠ mesures interchangeables.** Une IC50 dépend de la source de l’enzyme, du substrat sonde, de sa
    > concentration, de la préincubation et de la détection. OpenADMET distingue aussi l’inhibition **directe** de
    > l’inhibition **temps-dépendante** (TDI, préincubation avec NADPH). Dans cet atelier, nous comparons les **structures**
    > conjointement, mais nous gardons toujours les **activités** séparées par source et par essai. Nous ne les moyennons jamais entre sources.

    Le **SMILES** code une molécule sous forme de texte (`c1ccccc1O` est le phénol). L’**IC50** est la concentration qui
    réduit de moitié l’activité de l’enzyme ; comme elle s’étend sur de nombreux ordres de grandeur, nous utilisons

    $$\mathrm{pIC50} = -\log_{10}\big(\mathrm{IC50}\,[\mathrm{M}]\big) = 9 - \log_{10}\big(\mathrm{IC50}\,[\mathrm{nM}]\big).$$

    Plus le pIC50 est élevé, plus le composé est puissant. Essayez le convertisseur :
    """)
    return


@app.cell
def _():
    ic50_value = mo.ui.number(start=0.001, stop=1_000_000, step=0.001, value=1.0, label="IC50")
    ic50_unit = mo.ui.dropdown({"nM": 1.0, "µM": 1e3, "mM": 1e6}, value="µM", label="unité")
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
            mo.vstack([_ruler, mo.md("<small>Zone ombrée : pIC50 < 4 (IC50 > 100 µM) — faible, souvent extrapolée, "
                                     "à interpréter avec prudence.</small>")]),
        ],
        align="center",
    )
    return


@app.cell
def _():
    drug_picker = mo.ui.dropdown(options=list(REFERENCE_DRUGS), value="ketoconazole", label="Molécule d’exemple",
                                 searchable=True)
    drug_picker
    return (drug_picker,)


@app.cell
def _(drug_picker):
    smiles_box = mo.ui.text(value=REFERENCE_DRUGS[drug_picker.value][0], label="SMILES — modifiez-le !", full_width=True)
    smiles_box
    return (smiles_box,)


@app.cell(hide_code=True)
def _(drug_picker, smiles_box):
    _mol = Chem.MolFromSmiles(smiles_box.value)
    if _mol is None:
        _out = mo.callout("RDKit ne parvient pas à lire ce SMILES — vérifiez les parenthèses, les chiffres de fermeture de cycle et les valences.",
                          kind="danger")
    else:
        _role = REFERENCE_DRUGS[drug_picker.value][1] if smiles_box.value == REFERENCE_DRUGS[drug_picker.value][0] else "votre molécule"
        _out = mo.hstack(
            [
                mol_grid([smiles_box.value], [drug_picker.value if _role != "votre molécule" else "votre molécule"],
                         per_row=1, size=(330, 250)),
                mo.md(
                    f"**{_role}**\n\n"
                    f"- formule brute : {rdMolDescriptors.CalcMolFormula(_mol)}\n"
                    f"- masse molaire {Descriptors.MolWt(_mol):.1f} Da · cLogP {Descriptors.MolLogP(_mol):.2f}\n"
                    f"- {_mol.GetNumHeavyAtoms()} atomes lourds · "
                    f"{len(Chem.FindMolChiralCenters(_mol, includeUnassigned=True, useLegacyImplementation=False))} centres stéréogènes\n"
                    f"- SMILES canonique (RDKit) :\n\n  <small>`{Chem.MolToSmiles(_mol)}`</small>"
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
    """Dessine une liste de SMILES sous forme de grille SVG (RDKit). `highlights` est une liste facultative de listes d’indices d’atomes."""
    svg = mol_grid_svg(smiles, legends, per_row, size, highlights)
    if svg is None:
        return mo.md("_Aucune structure à dessiner._")
    svg = svg.replace("<svg ", "<svg style='max-width:100%;height:auto' ", 1)     # réduire pour tenir sur les écrans étroits
    return mo.Html(f'<div style="background:white;display:inline-block;border-radius:6px;max-width:100%">{svg}</div>')


@app.function
def mol_grid_svg(smiles, legends=None, per_row=4, size=(210, 165), highlights=None):
    """Le texte SVG derrière `mol_grid` (None si aucun SMILES ne peut être interprété)."""
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
    drawer.drawOptions().drawMolsSameScale = False          # chaque molécule remplit son propre panneau
    drawer.DrawMolecules(mols, legends=labels, highlightAtoms=marks if highlights is not None else None)
    drawer.FinishDrawing()
    return drawer.GetDrawingText().replace("<?xml version='1.0' encoding='iso-8859-1'?>", "")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### `CHEMBL340` est-il vraiment le CYP3A4 humain ?

    Avant de télécharger quoi que ce soit, vérifiez la cible : organisme, accession UniProt et type de cible. Une cible de
    type *protein family* ou *protein complex* mélangerait d’autres enzymes à nos données.
    """)
    return


@app.function
def chembl_get(url, params=None, retries=5, timeout=60):
    """Interroge une URL de l’API ChEMBL (GET) et renvoie le JSON, avec de nouvelles tentatives en cas de délai dépassé, de limite de débit ou d’erreur serveur."""
    problem = None
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            if response.status_code == 200:
                return response.json()
            problem = f"HTTP {response.status_code}"
            if response.status_code < 500 and response.status_code != 429:
                break                                   # une erreur client ne disparaît pas en réessayant
        except requests.RequestException as exc:
            problem = type(exc).__name__
        time.sleep(min(2**attempt, 30))                 # attente croissante : 1, 2, 4, 8 … s
    raise RuntimeError(f"ChEMBL request failed ({problem}) after {attempt + 1} attempt(s): {url}")


@app.function
def verify_target(target):
    """Compare une fiche cible ChEMBL à ce que l’on attend pour le CYP3A4 humain."""
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
        target_origin = "API ChEMBL en direct"
    except Exception as _exc:
        _bytes, _where = read_cached("api_target")
        target_record = json.loads(_bytes)
        target_origin = f"instantané en cache ({_where}) — l’API en direct était inaccessible : {_exc}"
    target_checks = verify_target(target_record)
    mo.vstack(
        [
            mo.md(f"Fiche de la cible issue de : **{target_origin}**"),
            mo.ui.table(target_checks, selection=None, pagination=False),
            mo.callout("Tous les contrôles sont réussis — CHEMBL340 est bien la protéine humaine unique CYP3A4 (UniProt P08684).", kind="success")
            if (target_checks["ok"] == "✅").all()
            else mo.callout("Un contrôle de la cible a échoué : arrêtez-vous et examinez la fiche avant de poursuivre.", kind="danger"),
        ]
    )
    return


@app.function
def read_cached(key):
    """Octets d’un fichier pédagogique en cache et une note indiquant leur provenance (copie locale d’abord, sinon GitHub)."""
    relative = CACHED[key]
    local = REPO_ROOT / relative
    if local.is_file():
        return local.read_bytes(), f"copie locale `{relative}`"
    url = f"{REPO_RAW}/{relative}"
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    return response.content, f"instantané du dépôt `{relative}` (récupéré sur GitHub)"


@app.function
def sha256_hex(data):
    return hashlib.sha256(data).hexdigest()


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    <a id="manual"></a>
    ## 1 · ChEMBL, option 1 — téléchargement manuel (10–20′)

    1. Ouvrez [ChEMBL](https://www.ebi.ac.uk/chembl/), tapez **CHEMBL340** dans la zone de recherche et ouvrez la fiche
       (report card) de la cible.
    2. **Vérifiez la cible** : *Cytochrome P450 3A4*, type *SINGLE PROTEIN*, organisme *Homo sapiens*, accession UniProt
       *P08684* (section des références croisées).
    3. Dans *Associated Bioactivities*, ouvrez les activités de cette cible. Dans le navigateur d’activités, cochez
       **Standard Type → IC50** dans les filtres de gauche. (Raccourci : [ce lien]({UI_ACTIVITIES_URL}) ouvre directement
       le tableau filtré — 13 887 enregistrements dans ChEMBL 37.)
    4. Cliquez sur **CSV** au-dessus du tableau. ChEMBL prépare le fichier sur son serveur (≈ 30 s), puis propose un
       **.zip** contenant un seul `.csv`.
    5. L’export des activités indique la description de l’essai mais **pas le score de confiance de l’essai**, dont nous
       avons besoin. Ouvrez les [essais de CHEMBL340]({UI_ASSAYS_URL}) et exportez-les aussi en CSV : nous joindrons les
       deux tableaux sur `Assay ChEMBL ID`.
    6. Déposez les deux fichiers ci-dessous (zip ou csv). Pas le temps, ou pas de réseau ? Le notebook continue avec un
       **export en cache réalisé de la même façon** — toujours signalé comme tel.

    *Si une colonne dont vous avez besoin manque dans un export (les structures, par exemple), exportez le tableau
    associé — les composés avec le bouton « Related compounds », les essais depuis le navigateur d’essais — et joignez-le
    sur `Molecule ChEMBL ID` ou `Assay ChEMBL ID`, exactement comme nous le faisons ici pour les scores de confiance.*
    """)
    return


@app.cell
def _():
    ui_activity_upload = mo.ui.file(filetypes=[".zip", ".csv", ".tsv", ".txt"], kind="area",
                                    label="Votre export **d’activités** ChEMBL (zip ou csv)")
    ui_assay_upload = mo.ui.file(filetypes=[".zip", ".csv", ".tsv", ".txt"], kind="area",
                                 label="Votre export **d’essais** ChEMBL (facultatif)")
    mo.hstack([ui_activity_upload, ui_assay_upload], widths="equal", gap=1)
    return ui_activity_upload, ui_assay_upload


@app.function
def read_ui_export(data, filename=""):
    """Lit un tableau exporté depuis l’interface web de ChEMBL (csv/tsv, éventuellement zippé) comme du texte, en détectant le séparateur.

    Renvoie le DataFrame (chaque cellule est une chaîne, rien n’est converti) et un dict décrivant le fichier."""
    inner = filename
    if data[:2] == b"PK":                                          # archive zip, telle que ChEMBL la fournit
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
    except csv.Error:                                              # repli sur le séparateur candidat le plus fréquent
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
        ui_upload_error = f"Impossible de lire le fichier déposé : {_exc}"
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
                mo.md(f"**{title}** — {origin} : `{info['file']}`"),
                mo.hstack(
                    [
                        mo.stat(info["delimiter"], label="séparateur", bordered=True),
                        mo.stat(f"{info['rows']:,}", label="lignes", bordered=True),
                        mo.stat(str(info["columns"]), label="colonnes", bordered=True),
                    ],
                    justify="start", wrap=True,
                ),
                mo.md("Les deux premières lignes, exactement telles qu’elles figurent dans le fichier :"),
                mo.md(f"```text\n{info['first lines']}\n```"),
                mo.callout(
                    mo.md(f"Noms de colonnes dupliqués dans l’en-tête : {', '.join(f'`{d}`' for d in _dup)}. "
                          "pandas renomme silencieusement la seconde (`Compounds.1`) — vérifiez laquelle est laquelle avant de l’utiliser."),
                    kind="warn",
                )
                if _dup
                else mo.md(""),
                mo.md("**Les valeurs manquantes se présentent sous deux formes** — chaînes vides *et* texte `None` :"),
                mo.ui.table(_missing, selection=None, page_size=6),
                mo.md("**`Standard Relation` est entouré de guillemets dans la cellule** (`'='`, et non `=`) : il faut le nettoyer avant "
                      "de le comparer à l’API.") if _relations is not None else mo.md(""),
                mo.ui.table(_relations, selection=None, pagination=False) if _relations is not None else mo.md(""),
                mo.accordion({f"Les {info['columns']} noms de colonnes": mo.md(", ".join(f"`{c}`" for c in table.columns))}),
            ]
        )

    _acts = (ui_uploaded_activities, ui_uploaded_activities_info, "votre dépôt") if ui_uploaded_activities is not None else (
        ui_cached_activities, ui_cached_activities_info, f"instantané en cache, {ui_cached_origin}")
    _assays = (ui_uploaded_assays, ui_uploaded_assays_info, "votre dépôt") if ui_uploaded_assays is not None else (
        ui_cached_assays, ui_cached_assays_info, "instantané en cache")
    mo.vstack(
        [
            mo.callout(ui_upload_error, kind="danger") if ui_upload_error else mo.md(""),
            mo.ui.tabs(
                {
                    "Export des activités": _inspect(_acts[0], _acts[1], "Export des activités", _acts[2]),
                    "Export des essais": _inspect(_assays[0], _assays[1], "Export des essais", _assays[2]),
                }
            ),
        ]
    )
    return


@app.function
def record_key(df):
    """Identité composite d’un enregistrement d’activité, disponible dans les deux voies (l’export web n’a pas d’identifiant d’activité)."""
    value = df["standard_value"].map(lambda v: "" if pd.isna(v) else repr(float(v)))
    parts = [df["molecule_chembl_id"], df["assay_chembl_id"], df["document_chembl_id"],
             df["standard_relation"].fillna(""), value, df["standard_units"].fillna("")]
    return parts[0].astype(str).str.cat([p.astype(str) for p in parts[1:]], sep="|")


@app.function
def normalize_ui(activities, assays):
    """Convertit un export web ChEMBL (activités + essais) vers le schéma commun."""
    missing = [c for c in UI_ACTIVITY_COLUMNS if c not in activities.columns]
    if missing:
        raise ValueError("the activity export lacks " + ", ".join(missing)
                         + " — export all columns, or join the related compound/assay export by ChEMBL ID")
    if not {"Assay ChEMBL ID", "Confidence Score"} <= set(assays.columns):
        raise ValueError("the assay export needs 'Assay ChEMBL ID' and 'Confidence Score'")
    df = activities.rename(columns=UI_ACTIVITY_COLUMNS)[list(UI_ACTIVITY_COLUMNS.values())]
    df = df.replace({"": np.nan, "None": np.nan})                       # les deux formes de « manquant »
    df["standard_relation"] = df["standard_relation"].str.strip("'")    # "'='"  ->  "="
    for col in ("standard_value", "pchembl_value", "potential_duplicate", "document_year"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    confidence = assays.drop_duplicates("Assay ChEMBL ID").set_index("Assay ChEMBL ID")["Confidence Score"]
    df["confidence_score"] = pd.to_numeric(df["assay_chembl_id"].map(confidence), errors="coerce")  # jointure par identifiant
    df["activity_id"] = pd.array([pd.NA] * len(df), dtype="Int64")      # absent de l’export web
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
                manual_upload_note = ("Votre export d’activités a été joint à l’export d’essais **en cache** — sans problème si les deux "
                                      "proviennent de la même version de ChEMBL.")
        except ValueError as _exc:
            manual_upload_note = f"Votre dépôt n’a pas pu être converti vers le schéma commun : {_exc}"

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
            mo.md("### Conversion de l’export web vers un schéma commun\nLes deux voies aboutissent aux mêmes colonnes, de sorte que "
                  "tout ce qui suit est partagé. Notez la relation nettoyée et le score de confiance joint :"),
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
    ## 2 · ChEMBL, option 2 — téléchargement par l’API (20–30′)

    Les mêmes enregistrements sont disponibles via l’API REST de ChEMBL. Une URL correspond à une requête :

    ```
    {ACTIVITY_URL}
    ```

    La réponse est une *page* JSON d’au plus 1 000 enregistrements, accompagnée de `page_meta` avec le `total_count` et
    l’URL de la page `next`. Nous suivons `next` jusqu’à ce qu’il soit vide, nous réessayons en cas de délai dépassé, puis
    nous récupérons les **métadonnées des essais** (score de confiance, description, format) par lots de 100
    identifiants d’essai. Le client officiel
    [`chembl_webresource_client`](https://github.com/chembl/chembl_webresource_client) (utilisé à la séance 03 des
    tutoriels) fait la même pagination à votre place ; ici nous utilisons `requests` tout simple pour que rien ne soit caché.
    ([Documentation de l’API](https://chembl.gitbook.io/chembl-interface-documentation/web-services/chembl-data-web-services))
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
                mo.md(f"**Aperçu en direct** (`limit=2`) : ChEMBL recense **{_page['page_meta']['total_count']:,}** enregistrements IC50 "
                      "pour le CYP3A4. Voici `page_meta` et le premier enregistrement :"),
                mo.hstack([mo.tree(_page["page_meta"]), mo.json({k: _page["activities"][0].get(k) for k in _keys})],
                          widths=[1, 2]),
            ]
        )
    except Exception as _exc:
        _api_peek = mo.callout(f"L’API en direct n’est pas joignable pour le moment ({_exc}). Le notebook continue avec "
                               "l’instantané en cache.", kind="warn")
    _api_peek
    return


@app.function
def fetch_activity_pages(query=None, page_size=1000, on_page=None):
    """Télécharge toutes les pages du point d’accès « activity » de ChEMBL pour `query` ; renvoie la liste des enregistrements bruts."""
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
    """Métadonnées des essais pour une liste d’identifiants ChEMBL d’essais, récupérées par lots."""
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
    """Enregistre les enregistrements bruts de l’API en JSON lines compressé gzip (octets reproductibles : pas d’horodatage dans l’en-tête gzip)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(r, sort_keys=True) + "\n" for r in records).encode("utf-8")
    with open(path, "wb") as fh, gzip.GzipFile(fileobj=fh, mode="wb", mtime=0, filename="") as gz:
        gz.write(payload)
    return sha256_hex(path.read_bytes())


@app.function
def read_jsonl(data):
    """Enregistrements bruts issus de JSON lines (compressé gzip), sous forme de DataFrame avec les valeurs telles que l’API les a envoyées."""
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    return pd.DataFrame.from_records([json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()])


@app.cell
def _():
    api_fetch_button = mo.ui.run_button(
        label="⬇️ Télécharger maintenant depuis l’API ChEMBL tous les enregistrements IC50 de CYP3A4 + les métadonnées des essais (≈ 1–2 min)",
        kind="success",
    )
    api_fetch_button
    return (api_fetch_button,)


@app.cell
def _(api_fetch_button):
    live_api = None
    if not api_fetch_button.value:
        _msg = mo.md("_Aucun téléchargement en direct dans cette session pour l’instant — le reste du notebook utilise l’**instantané "
                     "en cache de l’API** ci-dessous. Cliquez sur le bouton pour récupérer vous-même les données actuelles._")
    else:
        try:
            _status = chembl_get(f"{CHEMBL_API}/status.json", retries=2, timeout=30)
            _total = chembl_get(f"{CHEMBL_API}/activity.json", {**ACTIVITY_QUERY, "limit": 1})["page_meta"]["total_count"]
            with mo.status.progress_bar(total=math.ceil(_total / 1000), title="Pages d’activités ChEMBL") as _bar:
                _records = fetch_activity_pages(on_page=lambda done, total: _bar.update())
            _ids = sorted({r["assay_chembl_id"] for r in _records})
            with mo.status.progress_bar(total=math.ceil(len(_ids) / 100), title="Métadonnées des essais (lots de 100)") as _bar2:
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
                mo.md(f"Téléchargés : **{len(_records):,}** enregistrements d’activité et **{len(_assays):,}** essais depuis "
                      f"**{live_api['release']}** le {_stamp} (UTC). Fichiers bruts enregistrés dans `{LIVE_CACHE_DIR.name}/`. "
                      "Sélectionnez *API · live download* ci-dessous pour les utiliser."),
                kind="success",
            )
        except Exception as _exc:
            _msg = mo.callout(
                mo.md(f"**Le téléchargement en direct a échoué** ({_exc}). Rien n’a été remplacé : le notebook continue d’utiliser "
                      "l’instantané en cache, identifié ci-dessous."),
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
        _observed = sha256_hex(read_cached(_key)[0])         # recalcule chaque somme de contrôle à partir des octets utilisés
        _rows.append({"file": _info["path"], "rows / records": _info.get("records"),
                      "sha256 (recorded)": _info["sha256"][:16] + "…",
                      "check": "✅ identique" if _observed == _info["sha256"] else "❌ différent"})
    _c = provenance["chembl"]
    mo.vstack(
        [
            mo.md(
                f"### L’instantané en cache et sa provenance\n"
                f"- version ChEMBL **{_c['release']}** (publiée le {_c['release_date']}), récupérée le **{_c['retrieved_utc']}**\n"
                f"- requête API : `{_c['activity_query_url']}` → **{_c['n_activity_records']:,}** enregistrements d’activité, "
                f"**{_c['n_assays']:,}** essais\n"
                f"- exports du site web : activités `{_c['web_export_activity_query']}` ; essais `{_c['web_export_assay_query']}`\n"
                f"- OpenADMET : `{provenance['openadmet']['dataset']}` @ `{provenance['openadmet']['revision'][:12]}…`\n"
                f"- chargé depuis : {api_cached_origin}"
            ),
            mo.ui.table(pd.DataFrame(_rows), selection=None, pagination=False),
        ]
    )
    return


@app.function
def normalize_api(activities, assays, target):
    """Convertit les enregistrements d’activité bruts de l’API (+ leurs essais et l’enregistrement de la cible) vers le schéma commun."""
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
    """Compare les tableaux curés de plusieurs voies à une voie de référence, par clé composite d’enregistrement.

    `curated` associe l’étiquette d’une voie à son tableau de mesures curé. Renvoie une ligne de synthèse par voie."""
    ref = curated[reference]
    ref_ids = ref.drop_duplicates("record_key").set_index("record_key")["activity_id"]
    ref_kept = ref.loc[ref["subset"] != "excluded", "record_key"].value_counts()
    ref_all = ref["record_key"].value_counts()
    rows = []
    for label, table in curated.items():
        kept = table.loc[table["subset"] != "excluded", "record_key"].value_counts()
        all_keys = table["record_key"].value_counts()
        # identifiants d’activité des enregistrements retenus : l’export du site web n’en a pas, on les emprunte à l’API via la clé
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
            mo.md("### Les deux voies donnent-elles les mêmes données ?\nChaque voie passe par **les mêmes** fonctions de "
                  "normalisation et de curation. Nous comparons les enregistrements par leur clé composite "
                  "(molécule · essai · document · relation · valeur · unités), car l’export du site web n’a pas "
                  "d’identifiant d’activité ; les identifiants d’activité retenus sont ensuite retrouvés grâce à cette clé."),
            mo.ui.table(route_comparison.set_index("route").T.astype(str).reset_index(names="check"), selection=None,
                        pagination=False),
            mo.callout(
                "L’export manuel et l’instantané de l’API (même version, même requête) concordent enregistrement par enregistrement — y compris après curation."
                if _same else "Les deux voies en cache diffèrent — vérifiez la version et le périmètre de l’export.",
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
        label="**Enregistrements ChEMBL utilisés dans la suite du notebook** (tout choix passe par le même pipeline) :",
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
    ## 3 · Charger OpenADMET, puis inspecter et nettoyer les deux sources (30–45′)

    Nous lisons le fichier d’entraînement OpenADMET à partir d’une **révision figée** sur Hugging Face (jeu de données
    `openadmet/cyp-challenge-train-test`, configuration `default`, partition `train`) et vérifions sa taille *avant* de
    toucher à la moindre structure.
    """)
    return


@app.cell
def _(provenance):
    try:
        _response = requests.get(OPENADMET_TRAIN_URL, timeout=60)
        _response.raise_for_status()
        _oa_bytes = _response.content
        openadmet_origin = f"fichier Hugging Face figé, révision `{HF_REVISION[:12]}…`"
    except Exception as _exc:
        _oa_bytes, _where = read_cached("openadmet_train")
        openadmet_origin = f"⚠️ instantané en cache ({_where}) — le téléchargement en direct a échoué : {type(_exc).__name__}"
    openadmet_sha256 = sha256_hex(_oa_bytes)
    openadmet_raw = pd.read_csv(io.BytesIO(_oa_bytes))

    _checks = pd.DataFrame(
        [
            {"check": "lignes (les quatre isoformes)", "found": len(openadmet_raw), "expected": OPENADMET_EXPECTED["rows"]},
            {"check": "lignes avec une valeur CYP3A4", "found": int(openadmet_raw[CYP3A4].notna().sum()),
             "expected": OPENADMET_EXPECTED["cyp3a4_values"]},
            {"check": "sha256 = instantané enregistré", "found": openadmet_sha256[:16] + "…",
             "expected": provenance["files"]["openadmet_train"]["sha256"][:16] + "…"},
        ]
    )
    _checks["ok"] = ["✅" if str(f) == str(e) else "❌" for f, e in zip(_checks["found"], _checks["expected"])]
    mo.stop(
        (_checks["ok"] == "❌").iloc[:2].any(),
        mo.callout("Le fichier OpenADMET n’a pas la taille attendue : arrêtez-vous et vérifiez la révision.", kind="danger"),
    )
    mo.vstack([mo.md(f"Chargé depuis : **{openadmet_origin}**"), mo.ui.table(_checks, selection=None, pagination=False)])
    return (openadmet_raw,)


@app.cell(hide_code=True)
def _(chembl_common, chembl_route_label, openadmet_raw):
    def _overview(df, name, id_col, smiles_col):
        _types = df.dtypes.astype(str).value_counts().to_dict()
        return mo.vstack(
            [
                mo.hstack(
                    [
                        mo.stat(f"{df.shape[0]:,} × {df.shape[1]}", label="dimensions", bordered=True),
                        mo.stat(f"{df[id_col].nunique():,}", label=f"valeurs uniques de {id_col}", bordered=True),
                        mo.stat(f"{df[smiles_col].nunique():,}", label="chaînes SMILES uniques", bordered=True),
                    ],
                    justify="start", wrap=True,
                ),
                mo.md("types de colonnes : " + ", ".join(f"{v} × `{k}`" for k, v in _types.items())),
                mo.ui.table(df.head(50), selection=None, page_size=5),
            ]
        )

    mo.ui.tabs(
        {
            "OpenADMET (brut)": _overview(openadmet_raw, "OpenADMET", "Molecule_Name", "SMILES"),
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
                x=alt.X("missing:Q", axis=alt.Axis(format="%"), title="fraction manquante", scale=alt.Scale(domain=[0, 1])),
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
            mo.md("### Valeurs manquantes\nOpenADMET est une **matrice creuse** : chaque molécule n’a été mesurée que sur "
                  "certaines isoformes. Une cellule CYP3A4 vide signifie *non mesuré / non rapporté*, **pas inactif**."),
            _chart,
            mo.md("Quelles combinaisons d’isoformes ont été mesurées (fichier d’entraînement OpenADMET) ?"),
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
    oa_meas["weak_flag"] = oa_meas["pIC50"] < WEAK_PIC50       # signalé, jamais supprimé ni écrêté
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
            y=alt.Y("ci_width:Q", title="largeur de l’intervalle à 95 % (unités log)"),
            color=alt.Color("flag:N", scale=alt.Scale(domain=["pIC50 ≥ 4", "pIC50 < 4 (cautious)"],
                                                      range=["#3b6fb6", "#d4a017"]), title=None),
            tooltip=["name", alt.Tooltip("pIC50:Q", format=".2f"), alt.Tooltip("pIC50_ci_low:Q", format=".2f"),
                     alt.Tooltip("pIC50_ci_high:Q", format=".2f")],
        )
        .properties(width=520, height=300, title="L’incertitude augmente quand la puissance diminue")
    )
    oa_uncertainty_chart = mo.ui.altair_chart(fig_uncertainty, chart_selection="interval", legend_selection=True)
    mo.vstack(
        [
            mo.md(
                f"### Valeurs CYP3A4 d’OpenADMET et leur incertitude\n"
                f"**{len(oa_meas):,}** molécules ont un pIC50 pour CYP3A4 (médiane {oa_meas['pIC50'].median():.2f}). "
                f"**{int(oa_meas['weak_flag'].sum()):,} ({oa_meas['weak_flag'].mean():.0%})** sont inférieures à 4 : ces puissances "
                "se situent en dessous de la plus faible concentration testée ; elles sont donc extrapolées et leurs intervalles sont larges. "
                "Nous les **signalons** — nous ne les écrêtons pas, ne les censurons pas, ne les supprimons pas. *Faites glisser la souris sur le graphique pour inspecter les points.*"
            ),
            oa_uncertainty_chart,
        ]
    )
    return fig_uncertainty, oa_uncertainty_chart


@app.cell(hide_code=True)
def _(oa_uncertainty_chart):
    _sel = oa_uncertainty_chart.value
    mo.ui.table(_sel.drop(columns=["flag"], errors="ignore").reset_index(drop=True), selection=None, page_size=5) if len(_sel) else mo.md(
        "_Sélectionnez des points sur le graphique pour les lister ici._"
    )
    return


@app.cell(hide_code=True)
def _(chembl_meas, openadmet_raw):
    _no3a4 = openadmet_raw[openadmet_raw[CYP3A4].isna() & openadmet_raw.filter(regex="^CYP.*_direct_inhibition$").notna().any(axis=1)].iloc[0]
    _cens = chembl_meas[(chembl_meas["subset"] == "censored") & (chembl_meas["standard_relation"] == ">")].iloc[0]
    exercise1_q1 = mo.ui.radio(
        {
            "Elle est inactive sur CYP3A4 (pIC50 sous le seuil de détection).": "a",
            "Elle n’a pas été mesurée / rapportée pour CYP3A4 : nous n’en savons rien.": "b",
            "Son pIC50 pour CYP3A4 vaut 0.": "c",
        },
        label=f"**Q1.** `{_no3a4['Molecule_Name']}` a des valeurs pour d’autres isoformes mais une **cellule CYP3A4 vide**. Que pouvez-vous conclure ?",
    )
    _bound = 9 - np.log10(_cens["standard_value"])
    exercise1_q2 = mo.ui.radio(
        {
            f"pIC50 = {_bound:.2f}": "a",
            f"pIC50 < {_bound:.2f}": "b",
            f"pIC50 > {_bound:.2f}": "c",
            "L’enregistrement est inutile et doit être supprimé.": "d",
        },
        label=(f"**Q2.** L’enregistrement ChEMBL `{_cens['molecule_chembl_id']}` dans `{_cens['assay_chembl_id']}` indique "
               f"**IC50 {_cens['standard_relation']} {_cens['standard_value']:,.0f} nM**. Quelle affirmation est correcte ?"),
    )
    mo.vstack(
        [
            mo.md("### ✏️ Exercice 1 — une valeur manquante face à une mesure censurée"),
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
            mo.callout("Q1 ✔ Exact : manquant signifie *non mesuré*. Le traiter comme inactif reviendrait à inventer des données.", kind="success")
            if exercise1_q1.value == "b"
            else mo.callout("Q1 ✘ Pas tout à fait — une cellule vide ne dit rien sur la puissance.", kind="danger")
        )
    if exercise1_q2.value:
        _fb.append(
            mo.callout("Q2 ✔ Exact : un IC50 supérieur à une valeur signifie un pIC50 *inférieur* à la borne correspondante. "
                       "C’est une mesure réelle et informative — simplement pas un nombre exact. Nous la conservons dans un tableau censuré séparé.",
                       kind="success")
            if exercise1_q2.value == "b"
            else mo.callout("Q2 ✘ Rappelez-vous que pIC50 = 9 − log₁₀(IC50 / nM) : un IC50 *plus grand* donne un pIC50 *plus petit*, "
                            "et une inégalité reste une inégalité.", kind="danger")
        )
    mo.vstack(
        _fb
        + [
            mo.md("**Ensuite, en code :** comptez les molécules OpenADMET sans valeur CYP3A4 mais avec au moins une valeur pour une "
                  "autre isoforme, et comptez les enregistrements censurés de ChEMBL par relation (`chembl_meas`, colonne `subset`)."),
        ]
    )
    return


@app.cell
def _():
    # ✏️ Exercice 1 — votre code ici (préfixez vos variables par _ pour qu’elles restent locales à cette cellule)
    return


@app.cell(hide_code=True)
def _():
    mo.accordion(
        {
            "🔑 Solution — exercice 1": mo.md(r"""
    ```python
    _iso = openadmet_raw.filter(regex="^CYP.*_pIC50_direct_inhibition$")
    _other = _iso.drop(columns=[CYP3A4]).notna().any(axis=1)
    print("no CYP3A4 value but another isoform:", int((openadmet_raw[CYP3A4].isna() & _other).sum()))

    print(chembl_meas.loc[chembl_meas["subset"] == "censored", "standard_relation"].value_counts())
    ```
    Environ 2,570 molécules OpenADMET n’ont pas de valeur CYP3A4 : elles n’ont simplement pas été mesurées sur CYP3A4
    (la plupart proviennent des criblages des autres isoformes). Dans ChEMBL, les enregistrements censurés sont surtout
    de type `>` (par ex. « IC50 > 30 µM » : le composé était plus faible que la concentration maximale). Les deux sont
    conservés, mais aucun n’est transformé en nombre.
    """)
        }
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Curation de ChEMBL : une politique documentée

    Chaque enregistrement est vérifié selon les règles ci-dessous, dans cet ordre ; la **première** règle qu’il ne respecte
    pas devient son `exclusion_reason`. Rien n’est supprimé du tableau de mesures — les enregistrements exclus reçoivent
    simplement une raison.

    | règle | pourquoi |
    |---|---|
    | cible = `CHEMBL340`, *SINGLE PROTEIN*, *Homo sapiens* | l’enzyme humaine vérifiée |
    | **score de confiance 9** de l’essai | ChEMBL est sûr que l’essai mesure cette protéine unique |
    | critère mesuré **IC50** | un seul critère |
    | contexte de l’essai = **inhibition directe** | ni inhibition temps-dépendante, ni induction, ni autres critères, ni cellules entières (voir la revue des essais) |
    | pas de drapeau de validité des données, pas de *doublon potentiel* | les propres avertissements de ChEMBL |
    | unités **nM**, valeur positive et finie, une structure | nécessaires au calcul du pIC50 et à la chimie |
    | relation `=` → **quantitative** ; `>`, `<`, `≥`, `≤` → **censurée** (conservée à part) | les bornes ne sont jamais traitées comme des valeurs exactes |
    | un enregistrement strictement dupliqué (même molécule, essai, document, valeur) n’est conservé qu’une fois | le même nombre saisi deux fois ne fait pas deux mesures |

    Nous n’exigeons **pas** `assay_type = B` : la plupart des essais d’inhibition des CYP sont classés **A** (ADME) dans ChEMBL.
    """)
    return


@app.function
def classify_assay(description, bao_label=None):
    """Pré-classification par mots-clés d’un essai ChEMBL sur CYP3A4 : ('include' | 'exclude', raison)."""
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
    """Préparation enzymatique citée dans la description d’un essai."""
    if re.search(r"recombinant|supersome|baculo|expressed in|cDNA|insect|E\. ?coli|Escherichia|yeast|bactosome", description, re.I):
        return "recombinant enzyme"
    if re.search(r"microsom", description, re.I) and re.search(r"liver|hepat", description, re.I) or re.search(r"\bHLM\b", description):
        return "human liver microsomes"
    if re.search(r"hepatocyte", description, re.I):
        return "hepatocytes"
    return "not stated"


@app.function
def probe_substrate(description):
    """Substrat sonde cité dans la description d’un essai ('multiple' si plusieurs, 'not stated' si aucun)."""
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
    """Classe chaque essai avec les règles à mots-clés, puis applique les décisions revues par l’enseignant."""
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
    """Applique la politique de curation aux enregistrements du schéma commun. Ajoute les colonnes `exclusion_reason`, `subset` et pIC50.

    `skip` liste les clés de règles (voir CURATION_RULES) à désactiver — uniquement pour l’explorateur « et si ? »."""
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
    df["pIC50_bound"] = p.where(censored)                      # une borne, jamais une valeur exacte
    df["pIC50_relation"] = np.select(
        [quantitative, censored], ["=", df["standard_relation"].map(FLIP_RELATION).fillna("?")], default=""
    )
    df["weak_flag"] = df["pIC50"].lt(WEAK_PIC50)
    return df


@app.function
def curation_funnel(measurements):
    """Enregistrements retirés par chaque règle (première règle non respectée), dans l’ordre de la politique."""
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
            mo.md(f"#### Résultat de la politique sur *{chembl_route_label}*"),
            mo.hstack(
                [
                    mo.stat(f"{len(chembl_meas):,}", label="enregistrements", bordered=True),
                    mo.stat(f"{len(_q):,}", label="quantitatifs", caption=f"{_q['molecule_chembl_id'].nunique():,} molécules · {_q['assay_chembl_id'].nunique():,} essais", bordered=True),
                    mo.stat(f"{len(_c):,}", label="censurés (conservés à part)", bordered=True),
                    mo.stat(f"{int(_q['weak_flag'].sum()):,}", label="quantitatifs avec pIC50 < 4", bordered=True),
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
        ["all assays", "included", "excluded", "instructor-reviewed"], value="all assays", label="Afficher"
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
            mo.md("#### La revue des essais (essais de score de confiance 9 avec des enregistrements IC50)\nDes règles à mots-clés "
                  "pré-classent chaque description d’essai ; l’enseignant a lu les plus gros essais avant le cours (`decision_basis`). "
                  "Utilisez la barre de recherche et les filtres de colonnes pour trouver, par ex., *time dependent* ou "
                  "*hepatocyte* (les descriptions sont en anglais)."),
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
        label="**Et si… ?** Règles activées (retirez-en une pour voir son effet) :",
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
            mo.md(f"Règles **désactivées** : {', '.join(CURATION_RULES[k] for k in _skip) if _skip else '_aucune_'}. "
                  "La politique de l’atelier elle-même ne change jamais — le reste du notebook continue de l’utiliser."),
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
            mo.md("#### Vérifications de la conversion d’unités"),
            mo.ui.table(_examples, selection=None, pagination=False),
            mo.callout(
                mo.md(f"Les {len(_q):,} valeurs quantitatives sont toutes en **nM**, et notre pIC50 concorde avec le "
                      f"`pchembl_value` propre à ChEMBL pour les {len(_with_pchembl):,} enregistrements qui en possèdent un "
                      f"(écart maximal {_max_diff:.3f}, soit un arrondi à deux décimales)."),
                kind="success" if _ok else "danger",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Une seule politique de standardisation des structures pour les deux sources

    Mêmes étapes RDKit pour ChEMBL et OpenADMET (comme dans la session 04 des tutoriels) :
    `Cleanup` (normalisation des groupes fonctionnels, réionisation) → conservation du **plus grand fragment organique**
    (élimine les contre-ions et les solvants) → **neutralisation** → SMILES **isomérique** canonique. La stéréochimie
    est **conservée** ; les tautomères **ne sont pas** canonisés (lent, et un choix de modélisation dont nous n’avons
    pas besoin aujourd’hui). Les SMILES d’origine sont conservés à côté des SMILES standardisés, et chaque modification
    est enregistrée.

    L’**identité** de deux structures = SMILES isomériques standardisés identiques — *pas* une similarité d’empreinte
    égale à 1.
    """)
    return


@app.function
def standardize_smiles(smiles):
    """Standardise un SMILES selon la politique de l’atelier ; renvoie le nouveau SMILES, un statut et ce qui a changé."""
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
    """Standardise chaque SMILES distinct une seule fois ; renvoie une table de correspondance indexée par le SMILES d’origine."""
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
    mo.vstack([mo.md("#### Ce qu’a fait la standardisation"), mo.ui.table(_rep.reset_index(names="count"), selection=None, pagination=False)])
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
                                     label="Structures modifiées par la standardisation — cliquez sur une ligne pour voir avant/après")
    std_examples_table
    return (std_examples_table,)


@app.cell(hide_code=True)
def _(std_examples_table):
    _sel = std_examples_table.value
    if len(_sel):
        _r = _sel.iloc[0]
        _out = mol_grid([_r["smiles"], _r["smiles_std"]], ["originale", "standardisée"], per_row=2, size=(300, 220))
    else:
        _out = mo.md("_Cliquez sur une ligne ci-dessus pour comparer la structure originale et la structure standardisée._")
    _out
    return


@app.function
def build_structure_table(chembl, openadmet):
    """Une ligne par structure standardisée unique, avec l’appartenance aux sources et des résumés d’activité propres à chaque source.

    Ensembles de référence ChEMBL, du plus strict au plus large : `ref_quantitative` (≥ 1 enregistrement quantitatif),
    `ref_curated` (quantitatif ou censuré — le défaut de l’atelier), `ref_broad` (tout enregistrement IC50 humain CYP3A4
    de confiance 9, quel que soit le contexte de l’essai) et `ref_any` (tout enregistrement IC50 CHEMBL340, y compris
    avec une confiance d’essai plus faible)."""
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
            mo.md("### Avant / après\nLes analyses structurales ci-dessous utilisent **une ligne par structure unique** ; "
                  "les mesures restent dans leurs propres tables, par source et par essai."),
            mo.ui.table(before_after, selection=None, pagination=False),
        ]
    )
    return (before_after,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    <a id="eda"></a>
    ## 4 · Analyse exploratoire basée sur les descripteurs (45–60′)

    Neuf descripteurs classiques (RDKit) pour chaque structure unique. Une structure présente dans **les deux** sources
    compte dans les deux distributions. N’oubliez pas : les différences entre sources peuvent provenir de **la chimie
    que chaque source a sélectionnée** (séries de chimie médicinale vs bibliothèque de diversité + médicaments
    approuvés) *et* de **la manière dont l’activité a été mesurée**.
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
    """Squelette de Bemis–Murcko en SMILES canonique sans stéréochimie ; '' pour une molécule acyclique."""
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold, isomericSmiles=False) if scaffold.GetNumAtoms() else ""


@app.cell
def _(structures):
    _mols = [Chem.MolFromSmiles(s) for s in mo.status.progress_bar(structures["smiles_std"], title="Descripteurs et squelettes", remove_on_exit=True)]
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
    """Histogramme par groupe sur des classes communes, en fractions de chaque groupe (pour comparer des groupes de tailles différentes)."""
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
    descriptor_picker = mo.ui.dropdown(DESCRIPTORS, value="MW", label="Descripteur")
    descriptor_bins = mo.ui.slider(10, 80, step=5, value=35, label="classes (descripteurs continus)", show_value=True)
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
            y=alt.Y("fraction:Q", stack=None, axis=alt.Axis(format="%"), title="fraction de la source"),
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
                    "Médianes des neuf descripteurs et décalage exprimé en écarts interquartiles de ChEMBL": mo.ui.table(
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
                             label="Corrélation de Spearman des descripteurs dans :")
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
    ### Distributions d’activité — par source et par essai, jamais regroupées

    Les valeurs ChEMBL proviennent de nombreux protocoles ; celles d’OpenADMET d’un seul. Regrouper les enregistrements
    ChEMBL par **substrat sonde** ou par **source de l’enzyme** montre à quel point le « même » critère varie avec
    l’essai. Il n’y a **aucun seuil actif/inactif** ici : nous examinons les distributions entières. Les valeurs
    inférieures à 4 (zone ombrée) sont imprécises dans les deux sources.
    """)
    return


@app.cell
def _():
    pic50_group_by = mo.ui.radio({"famille de substrats sondes": "probe_family", "source de l’enzyme": "enzyme_source"}, value="famille de substrats sondes",
                                 inline=True, label="Regrouper les enregistrements ChEMBL par")
    pic50_group_by
    return (pic50_group_by,)


@app.cell(hide_code=True)
def _(chembl_std, oa_std, pic50_group_by):
    _OA = "OpenADMET · recombinant, DBOMF"
    _CENS = "ChEMBL · bornes censurées « IC50 > » (le vrai pIC50 est plus faible)"
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
                mo.md(f"Les médianes quantitatives de ChEMBL semblent **plus élevées** que celles d’OpenADMET. Cela tient "
                      f"en partie à la manière de rapporter les données : dans ChEMBL, les composés faibles sont le plus souvent "
                      f"enregistrés comme **« IC50 > x »** ({len(_c):,} enregistrements censurés, ligne grise : "
                      "le vrai pIC50 se situe *en dessous* de chaque borne), alors qu’OpenADMET a ajusté une valeur — souvent "
                      "inférieure à 4 — pour chaque composé. Ne comparez jamais la ligne grise comme s’il s’agissait de valeurs."),
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
            color=alt.Color("probe_family:N", title="substrat sonde ChEMBL"),
            tooltip=["name", "molecule_chembl_id", "assay_chembl_id", "probe_family", "enzyme_source",
                     alt.Tooltip("ChEMBL pIC50:Q", format=".2f"), alt.Tooltip("OpenADMET pIC50:Q", format=".2f")],
        )
    )
    _rho = _pairs[["ChEMBL pIC50", "OpenADMET pIC50"]].corr(method="spearman").iloc[0, 1] if len(_pairs) > 2 else float("nan")
    mo.vstack(
        [
            mo.md(
                f"#### Les mêmes molécules, mesurées deux fois\n**{_both.size}** structures sont présentes dans les deux sources ; "
                f"**{_pairs['smiles_std'].nunique()}** d’entre elles ont des enregistrements ChEMBL quantitatifs ({len(_pairs)} mesures "
                f"ChEMBL). Un point par mesure ChEMBL — rien n’est moyenné. ρ de Spearman = {_rho:.2f}. "
                "Les points éloignés de la diagonale ne sont pas des erreurs : ce sont des essais différents."
            ),
            (_line + _pts).properties(width=480, height=380),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### ✏️ Exercice 2 — expliquer une différence entre sources

    Choisissez un descripteur dont la distribution diffère nettement entre les sources (regardez la colonne
    « in ChEMBL IQRs », le décalage exprimé en écarts interquartiles de ChEMBL). Rédigez deux ou trois phrases : la
    différence est-elle plus probablement due à la **sélection chimique** (quelles molécules contient chaque source) ou
    aux **différences d’essai** ? Un descripteur de la *structure* peut-il seulement être affecté par l’essai ?
    Vérifiez ensuite une hypothèse en code, par exemple en comparant les 50 composés ChEMBL les plus puissants au reste.
    """)
    return


@app.cell
def _():
    exercise2_answer = mo.ui.text_area(placeholder="Votre explication…", rows=4, full_width=True)
    exercise2_answer
    return


@app.cell
def _():
    # ✏️ Exercice 2 — votre code ici (utilisez des noms préfixés par _)
    return


@app.cell(hide_code=True)
def _():
    mo.accordion(
        {
            "🔑 Solution — exercice 2": mo.md(r"""
    Les descripteurs de structure sont calculés à partir de la seule structure ; une différence de masse molaire (MW),
    de cLogP ou de nombre de cycles aromatiques ne peut donc venir que de **la nature des molécules de chaque
    ensemble**. L’ensemble CYP3A4 d’OpenADMET provient d’une bibliothèque de diversité et de médicaments approuvés :
    molécules plus petites, moins lipophiles, avec moins de cycles aromatiques que le composé ChEMBL typique, lequel a
    été synthétisé dans un projet de chimie médicinale (souvent un chef de file optimisé et lipophile) puis testé sur
    CYP3A4. L’essai peut tout de même jouer **indirectement** : les articles ChEMBL tendent à rapporter des IC50 sur
    CYP3A4 pour des composés intéressants (puissants sur la cible principale, ou posant un problème de CYP), si bien que
    la *sélection* de ce qui est publié dépend du résultat de l’essai.

    ```python
    _q = chembl_std[chembl_std["subset"] == "quantitative"]
    _per_structure = _q.groupby("smiles_std")["pIC50"].median()
    _f = features.set_index("smiles_std").loc[_per_structure.index, DESCRIPTORS]
    _top = _per_structure.nlargest(50).index
    print(_f.loc[_top].median().round(2).to_frame("top-50 potent").join(_f.drop(_top).median().round(2).rename("rest")))
    ```
    Les inhibiteurs de CYP3A4 les plus puissants dans ChEMBL sont plus grands et plus lipophiles que les autres — le
    grand site actif hydrophobe de CYP3A4 favorise de telles molécules, ce qui explique en partie pourquoi la
    lipophilie suit l’inhibition de CYP3A4.
    """)
        }
    )
    return


@app.cell(hide_code=True)
def _(by_source):
    mo.accordion(
        {
            "🔍 Explorer librement (constructeur de graphiques par glisser-déposer)": mo.ui.data_explorer(
                by_source[["source", "label"] + DESCRIPTORS + ["openadmet_pIC50", "chembl_median_pIC50"]]
            )
        }
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    <a id="similarity"></a>
    ## 5 · Empreintes de Morgan et similarité de Tanimoto (60–80′)

    Une **empreinte de Morgan** (rayon 2, 2048 bits — de type ECFP4) active un bit pour chaque environnement atomique
    circulaire s’étendant jusqu’à deux liaisons. La similarité de **Tanimoto** de deux empreintes binaires $A$ et $B$ est

    $$T(A,B) = \frac{|A \cap B|}{|A \cup B|} \in [0, 1].$$

    Choisissez une paire, ou saisissez vos propres SMILES. Les atomes **surlignés** se trouvent en dehors de la
    sous-structure commune maximale.
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
    pair_picker = mo.ui.dropdown(list(pair_presets), value="midazolam vs alprazolam (close analogues)", label="Paire",
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
    """Dessine deux molécules côte à côte en surlignant les atomes situés en dehors de leur sous-structure commune maximale."""
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
        _out = mo.callout("L’un des SMILES ne peut pas être interprété.", kind="danger")
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
                        mo.stat(f"{_fa.GetNumOnBits()} / {_fb.GetNumOnBits()}", label="bits actifs (A / B)", bordered=True),
                        mo.stat(str(_shared), label="bits communs", bordered=True),
                        mo.stat("oui" if _identical else "non", label="structures identiques ?", bordered=True),
                    ],
                    justify="start", wrap=True,
                ),
                pair_with_differences(pair_a.value, pair_b.value, ["A", "B"]),
                mo.callout("Tanimoto = 1 mais les structures diffèrent : cette empreinte ignore la stéréochimie. C’est pourquoi "
                           "le recouvrement exact est déterminé par les SMILES isomériques standardisés, jamais par T = 1.", kind="warn")
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
            mo.md("Quelques médicaments du CYP3A4 comparés deux à deux. Les analogues proches (un atome ou un cycle modifié) "
                  "atteignent T ≈ 0.6–0.8 avec cette empreinte ; les médicaments sans lien se situent autour de 0.05–0.2."),
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
    ### Le plus proche voisin ChEMBL de chaque composé OpenADMET

    Pour chaque structure OpenADMET, nous calculons sa similarité de Tanimoto avec **toutes** les structures de
    l’ensemble de référence ChEMBL en une seule fois (`BulkTanimotoSimilarity` de RDKit), puis nous conservons la
    meilleure correspondance **non identique**. Le recouvrement exact des structures est rapporté séparément — une
    molécule n’est pas son propre voisin.
    """)
    return


@app.function
def similarity_matrix(query_fps, reference_fps):
    """Similarité de Tanimoto de chaque empreinte requête avec chaque empreinte de référence (matrice float32)."""
    out = np.empty((len(query_fps), len(reference_fps)), dtype=np.float32)
    for i, fp in enumerate(mo.status.progress_bar(query_fps, title="Tanimoto par lots", remove_on_exit=True)):
        out[i] = DataStructs.BulkTanimotoSimilarity(fp, reference_fps)
    return out


@app.function
def nearest_neighbours(sim, query_keys, reference_keys, reference_mask):
    """Meilleure référence non identique pour chaque ligne requête de `sim`, parmi les colonnes autorisées par `reference_mask`.

    Identité = même SMILES isomérique standardisé ; une telle correspondance est renvoyée dans `exact` et exclue de la recherche."""
    s = np.where(reference_mask[None, :], sim, -1.0)
    position = {k: j for j, k in enumerate(reference_keys)}
    exact = np.zeros(len(query_keys), dtype=bool)
    for i, key in enumerate(query_keys):
        j = position.get(key)
        if j is not None and reference_mask[j]:
            exact[i] = True
            s[i, j] = -1.0                                   # pas d’auto-correspondance
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
            "enregistrements quantitatifs + censurés (défaut de l’atelier)": "ref_curated",
            "enregistrements quantitatifs uniquement": "ref_quantitative",
            "tout enregistrement IC50 humain CYP3A4 de confiance 9 (y compris TDI, autres contextes d’essai)": "ref_broad",
            "tout enregistrement IC50 CHEMBL340, même avec une confiance d’essai < 9": "ref_any",
        },
        value="enregistrements quantitatifs + censurés (défaut de l’atelier)",
        label="**Ensemble de référence ChEMBL**",
    )
    coverage_threshold = mo.ui.slider(0.2, 0.9, step=0.05, value=0.4, label="« couvert » si le T du plus proche voisin ≥",
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
            x=alt.X("bin:Q", title="Tanimoto avec la structure ChEMBL non identique la plus proche",
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
                    mo.stat(f"{nn_reference_size:,}", label="structures de référence ChEMBL", bordered=True),
                    mo.stat(f"{len(nn_table):,}", label="structures OpenADMET", bordered=True),
                    mo.stat(f"{int(nn_table['exact_in_chembl'].sum())} ({nn_table['exact_in_chembl'].mean():.1%})",
                            label="recouvrement exact", caption="même structure standardisée", bordered=True),
                    mo.stat(f"{nn_table['nn_tanimoto'].median():.2f}", label="T médian du plus proche voisin", bordered=True),
                    mo.stat(f"{(nn_table['nn_tanimoto'] >= _thr).mean():.1%}", label=f"avec un voisin T ≥ {_thr:.2f}", bordered=True),
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
        label="Cliquez sur un composé (trié du moins couvert au plus couvert ; utilisez la zone de recherche pour un nom)",
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
        _out = mo.md("_Sélectionnez une ligne ci-dessus pour voir le composé à côté de ses trois plus proches voisins ChEMBL._")
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
                mo.md(f"**{_r['name']}** — structure exacte dans ChEMBL : **{'oui' if _r['exact_in_chembl'] else 'non'}**. "
                      "Plus proches voisins non identiques (les atomes situés en dehors de la sous-structure commune "
                      "avec le premier voisin sont surlignés) :"),
                pair_with_differences(_r["smiles_std"], _nbrs.iloc[0]["smiles_std"], _legends[:2]),
                mol_grid(_nbrs["smiles_std"].tolist()[1:], _legends[2:], per_row=2, size=(260, 200)),
                mo.md("Enregistrements ChEMBL du plus proche voisin (les activités restent par essai) :"),
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
            mo.md("#### Dans quelle mesure la réponse dépend-elle de la définition de l’ensemble de référence ?\n"
                  "La politique de curation "
                  "fait partie de la réponse. La plupart des molécules OpenADMET que ChEMBL contient *bel et bien* — beaucoup "
                  "sont des médicaments approuvés — ont été testées dans des essais que notre politique exclut :"),
            mo.ui.table(pd.DataFrame(_rows), selection=None, pagination=False),
            mo.md(f"Pourquoi les **{len(_extra)}** correspondances exactes supplémentaires de la définition la plus large ne "
                  "figurent pas dans la référence par défaut (première règle non satisfaite de leurs enregistrements ChEMBL, "
                  "et l’essai) :"),
            mo.ui.table(_why, selection=None, page_size=6),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### ✏️ Exercice 3 — examiner une paire de plus proches voisins

    Dans le tableau ci-dessus, choisissez un composé avec T ≈ 0.4–0.5 et un autre avec T > 0.7.
    1. Qu’est-ce qui diffère exactement entre chaque composé et son voisin (regardez les atomes surlignés) ?
    2. Appelleriez-vous, en tant que chimiste, la première paire des « analogues » ? Que dit cela d’un seuil fixe ?
    3. Le voisin possède-t-il une valeur ChEMBL **quantitative**, une valeur **censurée**, ou seulement des
       enregistrements exclus ? Pourriez-vous utiliser son activité pour deviner la valeur OpenADMET ? (Pensez aux
       différences d’essai.)
    """)
    return


@app.cell
def _():
    # ✏️ Exercice 3 — votre code ici, par ex. nn_table[nn_table["nn_tanimoto"].between(0.4, 0.5)].head()
    return


@app.cell(hide_code=True)
def _():
    mo.accordion(
        {
            "🔑 Solution — exercice 3": mo.md(r"""
    ```python
    _mid = nn_table[nn_table["nn_tanimoto"].between(0.4, 0.5)].iloc[0]
    _high = nn_table[(nn_table["nn_tanimoto"] > 0.7) & ~nn_table["exact_in_chembl"]].iloc[0]
    for _r in (_mid, _high):
        print(_r["name"], _r["nn_chembl_id"], _r["nn_tanimoto"])
    pair_with_differences(_mid["smiles_std"], _mid["nn_smiles"], ["OpenADMET", "ChEMBL"])
    ```
    À T ≈ 0.45, la paire partage généralement un système cyclique ou un segment de liaison mais diffère par ses
    substituants ou par un cycle — un chimiste médicinal peut ou non la qualifier d’analogue. Au-dessus de 0.7, la
    modification porte typiquement sur un seul substituant.
    Tout seuil est une convention : nous présentons donc des distributions entières et testons plusieurs seuils (curseur).
    L’activité ChEMBL d’un voisin provient d’un autre essai (sonde, source de l’enzyme) et, lorsqu’elle est censurée, n’est qu’une borne :
    c’est un indice, pas un substitut de la valeur OpenADMET.
    """)
        }
    )
    return


@app.cell
def _():
    test_set_button = mo.ui.run_button(label="Facultatif : vérifier la couverture des 750 structures du jeu de test masqué")
    test_set_button
    return (test_set_button,)


@app.cell(hide_code=True)
def _(nn_table, sim_refs, test_set_button):
    mo.stop(not test_set_button.value)
    try:
        _r = requests.get(OPENADMET_TEST_URL, timeout=60)
        _r.raise_for_status()
        _bytes, _origin = _r.content, "fichier Hugging Face épinglé"
    except Exception as _exc:
        _bytes, _where = read_cached("openadmet_test")
        _origin = f"instantané en cache ({_where}) ; échec du téléchargement en direct : {type(_exc).__name__}"
    _test = pd.read_csv(io.BytesIO(_bytes))
    _std = standardize_many(_test["SMILES"], title="Standardisation des structures de test")
    _keys = _std["smiles_std"].dropna().unique().tolist()
    _mask = sim_refs["ref_curated"].to_numpy()
    _sim = similarity_matrix([FP_GEN.GetFingerprint(Chem.MolFromSmiles(s)) for s in _keys],
                             [FP_GEN.GetFingerprint(Chem.MolFromSmiles(s)) for s in sim_refs["smiles_std"]])
    _exact, _best, _t = nearest_neighbours(_sim, _keys, sim_refs["smiles_std"].tolist(), _mask)
    _cmp = pd.concat([pd.DataFrame({"set": "training (CYP3A4)", "T": nn_table["nn_tanimoto"]}),
                      pd.DataFrame({"set": "blinded test (structures only)", "T": _t})])
    mo.vstack(
        [
            mo.md(f"Jeu de test issu de {_origin} : {len(_test)} lignes, {len(_keys)} structures uniques, {int(_exact.sum())} présentes "
                  f"exactement dans ChEMBL, T médian au plus proche voisin = {np.median(_t):.2f}. Aucune étiquette n’est utilisée."),
            alt.Chart(_cmp).transform_density("T", groupby=["set"], as_=["T", "density"], extent=[0, 1])
            .mark_line().encode(x="T:Q", y="density:Q", color="set:N").properties(width=560, height=220),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    <a id="scaffolds"></a>
    ## 6 · Squelettes de Bemis–Murcko (80–95′)

    Le **squelette** (« scaffold ») de **Bemis–Murcko** conserve les systèmes cycliques et les segments de liaison qui les
    relient, et supprime toutes les chaînes latérales (stéréochimie retirée). Les molécules **acycliques** n’ont pas de
    squelette — nous les comptons séparément au lieu de les regrouper dans un faux squelette « vide ».
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
                            mo.stat(f"{len(_counts):,}", label="squelettes", caption=f"{len(_counts) / max((_f['scaffold'] != '').sum(), 1):.2f} par composé cyclique", bordered=True),
                            mo.stat(f"{(_counts == 1).mean():.0%}", label="singletons", caption="squelettes vus une seule fois", bordered=True),
                            mo.stat(f"{int((_f['scaffold'] == '').sum())}", label="composés acycliques", bordered=True),
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
                    mo.stat(f"{int((scaffold_table['status'] == 'shared').sum()):,}", label="squelettes partagés", bordered=True),
                    mo.stat(f"{_novel.mean():.1%}", label="composés cycliques OpenADMET dont le squelette est absent de ChEMBL",
                            caption=f"{int(_novel.sum()):,} sur {len(_oa):,}", bordered=True),
                    mo.stat(f"{int((scaffold_table['status'] == 'OpenADMET only').sum()):,}", label="squelettes présents uniquement dans OpenADMET", bordered=True),
                ],
                justify="start",
                wrap=True,
            ),
        ]
    )
    return


@app.cell
def _():
    top_n_scaffolds = mo.ui.slider(4, 24, step=4, value=8, label="squelettes les plus fréquents à dessiner", show_value=True)
    top_n_scaffolds
    return (top_n_scaffolds,)


@app.cell(hide_code=True)
def _(scaffold_table, top_n_scaffolds):
    def _grid(col, n):
        _t = scaffold_table[scaffold_table[col] > 0].sort_values(col, ascending=False).head(n)
        return mol_grid(_t["scaffold"].tolist(), [f"{c} comp. · {s}" for c, s in zip(_t[col], _t["status"])], per_row=4,
                        size=(190, 150))

    _shared = scaffold_table[scaffold_table["status"] == "shared"].assign(m=lambda t: t[["ChEMBL", "OpenADMET"]].min(axis=1))
    _shared = _shared.sort_values("m", ascending=False).head(top_n_scaffolds.value)
    mo.ui.tabs(
        {
            "ChEMBL": _grid("ChEMBL", top_n_scaffolds.value),
            "OpenADMET": _grid("OpenADMET", top_n_scaffolds.value),
            "Partagés": mol_grid(_shared["scaffold"].tolist(), [f"ChEMBL {a} · OA {b}" for a, b in zip(_shared["ChEMBL"], _shared["OpenADMET"])],
                               per_row=4, size=(190, 150)),
        }
    )
    return


@app.cell
def _():
    scaffold_status_filter = mo.ui.dropdown(["all", "shared", "OpenADMET only", "ChEMBL only"], value="OpenADMET only",
                                            label="Tableau des squelettes : afficher")
    scaffold_status_filter
    return (scaffold_status_filter,)


@app.cell
def _(scaffold_status_filter, scaffold_table):
    _t = scaffold_table if scaffold_status_filter.value == "all" else scaffold_table[scaffold_table["status"] == scaffold_status_filter.value]
    scaffold_browser = mo.ui.table(_t.reset_index(drop=True), selection="multi", page_size=6,
                                   label="Sélectionnez un ou plusieurs squelettes : leurs membres sont dessinés ci-dessous, et ils peuvent colorer les cartes de la partie 7")
    scaffold_browser
    return (scaffold_browser,)


@app.cell(hide_code=True)
def _(features, scaffold_browser):
    _sel = scaffold_browser.value
    if not len(_sel):
        _out = mo.md("_Sélectionnez des squelettes dans le tableau pour voir les composés qui les contiennent._")
    else:
        _m = features[features["scaffold"].isin(_sel["scaffold"])].sort_values("membership").head(16)
        _out = mol_grid(_m["smiles_std"].tolist(), [f"{l} · {m}" for l, m in zip(_m["label"], _m["membership"])], per_row=4,
                        size=(200, 160))
    _out
    return


@app.function
def scaffold_rarefaction(scaffolds, sizes, repeats=20, seed=SEED):
    """Squelettes uniques dans des sous-échantillons aléatoires de chaque taille : diversité comparée à tailles d’échantillon *égales*."""
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
        x=alt.X("n:Q", title="composés échantillonnés (cycliques uniquement)", scale=alt.Scale(type="log")),
        y=alt.Y("scaffolds per compound:Q", scale=alt.Scale(zero=False)),
        color=alt.Color("source:N", scale=alt.Scale(domain=SOURCES[:2], range=SOURCE_COLORS[:2])),
    )
    _line = alt.Chart(_rare).mark_line(point=True).encode(x="n:Q", y="mean(scaffolds per compound):Q", color="source:N")
    fig_rarefaction = (_band + _line).properties(width=520, height=260)
    _at = _rare[_rare["n"] == 1000].groupby("source")[["scaffolds per compound", "singleton fraction"]].agg(["mean", "std"]).round(3)
    mo.vstack(
        [
            mo.md("#### Diversité à tailles d’échantillon égales\nLe nombre de squelettes par composé *diminue* toujours quand un ensemble grandit ; "
                  "comparer un ensemble de 2 300 composés à un ensemble de 7 000 composés serait donc injuste. Ici, les deux sont "
                  "sous-échantillonnés aux mêmes tailles (20 tirages aléatoires chacun, graine 42) :"),
            fig_rarefaction,
            mo.md("Pour n = 1 000 :"),
            mo.ui.table(_at.reset_index(), selection=None, pagination=False),
        ]
    )
    return (fig_rarefaction,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### ✏️ Exercice 4 — trouver un squelette absent de la référence ChEMBL

    Utilisez le tableau des squelettes (filtre *OpenADMET only*) pour choisir le squelette OpenADMET **le plus peuplé**
    qui n’apparaît jamais dans ChEMBL. Dessinez-le avec ses membres. Vérifiez ensuite avec le tableau des plus proches
    voisins : ses membres sont-ils aussi éloignés de ChEMBL dans l’espace des empreintes, ou ont-ils des voisins ChEMBL
    proches construits sur un *autre* squelette ?
    """)
    return


@app.cell
def _():
    # ✏️ Exercice 4 — votre code ici
    return


@app.cell(hide_code=True)
def _():
    mo.accordion(
        {
            "🔑 Solution — exercice 4": mo.md(r"""
    ```python
    _novel = scaffold_table[scaffold_table["status"] == "OpenADMET only"].sort_values("OpenADMET", ascending=False)
    _scaf = _novel.iloc[0]["scaffold"]
    _members = features[features["scaffold"] == _scaf]
    print(_scaf, len(_members), "compounds")
    print(nn_table[nn_table["smiles_std"].isin(_members["smiles_std"])][["name", "nn_tanimoto", "nn_chembl_id"]])
    mol_grid([_scaf] + _members["smiles_std"].tolist()[:7], ["scaffold"] + _members["label"].tolist()[:7])
    ```
    Un squelette peut être « nouveau » alors que ses composés ont pourtant des voisins ChEMBL assez similaires (mêmes
    chaînes latérales ou cycle remplacé par un isostère proche) : la nouveauté de squelette et la nouveauté en empreinte
    sont liées mais pas identiques. Les petits squelettes (un seul cycle benzénique) ne sont presque jamais nouveaux ;
    les squelettes nouveaux sont généralement des systèmes cycliques fusionnés ou reliés.
    """)
        }
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    <a id="maps"></a>
    ## 7 · Espace chimique en 2D : ACP (PCA) et UMAP (95–110′)

    Les cartes sont tracées sur un **échantillon aléatoire reproductible** : au plus 1 000 structures uniques par source
    (graine 42), puis l’union est dédupliquée ; une structure présente dans les deux sources conserve son étiquette
    **both**. L’échantillon n’est *pas* enrichi en composés actifs ou inhabituels.

    - **ACP (PCA)** des neuf descripteurs standardisés : une carte linéaire dont on peut interpréter les axes grâce aux
      contributions (loadings).
    - **UMAP** des empreintes de Morgan binaires avec la distance de Jaccard (= 1 − Tanimoto) : une carte non linéaire qui
      préserve les voisinages. Nous ajustons **une seule carte commune** ; des cartes ajustées séparément auraient des axes
      sans rapport entre eux.
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
        f"Échantillon des cartes : **{len(map_sample):,}** structures uniques — "
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
        x=alt.X("component:N", title=None), y=alt.Y("explained:Q", axis=alt.Axis(format="%"), title="variance expliquée"))
    _cum = alt.Chart(_ev).mark_line(point=True, color="black").encode(x="component:N", y="cumulative:Q")
    _heat = alt.Chart(_load_long).mark_rect().encode(
        x=alt.X("component:N", title=None), y=alt.Y("descriptor:N", sort=DESCRIPTORS, title=None),
        color=alt.Color("loading:Q", scale=alt.Scale(scheme="redblue", domain=[-0.7, 0.7], reverse=True)),
        tooltip=["descriptor", "component", alt.Tooltip("loading:Q", format=".2f")])
    _txt = alt.Chart(_load_long).mark_text(fontSize=10).encode(x="component:N", y=alt.Y("descriptor:N", sort=DESCRIPTORS),
                                                               text=alt.Text("loading:Q", format=".2f"))
    mo.hstack(wrap=True, items=[(_bars + _cum).properties(width=240, height=220, title="Variance expliquée"),
               (_heat + _txt).properties(width=200, height=220, title="Principales contributions (loadings)")], justify="start", gap=2)
    return


@app.cell
def _():
    _options = ["source", "scaffold (your selection in part 6, else the 8 most common)", "OpenADMET pIC50",
                "ChEMBL median pIC50 (quantitative records)", "OpenADMET pIC50 < 4 (cautious)"]
    map_color = mo.ui.dropdown(_options, value="source", label="Colorer les cartes selon")
    map_color
    return (map_color,)


@app.function
def map_frame(sample, coords, color_mode, selected_scaffolds, top_scaffolds):
    """Données d’une carte 2D : coordonnées plus une colonne `colour` (catégorielle ou numérique) choisie par `color_mode`."""
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
    fig_pca = map_chart(_d, "PC1", "PC2", "ACP de neuf descripteurs standardisés")
    pca_chart = mo.ui.altair_chart(fig_pca, chart_selection="interval", legend_selection=False)
    pca_chart
    return fig_pca, pca_chart


@app.cell(hide_code=True)
def _(pca_chart):
    _sel = pca_chart.value
    mo.vstack([mo.md(f"**{len(_sel)} sélectionnés** — 12 premiers dessinés :"),
               mol_grid(_sel["smiles_std"].tolist()[:12], [f"{l} · {m}" for l, m in zip(_sel["label"], _sel["membership"])][:12])]
              ) if len(_sel) else mo.md("_Tracez un rectangle sur la carte ACP pour dessiner les molécules qu’il contient._")
    return


@app.cell
def _():
    umap_params = (
        mo.md("**Paramètres UMAP** (commencez avec 15 et 0.1)  \n{n_neighbors}  \n{min_dist}")
        .batch(
            n_neighbors=mo.ui.slider(5, 50, step=5, value=15, label="n_neighbors", show_value=True),
            min_dist=mo.ui.slider(0.0, 0.8, step=0.05, value=0.1, label="min_dist", show_value=True),
        )
        .form(submit_button_label="Réajuster UMAP", bordered=True)
    )
    umap_params
    return (umap_params,)


@app.cell
def _(map_sample, umap_params):
    _p = umap_params.value or {"n_neighbors": 15, "min_dist": 0.1}
    map_fps = np.array([FP_GEN.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in map_sample["smiles_std"]], dtype=bool)
    with mo.status.spinner(title=f"Ajustement d’UMAP (n_neighbors={_p['n_neighbors']}, min_dist={_p['min_dist']})…"), UMAP_LOCK:
        umap_model = umap.UMAP(n_neighbors=int(_p["n_neighbors"]), min_dist=float(_p["min_dist"]), metric="jaccard",
                               random_state=SEED).fit(map_fps)
    map_sample_umap = map_sample.assign(UMAP1=umap_model.embedding_[:, 0], UMAP2=umap_model.embedding_[:, 1])
    return map_sample_umap, umap_model


@app.cell
def _(features, map_color, map_sample_umap, scaffold_browser):
    _top = features.loc[features["scaffold"] != "", "scaffold"].value_counts().head(8).index
    _sel = scaffold_browser.value["scaffold"].tolist() if len(scaffold_browser.value) else []
    _d = map_frame(map_sample_umap, map_sample_umap[["UMAP1", "UMAP2"]].to_numpy(), map_color.value, _sel, _top)
    fig_umap = map_chart(_d, "UMAP 1", "UMAP 2", "UMAP des empreintes de Morgan (distance de Jaccard)")
    umap_chart = mo.ui.altair_chart(fig_umap, chart_selection="interval", legend_selection=False)
    umap_chart
    return fig_umap, umap_chart


@app.cell(hide_code=True)
def _(umap_chart):
    _sel = umap_chart.value
    mo.vstack([mo.md(f"**{len(_sel)} sélectionnés** — 12 premiers dessinés :"),
               mol_grid(_sel["smiles_std"].tolist()[:12], [f"{l} · {m}" for l, m in zip(_sel["label"], _sel["membership"])][:12])]
              ) if len(_sel) else mo.md("_Tracez un rectangle sur la carte UMAP pour dessiner les molécules qu’il contient. Essayez "
                                        "un îlot propre à OpenADMET, puis une région mixte._")
    return


@app.cell(hide_code=True)
def _():
    mo.callout(
        mo.md("**Lecture des cartes.** Les axes de l’ACP ont un sens (voir les contributions) ; ceux d’UMAP n’en ont pas, et les "
              "distances *entre* îlots UMAP ne sont pas interprétables. Colorez tour à tour par *ChEMBL median pIC50* et par "
              "*OpenADMET pIC50* : chaque coloration n’utilise que les mesures propres à une source, sur la même carte commune."),
        kind="info",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    <a id="outliers"></a>
    ## 8 · Valeurs atypiques (outliers) : propriétés inhabituelles et nouveauté structurelle (110–120′)

    Deux questions différentes :

    1. **Propriétés physicochimiques inhabituelles** — une *Isolation Forest* sur les neuf descripteurs standardisés de
       toutes les structures uniques (les deux sources). L’Isolation Forest isole les points par des coupures aléatoires ;
       les points isolés après peu de coupures obtiennent un score d’anomalie élevé.
    2. **Nouveauté structurelle par rapport à ChEMBL** — les composés OpenADMET dont le plus proche voisin ChEMBL est
       éloigné (Tanimoto maximal faible, partie 5).

    Un composé peut être l’un, l’autre, les deux ou aucun. Les marquages (flags) sont des **raisons de regarder**, jamais
    de supprimer, et ils sont calculés dans l’espace complet des descripteurs / empreintes — jamais à partir des images 2D.
    """)
    return


@app.cell
def _(features):
    _scaler = StandardScaler().fit(features[DESCRIPTORS])
    _z = _scaler.transform(features[DESCRIPTORS])
    _forest = IsolationForest(n_estimators=500, random_state=SEED).fit(_z)
    outlier_scores = features[["smiles_std", "label", "membership"] + DESCRIPTORS].copy()
    outlier_scores["anomaly_score"] = -_forest.score_samples(_z)          # plus élevé = plus inhabituel
    _zdf = pd.DataFrame(_z, columns=DESCRIPTORS)
    outlier_scores["most extreme descriptors"] = [
        ", ".join(f"{c} {'+' if v > 0 else '−'}{abs(v):.1f}σ" for c, v in row.abs().nlargest(2).index.map(lambda c: (c, row[c])))
        for _, row in _zdf.iterrows()
    ]
    outlier_scores["anomaly_rank_percent"] = outlier_scores["anomaly_score"].rank(pct=True, ascending=False) * 100
    return (outlier_scores,)


@app.cell
def _():
    outlier_scope = mo.ui.radio(["OpenADMET", "ChEMBL", "all"], value="OpenADMET", inline=True, label="Classer les composés de")
    outlier_top_k = mo.ui.slider(4, 40, step=4, value=12, label="nombre à inspecter", show_value=True)
    unusual_percent = mo.ui.slider(0.5, 10, step=0.5, value=2.0, label="« propriétés inhabituelles » = % supérieur des scores d’anomalie",
                                   show_value=True)
    novelty_threshold = mo.ui.slider(0.15, 0.5, step=0.05, value=0.3, label="« structurellement nouveau » = T du plus proche voisin ChEMBL <",
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
            mo.md(f"#### Scores d’Isolation Forest les plus élevés ({outlier_scope.value})\n`most extreme descriptors` donne les "
                  "deux descripteurs les plus éloignés de la moyenne (en écarts-types) — un *pourquoi* rapide, pas la "
                  "logique interne de la forêt."),
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
            mo.md(f"#### Composés OpenADMET les moins couverts\n**{_n_novel}** structures OpenADMET "
                  f"({_n_novel / len(nn_table):.1%}) n’ont aucun voisin ChEMBL avec T ≥ {novelty_threshold.value:.2f}. "
                  f"Les {outlier_top_k.value} plus isolées :"),
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
            mo.md(f"#### Deux types de valeurs atypiques (composés OpenADMET)\nLignes : parmi les {unusual_percent.value} % les plus élevés des scores "
                  f"d’anomalie. Colonnes : T du plus proche voisin ChEMBL < {novelty_threshold.value:.2f}."),
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

    fig_outlier_maps = alt.hconcat(_panel(_pca_bg, _pca_fg.assign(placed="PCA coordinates"), "ACP"),
                                   _panel(_umap_bg, _umap_fg, "UMAP")).resolve_scale(shape="independent")
    mo.vstack(
        [
            mo.md("#### Où se trouvent les composés signalés sur les cartes ?\nLes composés signalés hors de l’échantillon des cartes sont "
                  "*projetés* sur les cartes ajustées (ACP : même normalisation et mêmes axes ; UMAP : `transform`), et non ajoutés à l’ajustement."),
            fig_outlier_maps,
        ]
    )
    return (fig_outlier_maps,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### ✏️ Exercice 5 — pourquoi une valeur atypique signalée peut-elle rester de la chimie valide ?

    Choisissez un composé dans chacune des listes ci-dessus (propriétés inhabituelles ; structurellement nouveau). Pour
    chacun, décidez si le marquage révèle (a) un **problème de données** (structure erronée, sel non retiré, mélange) ou
    (b) une **chimie réelle mais rare** (macrocycle, produit naturel, peptide, fragment très petit, médicament approuvé
    d’une classe inhabituelle). Que vérifieriez-vous avant de faire confiance à sa valeur CYP3A4 — ou de l’écarter ?
    """)
    return


@app.cell
def _():
    exercise5_answer = mo.ui.text_area(placeholder="Composé, marquage, votre diagnostic…", rows=4, full_width=True)
    exercise5_answer
    return


@app.cell(hide_code=True)
def _():
    mo.accordion(
        {
            "🔑 Solution — exercice 5": mo.md(r"""
    Les scores d’anomalie les plus élevés vont typiquement à des molécules **très grandes** (macrolides, peptides,
    produits naturels complexes ou médicaments comme les structures de type cyclosporine : beaucoup d’atomes lourds,
    TPSA élevée, nombreuses liaisons rotatives) et à des molécules **très petites et très polaires**. Beaucoup de
    médicaments approuvés de la bibliothèque FDA d’OpenADMET sont exactement dans ce cas — des molécules réelles,
    mesurées dans le même essai que les autres. La bonne réaction est de *vérifier*, pas de supprimer :

    - la structure est-elle complète et correctement standardisée (pas de contre-ion restant, pas de fragment d’un mélange) ?
    - le pIC50 mesuré est-il inférieur à 4 (auquel cas il est très incertain), et son intervalle est-il large ?
    - pour un composé structurellement nouveau : est-il nouveau parce que c’est un nouveau chémotype (intéressant pour le
      défi) ou à cause d’une représentation inhabituelle (tautomère rare, complexe métallique) ?

    Un composé signalé par les deux critères est le meilleur candidat pour un examen approfondi ; c’est aussi le type de
    composé pour lequel tout modèle entraîné sur ChEMBL serait en extrapolation.
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
    _verdict = "faible" if (_t.median() < 0.4 and _novel_scaf > 0.5) else "partiel"
    _low = nn_table[_t < 0.3]
    _props = features.set_index("smiles_std").loc[nn_table["smiles_std"], ["HeavyAtoms", "MW"]].reset_index(drop=True)
    _is_low = (_t < 0.3).to_numpy()
    _ha_low, _ha_rest = np.median(_props["HeavyAtoms"][_is_low]), np.median(_props["HeavyAtoms"][~_is_low])
    _size_note = (
        f"Leur taille médiane est de {_ha_low:.0f} atomes lourds, contre {_ha_rest:.0f} pour les autres composés ; la faible "
        "couverture n’est donc pas qu’un effet de taille : ce sont de nouveaux chémotypes pour les données CYP3A4 de ChEMBL."
        if abs(_ha_low - _ha_rest) < 3
        else f"Ils sont {'plus petits' if _ha_low < _ha_rest else 'plus grands'} que les autres (médiane {_ha_low:.0f} contre "
        f"{_ha_rest:.0f} atomes lourds). Les petites molécules activent peu de bits d’empreinte, ce qui plafonne le Tanimoto qu’elles peuvent atteindre."
    )
    _both_flags = outlier_crosstab[(outlier_crosstab["unusual properties"] == "yes") & (outlier_crosstab["structurally novel"] == "yes")]
    _prop_examples = ", ".join(f"{l} ({e})" for l, e in zip(property_candidates["label"].head(3), property_candidates["most extreme descriptors"].head(3)))
    mo.md(
        f"""
    <a id="answers"></a>
    ## 9 · Réponses fondées sur les données

    *Chaque nombre ci-dessous est calculé en direct à partir des choix faits plus haut. Sauf si vous avez modifié la
    référence ChEMBL dans la partie 5, il utilise la valeur par défaut de l’atelier.*

    **Quelle part de l’espace chimique se recouvre ?**
    Sur {_n_oa:,} structures OpenADMET CYP3A4 uniques, seules **{_both:,} ({_both / _n_oa:.1%})** sont exactement présentes dans
    l’ensemble ChEMBL CYP3A4 filtré. Ce nombre monte à {_both_any} si l’on admet les essais ChEMBL de moindre confiance, tels que
    ceux de DrugMatrix. Le Tanimoto médian vers la structure ChEMBL *non identique* la plus proche est **{_t.median():.2f}** :
    {(_t >= 0.4).mean():.1%} des composés ont un voisin ChEMBL à T ≥ 0.4, et {(_t >= 0.6).mean():.1%} à
    T ≥ 0.6. **{_novel_scaf:.0%}** des composés OpenADMET cycliques ont un squelette de Bemis–Murcko qui n’apparaît jamais dans
    ChEMBL, et les deux sources ne partagent que {int((scaffold_table['status'] == 'shared').sum())} squelettes.

    Le recouvrement est donc **{_verdict}**. Les cartes communes montrent le même tableau : OpenADMET occupe des régions que
    les enregistrements CYP3A4 de ChEMBL n’échantillonnent guère. Ce sont des molécules plus petites et moins lipophiles (masse molaire médiane
    {features.loc[features['membership'] != 'ChEMBL', 'MW'].median():.0f} contre {features.loc[features['membership'] != 'OpenADMET', 'MW'].median():.0f} Da)
    issues d’une bibliothèque de diversité, à comparer aux séries de chimie médicinale optimisées de ChEMBL.

    **Quels composés du défi ont une faible couverture ChEMBL ?**
    **{len(_low):,} ({len(_low) / len(nn_table):.0%})** n’ont aucun voisin ChEMBL avec T ≥ 0.30. Les plus isolés
    sont {", ".join(novel_candidates["name"].head(5))}. {_size_note} Pour ces composés, ChEMBL n’offre aucun analogue
    proche, et tout modèle appris sur ChEMBL serait en extrapolation.

    **Quels composés inhabituels méritent d’être inspectés, et pourquoi ?**
    Les scores d’Isolation Forest les plus élevés vont à des composés aux descripteurs extrêmes, par exemple {_prop_examples} : longues
    chaînes flexibles, molécules très polaires ou très petites. Avec les réglages actuels des curseurs, **{len(_both_flags)}**
    composés OpenADMET sont signalés *à la fois* comme atypiques par leurs propriétés et comme structurellement nouveaux, et ce sont les premiers à
    inspecter. Un marquage n’est pas une erreur. Vérifiez la structure, la standardisation et l’incertitude de la valeur
    (le pIC50 est-il inférieur à 4 ?), puis conservez le composé.

    > Réserve valable pour toutes les réponses : les puissances ChEMBL et OpenADMET proviennent d’essais différents. Nous avons comparé des **structures**,
    > et n’avons examiné les activités que côte à côte, par source et par essai.
    """
    )
    return


@app.cell
def _():
    export_button = mo.ui.run_button(label="💾 Écrire les tableaux et figures de résultats dans le dossier outputs")
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
        for _name, _df in _tables.items():                       # les grands tableaux sont compressés en gzip (pandas les lit tels quels)
            _big = len(_df) > 5000
            _df.to_csv(OUTPUT_DIR / (_name + ".gz" if _big else _name), index=False,
                       compression={"method": "gzip", "mtime": 0} if _big else None)
        _png = 0
        for _name, _fig in _figures.items():
            _fig.save(OUTPUT_DIR / f"{_name}.html")
            try:                                                  # le PNG nécessite le paquet optionnel vl-convert-python
                _fig.save(OUTPUT_DIR / f"{_name}.png", scale_factor=2)
                _png += 1
            except Exception:
                pass
        for _name, (_smiles, _legends) in _grids.items():
            (OUTPUT_DIR / _name).write_text(mol_grid_svg(list(_smiles), _legends) or "")
        _written = mo.callout(mo.md(f"{len(_tables)} tableaux, {len(_figures)} figures interactives (HTML"
                                    f"{', ' + str(_png) + ' PNG' if _png else ''}) et {len(_grids)} grilles de molécules "
                                    f"(SVG) écrits dans `{OUTPUT_DIR}`."), kind="success")
    mo.vstack(
        [
            _written or mo.md("Ou téléchargez les tableaux un par un :"),
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
    ### Pour aller plus loin (hors du cœur de l’atelier)
    L’entraînement de modèles, l’amarrage moléculaire (docking), les falaises d’activité (activity cliffs) et les autres
    représentations (t-SNE, TMAP) sont volontairement laissés de côté. Suites naturelles : la séance d’apprentissage
    automatique classique des [tutoriels](https://github.com/AxelRolov/chemoinformatics_tutorials)
    avec une *séparation par squelettes* (scaffold split), et le [défi OpenADMET CYP](https://huggingface.co/spaces/openadmet/cyp-challenge)
    lui-même (ouvert jusqu’en novembre 2026).

    **Références.** Zdrazil *et al.*, *The ChEMBL database in 2023*, Nucleic Acids Res. 2024 · Bento *et al.*,
    *An open source chemical structure curation pipeline using RDKit*, J. Cheminform. 2020 · Bemis & Murcko,
    *The properties of known drugs. 1. Molecular frameworks*, J. Med. Chem. 1996 · Landrum & Riniker, *Combining IC50 or
    Ki values from different sources is a source of significant noise*, J. Chem. Inf. Model. 2024 · Liu, Ting & Zhou,
    *Isolation Forest*, ICDM 2008 · McInnes, Healy & Melville, *UMAP*, arXiv:1802.03426.
    """)
    return


if __name__ == "__main__":
    app.run()
