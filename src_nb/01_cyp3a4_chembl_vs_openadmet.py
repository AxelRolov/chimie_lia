# %% [markdown]
"""
# Exploring CYP3A4 inhibitors: ChEMBL versus OpenADMET

**A two-hour chemoinformatics workshop — `chimie_lia`**

> **Where this comes from.** This workshop is built on the open teaching material of the
> [chemoinformatics tutorials](https://github.com/AxelRolov/chemoinformatics_tutorials) collection, which is itself a
> compilation of resources generously published by the chemoinformatics community. The original authors are named in
> the credits below and in [`CREDITS.md`](https://github.com/AxelRolov/chimie_lia/blob/main/CREDITS.md) — all credit
> for the substance belongs to them, and we thank them for making their work reusable. This session replaces the
> EGFR / ExpansionRx examples of those tutorials with a **matched pair of datasets measured on the same protein**, and
> adds the manual-download route, the cross-source comparison and the outlier analysis. The assembly was done with
> **Claude** (Anthropic) and then reviewed.

**The question we are going to answer with evidence:**

> ### How well does ChEMBL cover the chemical space of the OpenADMET compounds?

Two public datasets measure inhibition of **the same human enzyme, CYP3A4**:

| | **ChEMBL** | **OpenADMET CYP challenge** |
|---|---|---|
| what | ~30 years of medicinal-chemistry literature, manually curated | one industrial screening campaign, released for a blind challenge |
| target | `CHEMBL340`, UniProt `P08684` | CYP3A4 (plus 1A2, 2C9, 2D6) |
| size (this snapshot) | 13 887 IC50 records, 11 237 unique structures | 4 905 compounds, **2 335** with a CYP3A4 value |
| endpoint | IC50 (nM) from hundreds of different assays | pIC50 from **one** assay, with a credible interval |
| why it exists | people published what they found interesting | a library was screened systematically |

That last row is the whole story, and we will see it in the data.

**Learning goals.** After this session you will be able to
- retrieve bioactivity data for a target from ChEMBL **two ways** — from the website and from the API — and get the same answer;
- read assay descriptions and decide **which measurements may be compared**;
- curate structures with a documented, reproducible standardisation policy;
- quantify how much of one compound set is **covered** by another, with similarity, scaffolds and dimensionality reduction;
- flag unusual compounds, and say *in what sense* they are unusual.

**What you need.** Basic Python and pandas. Everything runs in Google Colab on CPU in well under two hours.

---
> **Credits — thank you to the original authors.** This session adapts, updates and revises material from
> - the [chemoinformatics tutorials](https://github.com/AxelRolov/chemoinformatics_tutorials) sessions
>   **02 · Molecular representations** (Morgan fingerprints, Tanimoto), **03 · Chemical databases** (ChEMBL client,
>   IC50/pIC50, Hugging Face datasets) and **04 · Exploratory data analysis** (standardisation, descriptors,
>   scaffolds, PCA/UMAP) — CC BY 4.0;
> - **TeachOpenCADD** talktorials **T001 · Compound data acquisition (ChEMBL)**, **T002 · Molecular filtering**,
>   **T005 · Compound clustering**, **T011 · Querying online API webservices** (Volkamer lab;
>   [GitHub](https://github.com/volkamerlab/teachopencadd), CC BY 4.0);
> - *Practical Cheminformatics Tutorials* by **Pat Walters** — `ChEMBL_data_curation`, `find_scaffolds`,
>   `visualizing_chemical_space` ([GitHub](https://github.com/PatWalters/practical_cheminformatics_tutorials), MIT);
> - *AI for Chemistry* (EPFL CH-457), **Schwaller group** — `04 - Unsupervised Learning`
>   ([GitHub](https://github.com/schwallergroup/ai4chem_course), MIT);
> - **OpenADMET** — the [CYP inhibition blind challenge](https://openadmet.ghost.io/announcing-openadmets-cyp-inhibition-blind-challenge/)
>   and its [dataset](https://huggingface.co/datasets/openadmet/cyp-challenge-train-test) (Apache-2.0);
> - **ChEMBL** (EMBL-EBI), data under CC BY-SA 3.0; and **RDKit**, without which none of this exists.
"""

# %%
# @title ⚙️ Setup — run this cell first (≈1–2 min on Colab)
import sys, os, io, json, time, subprocess, hashlib, textwrap, warnings

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "rdkit", "mols2grid", "umap-learn", "plotly", "seaborn",
                    "chembl_webresource_client", "ipywidgets"], check=False)
    try:                                   # lets mols2grid / plotly widgets render in Colab
        from google.colab import output as _colab_output
        _colab_output.enable_custom_widget_manager()
    except Exception:
        pass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import plotly.express as px
import plotly.graph_objects as go
from ipywidgets import interact, interact_manual
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Draw, Descriptors, rdMolDescriptors, rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.Draw import IPythonConsole
from tqdm.auto import tqdm

IPythonConsole.ipython_useSVG = True
RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.width", 160)
sns.set_theme(style="whitegrid", palette="deep")

# The two sources get one colour each, everywhere in the notebook.
SRC_COLOR = {"ChEMBL": "#4C72B0", "OpenADMET": "#DD8452", "both": "#55A868"}
RANDOM_SEED = 42

REPO_RAW = "https://raw.githubusercontent.com/AxelRolov/chimie_lia/main"

def data_path(filename):
    """Local copy if the repository is checked out, otherwise the raw GitHub URL (works in Colab)."""
    for candidate in (os.path.join("..", "data", filename), os.path.join("data", filename)):
        if os.path.exists(candidate):
            return candidate
    return f"{REPO_RAW}/data/{filename}"

def online(url, timeout=15):
    """True if a web service answers. Used to offer the live route and fall back to the cache."""
    try:
        return requests.head(url, timeout=timeout, allow_redirects=True).status_code < 500
    except requests.RequestException:
        return False

def draw(smiles, legends=None, per_row=4, size=(240, 180)):
    """Draw a list of SMILES as a grid image."""
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    keep = [i for i, m in enumerate(mols) if m is not None]
    return Draw.MolsToGridImage([mols[i] for i in keep], molsPerRow=per_row, subImgSize=size,
                                legends=None if legends is None else [str(legends[i]) for i in keep])

PROVENANCE = json.loads(requests.get(data_path("provenance.json")).text
                        if data_path("provenance.json").startswith("http")
                        else open(data_path("provenance.json")).read())
print(f"cached snapshot prepared on {PROVENANCE['prepared_on']} · "
      f"ChEMBL {PROVENANCE.get('chembl_release', '?')} · "
      f"OpenADMET revision {PROVENANCE['sources']['openadmet']['revision'][:8]}")
print("RDKit", Chem.rdBase.rdkitVersion, "| pandas", pd.__version__, "| Colab:", IN_COLAB)

# %% [markdown]
"""
## 0 · The target, the endpoint, and why this pair of datasets (0–10 min)

### CYP3A4

Cytochrome P450 3A4 is the single most important drug-metabolising enzyme in humans: it is responsible for the
clearance of roughly **half of all marketed drugs**. If your compound *inhibits* CYP3A4, every other drug the patient
takes that is cleared by CYP3A4 will accumulate — a **drug–drug interaction**. That is why CYP inhibition is measured
early in every discovery programme, and why it is an *anti-target*: you want your IC50 to be **high**, not low.

- ChEMBL target: [`CHEMBL340`](https://www.ebi.ac.uk/chembl/explore/target/CHEMBL340) · UniProt
  [`P08684`](https://www.uniprot.org/uniprotkb/P08684) · organism *Homo sapiens* · type **SINGLE PROTEIN**.

### IC50 and pIC50

**IC50** is the concentration of compound that halves the enzyme's activity. Because potencies span many orders of
magnitude we work on a log scale:
"""

# %% [markdown]
r"""
$$\mathrm{pIC50} \;=\; -\log_{10}\!\big(\mathrm{IC50\ in\ M}\big) \;=\; 9 - \log_{10}\!\big(\mathrm{IC50\ in\ nM}\big)$$

So 1 nM → pIC50 = 9 (very potent), 1 µM → 6, 100 µM → 4. **Move the slider** and watch the two scales track each other.
"""

# %%
@interact(log10_IC50_nM=(0.0, 6.0, 0.1))
def ic50_converter(log10_IC50_nM=3.0):
    ic50_nM = 10 ** log10_IC50_nM
    pic50 = 9 - np.log10(ic50_nM)
    unit = f"{ic50_nM:,.1f} nM" if ic50_nM < 1000 else f"{ic50_nM / 1000:,.2f} µM"
    verdict = ("very potent inhibitor" if pic50 >= 7 else
               "potent" if pic50 >= 6 else
               "moderate" if pic50 >= 5 else
               "weak — and in this range the measurement itself gets imprecise")
    print(f"IC50 = {unit:>18s}      pIC50 = {pic50:5.2f}      {verdict}")
    fig, ax = plt.subplots(figsize=(9, 1.5))
    ax.axvspan(4, 5, color="#eee"); ax.axvspan(7, 10, color="#e8f4e8")
    ax.scatter([pic50], [0], s=300, c="crimson", zorder=5)
    ax.set_xlim(2, 10); ax.set_yticks([]); ax.set_xlabel("pIC50")
    ax.set_title("grey: weak/imprecise region   ·   green: potent", fontsize=9)
    plt.show()

# %% [markdown]
"""
### The two datasets, and why comparing them is interesting

**ChEMBL** is an *aggregation*. Someone, somewhere, measured a CYP3A4 IC50, published it, and a curator entered it.
Hundreds of different assays, different probe substrates, different laboratories, thirty years apart. Compounds get
into ChEMBL because a medicinal chemist found them worth making and worth publishing.

**OpenADMET's CYP challenge** is a *campaign*. One library — Enamine diversity compounds plus FDA-approved drugs,
with hit expansion — screened in **one** assay protocol (recombinant enzyme, fluorogenic probe, 12-point dose–response,
Bayesian curve fitting that returns a credible interval for every pIC50). Compounds are there because they were in the
plate, not because anyone found them interesting.

> ⚠️ **The scientific point of this whole workshop.** The same protein does **not** mean interchangeable numbers.
> CYP3A4 inhibition depends on the probe substrate (CYP3A4 has a famously large, flexible active site and different
> probes report different IC50s for the same inhibitor), on the enzyme source, and on whether the compound was
> pre-incubated with NADPH. We will compare the two sets **structurally**, freely and jointly — and keep their
> **activity values separate**, labelled by source and assay, from beginning to end.

OpenADMET also distinguishes **direct inhibition** (no NADPH pre-incubation) from **time-dependent inhibition**
(TDI, +NADPH pre-incubation, which lets the enzyme turn the compound into something that inactivates it). This
workshop uses the **direct-inhibition** training file only; the TDI, single-concentration and Emax files are out of
scope. [Assay description](https://openadmet.ghost.io/announcing-openadmets-cyp-inhibition-blind-challenge/)

Let us look at three CYP3A4 inhibitors you may have met in a pharmacology course.
"""

# %%
KNOWN = pd.DataFrame([
    ("ketoconazole", "CC(=O)N1CCN(c2ccc(OCC3COC(Cn4ccnc4)(c4ccc(Cl)cc4Cl)O3)cc2)CC1",   # CHEMBL157101
     "antifungal; the textbook potent CYP3A4 inhibitor, used as the positive control in most assays"),
    ("ritonavir", "CC(C)c1nc(CN(C)C(=O)N[C@H](C(=O)N[C@@H](Cc2ccccc2)C[C@H](O)[C@H](Cc2ccccc2)NC(=O)OCc2cncs2)C(C)C)cs1",   # CHEMBL163
     "antiretroviral; inhibits CYP3A4 so strongly that it is co-dosed on purpose to boost other drugs"),
    ("midazolam", "Cc1ncc2n1-c1ccc(Cl)cc1C(c1ccccc1F)=NC2",
     "not an inhibitor — the classic *probe substrate*: its 1'-hydroxylation is what many assays measure"),
], columns=["name", "smiles", "why"])

for _, r in KNOWN.iterrows():
    print(f"{r['name']:<14s} {r['why']}")
draw(KNOWN["smiles"], legends=KNOWN["name"], per_row=3, size=(290, 210))

# %% [markdown]
"""
## 1 · Getting ChEMBL data, route A: the website (10–20 min)

Before writing any code, it is worth doing once by hand. You see what the database actually contains, and you learn
which columns exist — which is exactly what you need in order to write the API query afterwards.

### Do this now, in a second browser tab

1. Open <https://www.ebi.ac.uk/chembl/>.
2. Type **`CHEMBL340`** in the search box and open the target page.
3. **Check that it is the right target** before anything else:
   *Preferred name* **Cytochrome P450 3A4**, *Organism* **Homo sapiens**, *Type* **SINGLE PROTEIN**,
   and under *Component* the UniProt accession **P08684**. There are CYP3A4 entries for other organisms and
   `PROTEIN FAMILY` entries that lump several CYPs together — those are different targets.
4. Scroll to **Associated Bioactivities** and click the activities link (it will say ~14 000 items).
5. In the **Filters** panel on the left, filter **Standard Type → IC50**.
6. Press the **CSV** button in the *Tools* bar. ChEMBL queues an export job; when it finishes, your browser downloads
   a `.csv` file (inside a zip for big exports).

Keep that file. In Colab you can upload it with the cell below — and if you would rather not wait, the cell falls back
to a **200-row sample of exactly that export**, shipped with this repository.
"""

# %%
# Upload your own ChEMBL website export here (Colab: a "Choose files" button appears).
# Skip it and the cached 200-row sample is used instead — the rest of the notebook is identical either way.
UPLOAD_YOUR_OWN = False      # @param {type:"boolean"}

def read_export(raw_bytes, name=""):
    """Return the CSV text of a ChEMBL website export.

    ChEMBL delivers anything bigger than a few hundred rows as a **zip**, so a student following the
    instructions above will almost always have a `.zip` and not a `.csv`. Handle both.
    """
    if raw_bytes[:2] == b"PK":                       # zip magic number
        import zipfile
        with zipfile.ZipFile(io.BytesIO(raw_bytes)) as z:
            inner = [n for n in z.namelist() if n.lower().endswith((".csv", ".tsv", ".txt"))]
            if not inner:
                raise ValueError(f"no CSV inside {name}: {z.namelist()}")
            print(f"  {name} is a zip; reading {inner[0]} from inside it")
            raw_bytes = z.read(inner[0])
    return raw_bytes.decode("utf-8", errors="replace")

ui_raw = None
if UPLOAD_YOUR_OWN and IN_COLAB:
    from google.colab import files
    uploaded = files.upload()
    name = list(uploaded)[0]
    ui_raw = read_export(uploaded[name], name)
    print(f"uploaded {name}: {len(ui_raw):,} characters of CSV")
if ui_raw is None:
    path = data_path("chembl_ui_export_sample.csv")
    ui_raw = requests.get(path).text if path.startswith("http") else open(path, encoding="utf-8").read()
    print(f"using the cached website-export sample ({len(ui_raw):,} characters)")

# %% [markdown]
"""
### Never `read_csv` blindly — look at the bytes first

A file called `.csv` is not necessarily comma-separated. The ChEMBL website writes **semicolons**, quotes every field,
and writes the string `None` for missing values. Print the first two lines before you parse anything.
"""

# %%
first_lines = ui_raw.splitlines()[:2]
for line in first_lines:
    print(textwrap.shorten(line, width=300, placeholder=" …"))
print("\nseparator counts in the header:",
      {sep: first_lines[0].count(sep) for sep in [";", ",", "\t", "|"]})

# %%
ui = pd.read_csv(io.StringIO(ui_raw), sep=";", na_values=["None", "", "NA"])
print(ui.shape)
print("columns:", list(ui.columns)[:12], "…")
ui.head(3)

# %% [markdown]
"""
### Mapping the website columns onto a common schema

The website and the API describe the same records with **different column names**. Everything downstream in this
notebook speaks one vocabulary, so we translate once, here, and again for the API route in the next section. That is
what makes the two routes interchangeable.
"""

# %%
UI_TO_SCHEMA = {
    "Molecule ChEMBL ID": "molecule_chembl_id",
    "Molecule Name": "molecule_pref_name",
    "Smiles": "canonical_smiles",
    "Standard Type": "standard_type",
    "Standard Relation": "standard_relation",
    "Standard Value": "standard_value",
    "Standard Units": "standard_units",
    "pChEMBL Value": "pchembl_value",
    "Data Validity Comment": "data_validity_comment",
    "Potential Duplicate": "potential_duplicate",
    "Assay ChEMBL ID": "assay_chembl_id",
    "Assay Description": "assay_description",
    "Assay Type": "assay_type",
    "BAO Label": "bao_label",
    "Target ChEMBL ID": "target_chembl_id",
    "Target Organism": "target_organism",
    "Document ChEMBL ID": "document_chembl_id",
    "Document Year": "document_year",
}

def normalise_ui_export(df):
    """ChEMBL *website* export -> the common schema used everywhere below."""
    out = df.rename(columns=UI_TO_SCHEMA)[[c for c in UI_TO_SCHEMA.values() if c in
                                           df.rename(columns=UI_TO_SCHEMA).columns]].copy()
    # the website writes relations as "'='" (quoted inside the quoted field)
    out["standard_relation"] = out["standard_relation"].astype(str).str.strip("'\" ")
    out["standard_value"] = pd.to_numeric(out["standard_value"], errors="coerce")
    out["source_route"] = "website"
    return out

ui_norm = normalise_ui_export(ui)
print(ui_norm.shape)
print("relations:", ui_norm["standard_relation"].value_counts().to_dict())
ui_norm.head(3)

# %% [markdown]
"""
> **If your export has no structures.** Some ChEMBL export views give you activity rows without SMILES, or without
> the assay description. Then you export the *related compounds* and/or *related assays* lists from the same results
> page (the **RELATED COMPOUNDS** / **RELATED TARGETS** buttons next to CSV) and join them in pandas on
> `molecule_chembl_id` / `assay_chembl_id`. The join keys are stable ChEMBL IDs — that is what they are for:
>
> ```python
> acts  = pd.read_csv("activities.csv", sep=";")
> cpds  = pd.read_csv("compounds.csv",  sep=";")
> merged = acts.merge(cpds[["Molecule ChEMBL ID", "Smiles"]], on="Molecule ChEMBL ID", how="left")
> ```

### Exercise 1.1 — missing, or censored?

In `ui_norm`, some rows have **no** `standard_value`, and some have a value with a `standard_relation` of `>` rather
than `=`. These are **not** the same thing, and treating them the same way is one of the most common mistakes in
dataset curation.

1. Count the rows of each kind.
2. Look at the `assay_description` / `molecule_pref_name` of one row of each kind. What does each one mean physically?
3. Which of the two could legitimately be used in a *classification* model of "inhibitor / non-inhibitor", and which
   must **never** be used as if it were an exact number in a *regression*?
"""

# %%
# YOUR CODE HERE



# %% [markdown]
"""
<details><summary><b>Solution</b></summary>

```python
missing  = ui_norm["standard_value"].isna()
censored = ui_norm["standard_relation"].isin([">", "<", ">=", "<="]) & ui_norm["standard_value"].notna()
exact    = (ui_norm["standard_relation"] == "=") & ui_norm["standard_value"].notna()
print(f"{missing.sum()} missing, {censored.sum()} censored, {exact.sum()} exact")

display(ui_norm[missing].head(2)[["molecule_chembl_id", "standard_relation", "standard_value", "assay_description"]])
display(ui_norm[censored].head(2)[["molecule_chembl_id", "standard_relation", "standard_value", "assay_description"]])

# A MISSING value means: we do not know. The experiment may not have been done, the curve may have failed,
#   or the number may simply not be in the paper. It carries no information about the compound.
# A CENSORED value ("IC50 > 20000 nM") means: we DO know something. The compound was tested up to 20 µM and
#   did not reach 50 % inhibition there. That is real, usable information — it says "not potent".
#
# Classification: a censored ">" record is perfectly usable as an *inactive* (at that threshold). A missing
#   value is not usable at all; dropping it is right, calling it inactive is wrong.
# Regression: a censored record must NEVER be fed in as if 20000 were the measured IC50. It is a lower bound.
#   Either drop such rows, or use a model that understands censoring (a survival / Tobit-type loss).
# We keep them, in their own table, for exactly this reason — see section 3.
```
</details>
"""


# %% [markdown]
"""
## 2 · Getting ChEMBL data, route B: the API (20–30 min)

The same records, by URL. ChEMBL's REST API returns JSON and **pages** the results: each response carries a
`page_meta.next` link, and you keep following it until it is `null`.

The base query for our target:

```
https://www.ebi.ac.uk/chembl/api/data/activity.json?target_chembl_id=CHEMBL340&standard_type=IC50&limit=1000
```

Two things to do properly, and we do both: **retry** transient failures (a public API will occasionally return a 503),
and **cache** the raw answer so that a re-run — or a class of thirty students — does not hammer EMBL-EBI.
"""

# %%
EBI = "https://www.ebi.ac.uk"
ACTIVITY_URL = f"{EBI}/chembl/api/data/activity.json?target_chembl_id=CHEMBL340&standard_type=IC50&limit=1000"
CHEMBL_UP = online(f"{EBI}/chembl/api/data/status.json")
print("ChEMBL API reachable:", CHEMBL_UP)

if CHEMBL_UP:
    status = requests.get(f"{EBI}/chembl/api/data/status.json", timeout=30).json()
    print(f"release {status['chembl_db_version']} ({status['chembl_release_date']}), "
          f"{status['activities']:,} activities in total")

    # Always verify the target before downloading anything that depends on it.
    target = requests.get(f"{EBI}/chembl/api/data/target/CHEMBL340.json", timeout=30).json()
    print(f"\n{target['target_chembl_id']}: {target['pref_name']} | {target['organism']} | {target['target_type']}")
    print("components:", [(c['accession'], c['component_type']) for c in target['target_components']])
    assert target["organism"] == "Homo sapiens" and target["target_type"] == "SINGLE PROTEIN"
    assert "P08684" in [c["accession"] for c in target["target_components"]]
    print("✓ human CYP3A4, single protein, UniProt P08684 — this is the target we want")

# %% [markdown]
"""
### Paging, with retries

Set `MAX_PAGES` to a small number in class if you are impatient — but note the honest consequence: a truncated
download is a *biased* sample, because the API returns records in a fixed order, not a random one. Everything after
section 3 therefore uses the **complete** cached snapshot.
"""

# %%
MAX_PAGES = 3          # @param {type:"integer"}  — pages of 1000 records to pull live (the cache has all 14)

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def chembl_session():
    s = requests.Session()
    s.mount("https://", HTTPAdapter(max_retries=Retry(
        total=5, backoff_factor=1.5, status_forcelist=[429, 500, 502, 503, 504])))
    return s

def fetch_activities(url, max_pages=None):
    """Follow page_meta.next until the end (or max_pages), returning one DataFrame."""
    s, records, pages = chembl_session(), [], 0
    with tqdm(desc="downloading", unit=" rec") as bar:
        while url and (max_pages is None or pages < max_pages):
            r = s.get(url, timeout=120)
            r.raise_for_status()
            page = r.json()
            records += page["activities"]
            bar.total = page["page_meta"]["total_count"]
            bar.update(len(page["activities"]))
            nxt = page["page_meta"]["next"]
            url, pages = (EBI + nxt if nxt else None), pages + 1
    return pd.DataFrame(records)

api_live = fetch_activities(ACTIVITY_URL, max_pages=MAX_PAGES) if CHEMBL_UP else pd.DataFrame()
if len(api_live):
    print(f"\n{len(api_live)} records downloaded live, {api_live['assay_chembl_id'].nunique()} distinct assays")
    display(api_live[["molecule_chembl_id", "assay_chembl_id", "standard_type", "standard_relation",
                      "standard_value", "standard_units", "canonical_smiles"]].head(3))

# %% [markdown]
"""
### The same thing with the official client

`chembl_webresource_client` wraps the API: it handles paging and caching for you, and it is what the
[chemical-databases tutorial](https://github.com/AxelRolov/chemoinformatics_tutorials/blob/main/src_nb/03_chemical_databases.py)
uses. It is the right tool for everyday work; writing the paging loop once, as above, is how you understand what it does.
"""

# %%
if CHEMBL_UP:
    try:
        from chembl_webresource_client.new_client import new_client
        targets = pd.DataFrame(new_client.target.filter(target_components__accession="P08684")
                               .only("target_chembl_id", "pref_name", "organism", "target_type"))
        display(targets.head())
        single = targets.query("target_type == 'SINGLE PROTEIN' and organism == 'Homo sapiens'")
        print("the single-protein human entry:", single["target_chembl_id"].tolist())

        query = (new_client.activity
                 .filter(target_chembl_id="CHEMBL340", standard_type="IC50")
                 .only("activity_id", "molecule_chembl_id", "canonical_smiles", "assay_chembl_id",
                       "standard_relation", "standard_value", "standard_units"))
        print(f"the client reports {len(query):,} matching activities "
              f"(same number as the API's page_meta.total_count)")
    except Exception as e:
        print("client unavailable:", type(e).__name__, str(e)[:120])

# %% [markdown]
"""
### Both routes, one schema — and do they agree?

The API field names are already our schema; only `standard_value` needs a type fix. We can then check that the
website export and the API return the *same activity records*, which is the real test of "interchangeable routes".
"""

# %%
def normalise_api(df):
    """ChEMBL *API* records -> the common schema."""
    out = df.copy()
    out["standard_value"] = pd.to_numeric(out["standard_value"], errors="coerce")
    out["source_route"] = "api"
    return out

if len(api_live):
    api_norm = normalise_api(api_live)
    shared_cols = sorted(set(ui_norm.columns) & set(api_norm.columns))
    print("columns both routes provide:", shared_cols)

    # The website sample is 200 rows of the full export; compare on what they have in common.
    ui_ids = set(ui_norm["molecule_chembl_id"]) & set(api_norm["molecule_chembl_id"])
    print(f"\n{len(ui_ids)} molecules appear in both the (truncated) website sample and the (truncated) API pull")
    if ui_ids:
        probe = sorted(ui_ids)[0]
        print(f"\nsame molecule, {probe}, as each route reports it:")
        cols = ["molecule_chembl_id", "assay_chembl_id", "standard_relation", "standard_value", "standard_units"]
        display(pd.concat([ui_norm[ui_norm["molecule_chembl_id"] == probe][cols].assign(route="website"),
                           api_norm[api_norm["molecule_chembl_id"] == probe][cols].assign(route="api")]))

# %% [markdown]
"""
> **Record what you downloaded.** Databases change: ChEMBL ships a release every few months, and a query run today
> and in six months will return different numbers of rows. A result you cannot date is a result you cannot reproduce.
> The cached snapshot in this repository carries all of it in `data/provenance.json` — release, retrieval date, every
> query URL, row counts and SHA-256 checksums.
"""

# %%
print(json.dumps({k: v for k, v in PROVENANCE.items() if k != "files"}, indent=2)[:1400])
print("\nchecksummed files:", list(PROVENANCE["files"])[:4], "…")

# %% [markdown]
"""
### Exercise 2.1 — a second target, your choice

Write a function `chembl_ic50(target_chembl_id, max_pages=2)` that returns a normalised DataFrame for **any** target,
and run it on **CYP2D6** (`CHEMBL289`) or **CYP1A2** (`CHEMBL3356`) — two of the other isoforms in the OpenADMET file.
Check the target the same way we did for CYP3A4 before trusting the download. How many IC50 records does it have,
compared with CYP3A4's 13 887?
"""

# %%
# YOUR CODE HERE



# %% [markdown]
"""
<details><summary><b>Solution</b></summary>

```python
def chembl_ic50(target_chembl_id, max_pages=2):
    t = requests.get(f"{EBI}/chembl/api/data/target/{target_chembl_id}.json", timeout=30).json()
    print(f"{target_chembl_id}: {t['pref_name']} | {t['organism']} | {t['target_type']}")
    if t["organism"] != "Homo sapiens" or t["target_type"] != "SINGLE PROTEIN":
        raise ValueError("not a human single-protein target — check the ID before going further")
    url = (f"{EBI}/chembl/api/data/activity.json?target_chembl_id={target_chembl_id}"
           f"&standard_type=IC50&limit=1000")
    return normalise_api(fetch_activities(url, max_pages=max_pages))

cyp2d6 = chembl_ic50("CHEMBL289")
print(len(cyp2d6), "records downloaded")

# To get the TOTAL without downloading everything, ask for one record and read page_meta:
for tid, name in [("CHEMBL340", "CYP3A4"), ("CHEMBL289", "CYP2D6"), ("CHEMBL3356", "CYP1A2")]:
    meta = requests.get(f"{EBI}/chembl/api/data/activity.json?target_chembl_id={tid}"
                        f"&standard_type=IC50&limit=1", timeout=30).json()["page_meta"]
    print(f"{name:8s} {tid:12s} {meta['total_count']:6,} IC50 records")

# CYP3A4 has by far the most: it is the isoform that matters most for drug–drug interactions,
# so it is the one everybody measures and publishes.
```
</details>
"""

# %% [markdown]
"""
## 3 · Load, inspect and curate — both sources (30–45 min)

From here on we use the **complete cached snapshot** (all 13 887 ChEMBL records, and the pinned OpenADMET training
file), so that every number in the notebook is reproducible. Everything in this section is computed **live** in front
of you — the pre-computed CSVs in `data/` are only used at the end to check that we got the same answer.

### Two tables, not one

A habit worth acquiring: keep a **measurement table** (one row per experiment: which molecule, which assay, what
relation, what value, what units) separate from a **structure table** (one row per unique molecule). They answer
different questions, they have different numbers of rows, and merging them too early is how people end up averaging
across assays that should never have been averaged.
"""

# %%
acts = pd.read_csv(data_path("chembl_cyp3a4_ic50_raw.csv.gz"), low_memory=False)
assays = pd.read_csv(data_path("chembl_cyp3a4_assays.csv.gz"), low_memory=False)
oa_raw = pd.read_csv(data_path("openadmet_cyp_train_inhibition.csv"))

print(f"ChEMBL   : {len(acts):,} IC50 records · {acts['molecule_chembl_id'].nunique():,} molecules · "
      f"{acts['assay_chembl_id'].nunique():,} assays · {acts['document_chembl_id'].nunique():,} documents")
print(f"OpenADMET: {len(oa_raw):,} compounds × 4 isoforms")
acts.head(3)

# %%
# Shape, types, missingness — the three things to look at before anything else.
summary = pd.DataFrame({
    "dtype": acts.dtypes.astype(str),
    "missing": acts.isna().sum(),
    "missing %": (acts.isna().mean() * 100).round(1),
    "unique": acts.nunique(),
})
display(summary)
print("years:", int(acts["document_year"].min()), "–", int(acts["document_year"].max()))

# %% [markdown]
"""
Three columns in that table deserve a comment right away:

- **`standard_relation` is missing for 1 545 records.** Those are rows with no usable numeric result at all
  (`activity_comment` usually says *"Inhibition < 50 % @ 10 uM and thus dose-response curve not measured"* or
  *"Not Determined"*). Missing, not zero, not inactive.
- **`standard_units`**: almost everything is nM, but the raw `units` column underneath contains µM, mM and even
  µg/mL. ChEMBL's `standard_*` columns are the *harmonised* ones — always prefer them, and still check.
- **`data_validity_comment`**: ChEMBL's own curators flag suspicious records (*"Outside typical range"*,
  *"Potential transcription error"*). Free quality control; use it.
"""

# %%
print("standard_relation:", acts["standard_relation"].value_counts(dropna=False).to_dict())
print("standard_units   :", acts["standard_units"].value_counts(dropna=False).head(5).to_dict())
print("validity comments:", acts["data_validity_comment"].value_counts(dropna=False).to_dict())
print("\nwhat the value-less records say:")
print(acts.loc[acts["standard_value"].isna(), "activity_comment"].value_counts().head(4))

# %% [markdown]
"""
### The OpenADMET file

4 905 compounds, four isoforms, and for every measured pIC50 a **credible interval** and a **standard deviation**
from the Bayesian dose–response fit. Those uncertainty columns are the most valuable thing in the file and we keep
them to the end.

A blank CYP3A4 cell means **not measured on that isoform** — it does **not** mean "inactive".
"""

# %%
CYP = "CYP3A4_pIC50_direct_inhibition"
ISOFORMS = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]

print("rows:", len(oa_raw), "· unique SMILES:", oa_raw["SMILES"].nunique(),
      "· unique names:", oa_raw["Molecule_Name"].nunique())
coverage = pd.DataFrame({
    "measured": [oa_raw[f"{i}_pIC50_direct_inhibition"].notna().sum() for i in ISOFORMS],
}, index=ISOFORMS)
coverage["not measured"] = len(oa_raw) - coverage["measured"]
coverage["% of library"] = (coverage["measured"] / len(oa_raw) * 100).round(1)
display(coverage)

assert len(oa_raw) == 4905, "the pinned snapshot should have 4905 rows"
assert oa_raw[CYP].notna().sum() == 2335, "the pinned snapshot should have 2335 CYP3A4 values"
print("✓ 4 905 rows, 2 335 CYP3A4 values — as documented on the dataset card")

coverage[["measured", "not measured"]].plot.barh(stacked=True, figsize=(7, 2.6),
                                                 color=["#DD8452", "#dddddd"])
plt.xlabel("compounds"); plt.title("OpenADMET: how many compounds were measured on each isoform")
plt.legend(loc="lower right"); plt.tight_layout(); plt.show()

# %% [markdown]
"""
### What was actually measured? Reading the assay descriptions

2 810 different assays produced those 13 887 ChEMBL records. They are **not** the same experiment. Before we decide
which records may be compared, we read what they say. The repository ships
`data/chembl_cyp3a4_assay_review.csv` — one row per assay, with automatic flags from keyword rules over the
description (the rules are in [`utils/prepare_cached_data.py`](https://github.com/AxelRolov/chimie_lia/blob/main/utils/prepare_cached_data.py),
and are meant to be re-read and edited by an instructor before a class).

**Browse the biggest assays with the dropdown.** Look at how different they are.
"""

# %%
review = pd.read_csv(data_path("chembl_cyp3a4_assay_review.csv"))
print(f"{len(review)} assays · {int(review['direct_inhibition'].sum())} read as direct inhibition · "
      f"{int(review['time_dependent'].sum())} look time-dependent · "
      f"{int(review['induction'].sum())} induction · {int(review['activation'].sum())} activation · "
      f"{int(review['instructor_excluded'].sum())} excluded on review")
display(review.head(8)[["assay_chembl_id", "n_records", "n_equals", "confidence_score",
                        "direct_inhibition", "probe_substrate", "enzyme_system"]])

# The assays an instructor looked at and rejected — and why.
print("\nexcluded on review:")
display(review[review["instructor_excluded"]][
    ["assay_chembl_id", "n_records", "n_equals", "confidence_score", "excluded_reason"]])

# %%
TOP_ASSAYS = review.nlargest(25, "n_records")["assay_chembl_id"].tolist()

@interact(assay=TOP_ASSAYS)
def inspect_assay(assay=TOP_ASSAYS[0]):
    row = review[review["assay_chembl_id"] == assay].iloc[0]
    print(f"{assay}   {row['n_records']} records ({row['n_equals']} exact, {row['n_censored']} censored)")
    print(f"confidence {row['confidence_score']} · type {row['assay_type']} · {row['bao_label']}")
    stated = lambda v: str(v) if isinstance(v, str) and v.strip() else "(not stated in the description)"
    print(f"probe substrate : {stated(row['probe_substrate'])}")
    print(f"enzyme system   : {stated(row['enzyme_system'])}")
    flags = [f for f in ["time_dependent", "induction", "activation"] if row[f]]
    print(f"context flags   : {', '.join(flags) if flags else 'none → read as direct inhibition'}")
    print("\ndescription:")
    print(textwrap.fill(str(row["description"])[:900], width=110))
    sub = acts[(acts["assay_chembl_id"] == assay) & (acts["standard_relation"] == "=")]
    sub = sub[pd.to_numeric(sub["standard_value"], errors="coerce") > 0]
    if len(sub) > 3:
        p = 9 - np.log10(pd.to_numeric(sub["standard_value"]))
        plt.figure(figsize=(7, 2.2))
        plt.hist(p, bins=25, color=SRC_COLOR["ChEMBL"])
        plt.xlabel("pIC50"); plt.ylabel("records"); plt.xlim(2, 10)
        plt.title(f"{assay}: {len(p)} exact values, median pIC50 {p.median():.2f}")
        plt.tight_layout(); plt.show()

# %% [markdown]
"""
Notice as you click through:

- the biggest single assay, **CHEMBL1909138**, is an 871-compound DrugMatrix panel using the fluorogenic probe
  **BFC**. Only **38** of its records carry a number at all. The other **833** have no value and no relation —
  just the comment *"Inhibition < 50 % @ 10 uM and thus dose-response curve not measured"*. Read that carefully:
  in plain English it means the same thing as "IC50 > 10 µM", but ChEMBL stored it as **free text**, so
  programmatically it is a **missing** value, not a censored one. It is invisible to any filter on
  `standard_relation`, and that is exactly the trap Exercise 1.1 is about. A screening panel is not a potency
  measurement;
- several assays measure **time-dependent inhibition** (*"preincubated for 30 mins … followed by NADPH addition"*).
  That is a different quantity from direct inhibition and must not be pooled with it;
- the probe substrate changes from assay to assay: **midazolam**, **testosterone**, **BFC**, **BQ**, luciferin…
  CYP3A4's active site is large and accommodating, and different probes genuinely give different IC50s for the
  same inhibitor. This is not noise; it is the chemistry of the enzyme.

> ⚠️ **Do not blindly filter on `assay_type == "B"`.** The EGFR workflow in the original tutorial does, because
> kinase inhibition is a binding assay. Here, **11 582 of the 13 887** CYP3A4 records are typed **`A` (ADME)**, which
> is exactly where CYP inhibition data belongs. Requiring `B` would silently throw away five sixths of the data.

### The curation rules

Here is the policy, written out, and applied in one function so you can read it, disagree with it, and change it:

1. the target is human CYP3A4 (checked at the source);
2. the assay has **ChEMBL confidence score 9** — "direct single protein target assigned" (score 8 means the
   assignment was made by homology, which for a CYP family is a real risk of isoform confusion);
3. the assay context is **direct inhibition**: the description gives **positive evidence** that it is a CYP
   experiment at all (it mentions CYP3A4 / CYP / P450), it shows no pre-incubation/TDI, induction or activation,
   and it is not on the instructor's explicit exclusion list;
4. the record has a structure;
5. ChEMBL has flagged no data-validity problem;
6. the value is finite, positive, and in **nM**;
7. the relation is **`=`** → the quantitative set. Relations `>`, `<`, `>=`, `<=` go to a **separate censored table**
   and are never used as exact numbers.

`potential_duplicate` is reported but not used as a filter: ChEMBL sets it when the same value appears twice in one
document, which is often a legitimate replicate. We deduplicate on **structure** instead, further down.
"""

# %%
CONTEXT_COLS = ["direct_inhibition", "mentions_target", "time_dependent", "induction", "activation",
                "instructor_excluded", "excluded_reason", "probe_substrate", "enzyme_system",
                "confidence_score"]

def curate_chembl(df, review):
    """Apply the seven rules above. Returns (quantitative, censored, funnel)."""
    rv = review.set_index("assay_chembl_id")
    d = df.join(rv[CONTEXT_COLS], on="assay_chembl_id")
    steps = [("all IC50 records for CHEMBL340", len(d))]

    d = d[d["target_organism"] == "Homo sapiens"];        steps.append(("human target", len(d)))
    d = d[d["confidence_score"] == 9];                    steps.append(("assay confidence 9", len(d)))
    d = d[d["direct_inhibition"]];                        steps.append(("direct inhibition context", len(d)))
    d = d[d["canonical_smiles"].notna()];                 steps.append(("has a structure", len(d)))
    d = d[d["data_validity_comment"].isna()];             steps.append(("no validity flag", len(d)))
    v = pd.to_numeric(d["standard_value"], errors="coerce")
    d = d[np.isfinite(v) & (v > 0) & (d["standard_units"] == "nM")]
    steps.append(("finite positive value in nM", len(d)))

    censored = d[d["standard_relation"].isin([">", "<", ">=", "<="])].copy()
    quant = d[d["standard_relation"] == "="].copy()
    steps.append(("relation '=' → quantitative set", len(quant)))

    # Account for every record: anything that is neither exact nor censored is reported, not lost in silence.
    other = d[~d["standard_relation"].isin(["=", ">", "<", ">=", "<="])]
    if len(other):
        print(f"note: {len(other)} record(s) have a relation that is neither exact nor censored "
              f"({other['standard_relation'].value_counts().to_dict()}) — dropped, and counted here so the "
              f"three subsets add up: {len(quant)} + {len(censored)} + {len(other)} = {len(d)}")

    for frame in (quant, censored):
        frame["standard_value"] = pd.to_numeric(frame["standard_value"])
        frame["pIC50"] = 9 - np.log10(frame["standard_value"])
    funnel = pd.DataFrame(steps, columns=["step", "records"])
    funnel["% of raw"] = (funnel["records"] / len(df) * 100).round(1)
    funnel["lost here"] = funnel["records"].shift().sub(funnel["records"]).fillna(0).astype(int)
    return quant, censored, funnel

ch_quant, ch_censored, funnel = curate_chembl(acts, review)
display(funnel)

plt.figure(figsize=(9, 3.4))
plt.barh(funnel["step"], funnel["records"], color=SRC_COLOR["ChEMBL"])
plt.gca().invert_yaxis(); plt.xlabel("records surviving")
for y, n in enumerate(funnel["records"]):
    plt.text(n + 150, y, f"{n:,}", va="center", fontsize=9)
plt.title("ChEMBL curation funnel"); plt.tight_layout(); plt.show()

print(f"\nquantitative set : {len(ch_quant):,} records, {ch_quant['molecule_chembl_id'].nunique():,} molecules")
print(f"censored set     : {len(ch_censored):,} records, kept apart — "
      f"{(ch_censored['standard_relation'] == '>').sum():,} of them are '>' (tested, not potent)")

# %% [markdown]
"""
### Always look at your most extreme values

A curated table is a hypothesis, not a result. The cheapest and most effective check you will ever run is to sort
by the quantity you care about and **look at the top ten**. Extreme values are where errors hide, because an error
has to be large to reach the top.
"""

# %%
top10 = ch_quant.nlargest(10, "pIC50")[["molecule_chembl_id", "molecule_pref_name", "assay_chembl_id",
                                        "standard_value", "pIC50", "probe_substrate"]]
display(top10.round(2))
for aid in top10["assay_chembl_id"].unique()[:4]:
    print(f"{aid}: {str(review.loc[review['assay_chembl_id'] == aid, 'description'].iloc[0])[:130]}")

# %% [markdown]
"""
The most potent entry is **ketoconazole at 0.9 nM** — exactly the right answer. Ketoconazole is the textbook
CYP3A4 inhibitor and the positive control of half the assays in this dataset. When the extremes of your table are
the compounds a pharmacologist would have predicted, that is real (if weak) evidence that the pipeline works.

**It was not always so.** Before the `mentions_target` rule was added, the four most potent "CYP3A4 inhibitors" in
this very table came from `CHEMBL3705484` — *"Homogenous Time-Resolved Fluorescence Assay (HTRF) … GST-hMDM2
(aa 1-188), biotinylated-p53 …"*. An **MDM2–p53 binding assay**. It had confidence score 9, assay type B, a human
organism and no TDI keywords, so it passed every single rule above except the one that asks *"is this description
even about a CYP?"*

> ⚠️ **Confidence score 9 means ChEMBL is confident about the target *assignment*, not about your interpretation of
> the experiment.** No automatic rule set is ever finished. That is why `data/chembl_cyp3a4_assay_review.csv` ships
> with an `excluded_reason` column: the keyword flags do the first 99 %, a human does the rest, and the exclusions
> are written down where you can read and dispute them.
"""

# %% [markdown]
"""
Look at what the funnel says: of 13 887 raw records, **3 980** survive as exact, comparable, direct-inhibition
measurements. That is 29 %. The discarded majority is not junk — it is 3 710 censored records that *do* carry
information, plus records from assays measuring something else. Curation is mostly about **separating** data, not
deleting it.

### One standardisation policy, both sources

The same molecule can be stored as a salt, a charged form, or with different stereochemistry conventions. If we do
not unify them, "the same compound" will look like two different compounds in one source and one in the other — and
our overlap estimate will be wrong. The policy, applied **identically** to ChEMBL and to OpenADMET:

1. parse the SMILES (failures are counted, never silently dropped);
2. `Cleanup()` — sanitise, normalise functional groups (nitro, azide …), reionise;
3. `LargestFragmentChooser(preferOrganic=True)` — drop counter-ions and solvents;
4. `Uncharger()` — neutralise where chemistry allows (a quaternary ammonium stays charged, correctly);
5. canonical **isomeric** SMILES — **stereochemistry is kept**.

**No tautomer canonicalisation.** It is slow, it is a modelling choice rather than a truth, and ChEMBL's own
structure pipeline does not do it either. The original SMILES is preserved in every table, so you can always redo
this differently.
"""

# %%
_cleanup = rdMolStandardize.CleanupParameters()
_largest = rdMolStandardize.LargestFragmentChooser(preferOrganic=True)
_uncharge = rdMolStandardize.Uncharger()

def standardize(smiles):
    """-> (standardised isomeric SMILES or None, status, n_fragments, had_charge)"""
    if not isinstance(smiles, str) or not smiles.strip():
        return None, "empty", np.nan, np.nan
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "unparsable", np.nan, np.nan
    n_frag = len(Chem.GetMolFrags(mol))
    charged = any(a.GetFormalCharge() != 0 for a in mol.GetAtoms())
    try:
        mol = rdMolStandardize.Cleanup(mol, _cleanup)
        mol = _largest.choose(mol)
        mol = _uncharge.uncharge(mol)
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol, isomericSmiles=True), "ok", n_frag, charged
    except Exception:
        return None, "failed", n_frag, charged

# A few examples first, so you can see what it does and does not change.
demo = ["CC(=O)Oc1ccccc1C(=O)[O-].[Na+]", "OC(=O)CC[NH3+]", "Oc1ccccn1", "O=c1cccc[nH]1",
        "C[C@H](N)C(=O)O", "CCCCCCCCCCCCCCCCCC[N+](C)(C)CCCCCCCCCCCCCCCCCC"]
for s in demo:
    std, status, nf, ch = standardize(s)
    print(f"{s:<52s} -> {std}")

# %% [markdown]
"""
Read those six lines carefully, they contain the whole lesson:

- the **sodium salt** loses its counter-ion;
- the **ammonium** is neutralised to the free amine;
- the two **pyridinol/pyridone tautomers stay distinct** — we chose not to canonicalise tautomers;
- **alanine keeps its `[C@H]`** — stereochemistry survives;
- the **quaternary ammonium keeps its charge**, because there is no proton to remove. Correct behaviour.
"""

# %%
def standardize_frame(df, smiles_col, desc="standardising"):
    """Standardise the unique SMILES once, then map back — 11 000 records, ~10 s."""
    uniq = df[smiles_col].dropna().unique()
    table = {s: standardize(s) for s in tqdm(uniq, desc=desc, unit=" mol")}
    out = df.copy()
    out["smiles_std"] = out[smiles_col].map(lambda s: table.get(s, (None, "empty", np.nan, np.nan))[0])
    out["std_status"] = out[smiles_col].map(lambda s: table.get(s, (None, "empty", np.nan, np.nan))[1])
    out["n_fragments"] = out[smiles_col].map(lambda s: table.get(s, (None, "empty", np.nan, np.nan))[2])
    out["had_charge"] = out[smiles_col].map(lambda s: table.get(s, (None, "empty", np.nan, np.nan))[3])
    return out

t0 = time.time()
ch_quant = standardize_frame(ch_quant, "canonical_smiles", "ChEMBL quantitative")
ch_all = standardize_frame(acts[acts["canonical_smiles"].notna()], "canonical_smiles", "ChEMBL, all records")
oa = standardize_frame(oa_raw[oa_raw[CYP].notna()].copy(), "SMILES", "OpenADMET CYP3A4")
print(f"\n{time.time() - t0:.0f} s")

for name, frame in [("ChEMBL (all IC50 records)", ch_all), ("OpenADMET CYP3A4", oa)]:
    st = frame["std_status"].value_counts().to_dict()
    print(f"{name:28s} {st} · salts: {int((frame['n_fragments'] > 1).sum())} · "
          f"charged inputs: {int(frame['had_charge'].sum())}")

ch_quant = ch_quant[ch_quant["smiles_std"].notna()]
ch_all = ch_all[ch_all["smiles_std"].notna()]
oa = oa[oa["smiles_std"].notna()].copy()

# %% [markdown]
"""
### The structure tables

Now, and only now, we collapse to **one row per unique structure**. Two structure tables:

- the **ChEMBL reference set**: every unique structure ChEMBL has *any* CYP3A4 IC50 record for — censored ones
  included. This is "the chemical space ChEMBL has looked at for this target", and it is the fair reference for
  asking about coverage. Using only the 3 980 exact records would make ChEMBL's coverage look artificially small.
- the **OpenADMET set**: the 2 335 compounds with a CYP3A4 pIC50.

Membership — `ChEMBL`, `OpenADMET` or `both` — is decided on the **standardised isomeric SMILES**, not on a
fingerprint similarity of 1.0. Two different molecules can share a fingerprint; only identity is identity.
"""

# %%
ch_struct = (ch_all.sort_values("activity_id")
             .groupby("smiles_std", as_index=False)
             .agg(molecule_chembl_id=("molecule_chembl_id", "first"),
                  molecule_pref_name=("molecule_pref_name", "first"),
                  n_records=("activity_id", "size")))
oa_struct = oa.rename(columns={CYP: "pIC50",
                               f"{CYP}_conf_low": "pIC50_conf_low",
                               f"{CYP}_conf_high": "pIC50_conf_high",
                               f"{CYP}_std": "pIC50_std"}).copy()
oa_struct["pIC50_ci_width"] = oa_struct["pIC50_conf_high"] - oa_struct["pIC50_conf_low"]
oa_struct["weak_potency_flag"] = oa_struct["pIC50"] < 4          # a caution flag, never a filter

CH_SET, OA_SET = set(ch_struct["smiles_std"]), set(oa_struct["smiles_std"])
BOTH = CH_SET & OA_SET

counts = pd.DataFrame({
    "records in": [len(acts), len(oa_raw)],
    "records kept": [len(ch_quant), len(oa_struct)],
    "unique structures": [len(ch_struct), len(oa_struct)],
}, index=["ChEMBL", "OpenADMET"])
display(counts)
print(f"\nunion of the two structure sets : {len(CH_SET | OA_SET):,}")
print(f"present in BOTH (exact structure): {len(BOTH):,}  "
      f"= {len(BOTH) / len(OA_SET) * 100:.1f} % of the OpenADMET compounds")

# cross-check against the pre-computed snapshot
assert len(BOTH) == PROVENANCE["curation"]["exact_overlap_structures"], "overlap differs from the cached snapshot"
print("✓ matches the cached snapshot in provenance.json")

# %% [markdown]
"""
**52 molecules out of 2 335.** Two percent. That is the first — and already quite strong — answer to our question,
and it is worth pausing on: these are the only compounds for which both sources give you a CYP3A4 number for *the
very same molecule*. Let us look at a few of them, and compare the two measurements. Remember: different assays.
"""

# %%
shared = (oa_struct[oa_struct["smiles_std"].isin(BOTH)]
          .merge(ch_struct, on="smiles_std", suffixes=("_oa", "_ch")))
ch_med = (ch_quant.groupby("smiles_std")["pIC50"].agg(["median", "size"])
          .rename(columns={"median": "pIC50_chembl", "size": "n_chembl"}))
shared = shared.join(ch_med, on="smiles_std")
with_both = shared[shared["pIC50_chembl"].notna()]
print(f"{len(shared)} shared structures; only {len(with_both)} of them also have an exact, curated ChEMBL "
      f"pIC50 to compare against")
rest = shared[shared["pIC50_chembl"].isna()]
rest_records = acts[acts["canonical_smiles"].isin(
    ch_all.loc[ch_all["smiles_std"].isin(rest["smiles_std"]), "canonical_smiles"])]
print(f"the other {len(rest)} have no exact, curated ChEMBL value. Their {len(rest_records)} raw ChEMBL records "
      f"break down as:")
print(f"   {int(rest_records['standard_value'].isna().sum())} with no number at all "
      f"(mostly the DrugMatrix free-text 'Inhibition < 50 % @ 10 uM')")
print(f"   {int((rest_records['standard_relation'] == '>').sum())} true censored '>' records")
print(f"   {int((rest_records['standard_relation'] == '=').sum())} exact records that our curation rules excluded")
print("→ the molecules the two sources share are overwhelmingly ones ChEMBL screened and found inactive,\n"
      "  and recorded in a way that carries no usable number.")
display(with_both[["molecule_pref_name", "Molecule_Name", "pIC50", "pIC50_chembl", "n_chembl"]]
        .rename(columns={"pIC50": "pIC50 (OpenADMET)", "pIC50_chembl": "median pIC50 (ChEMBL)"})
        .sort_values("median pIC50 (ChEMBL)", ascending=False).head(12).round(2))

# %%
if len(with_both) >= 4:
    fig = px.scatter(with_both, x="pIC50_chembl", y="pIC50", hover_name="molecule_pref_name",
                     hover_data={"Molecule_Name": True, "n_chembl": True, "smiles_std": False},
                     labels={"pIC50_chembl": "median pIC50, ChEMBL (many assays)",
                             "pIC50": "pIC50, OpenADMET (one assay)"},
                     title=f"The {len(with_both)} molecules measured by both — hover for names",
                     width=620, height=480)
    lo, hi = 3, 9
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="y = x",
                             line=dict(dash="dash", color="grey")))
    fig.update_traces(marker=dict(size=10, color=SRC_COLOR["both"]), selector=dict(mode="markers"))
    fig.show()
    delta = (with_both["pIC50"] - with_both["pIC50_chembl"]).abs()
    print(f"Pearson r = {with_both['pIC50'].corr(with_both['pIC50_chembl']):.2f} on {len(with_both)} points")
    print(f"absolute differences (log units): {sorted(delta.round(2))}")
    print(f"median {delta.median():.2f} · largest {delta.max():.2f} · "
          f"{int((delta > 1).sum())} pair(s) differ by more than one log unit")

# %% [markdown]
"""
> **Be careful what you conclude from that plot — there are seven points on it.** Most of the agree well; one pair
> differs by more than a log unit. Seven comparisons cannot establish how well two sources agree, and pretending
> otherwise would be exactly the kind of over-reading this workshop is about.
>
> What we *can* say is this. First, the comparison is barely possible at all: of 52 shared molecules only 7 have an
> exact curated value on both sides. Second, the published systematic answer is not reassuring. Landrum & Riniker
> (2024) measured this properly across all of ChEMBL, comparing compounds assayed twice against the same target:
> with minimal curation the **mean absolute error is 0.50 log units**, 64 % of pairs differ by more than 0.3 and
> **27 % by more than a full log unit** (Kendall τ = 0.51). Even under their strictest metadata-matched curation —
> which throws away 99 % of the data — the MAE is still 0.27 and 13 % of pairs differ by more than a log unit.
> Different probe substrate, different enzyme preparation, different laboratory.
>
> So: **compare structures jointly; keep activities apart.** That rule holds for the rest of the notebook.

### Exercise 3.1 — rebuild the funnel with a different rule

Re-run `curate_chembl` with confidence score **8 or 9** accepted instead of 9 only (score 8 = *"homologous single
protein target assigned"*). How many extra records do you gain? Then look at which assays those extra records come
from — would you trust them for CYP3A4 specifically? (Hint: `review[review.confidence_score == 8]`.)
"""

# %%
# YOUR CODE HERE



# %% [markdown]
"""
<details><summary><b>Solution</b></summary>

```python
def curate_relaxed(df, review, min_conf=8):
    rv = review.set_index("assay_chembl_id")
    d = df.join(rv[CONTEXT_COLS], on="assay_chembl_id")
    d = d[(d["target_organism"] == "Homo sapiens") & (d["confidence_score"] >= min_conf)
          & d["direct_inhibition"] & d["canonical_smiles"].notna() & d["data_validity_comment"].isna()]
    v = pd.to_numeric(d["standard_value"], errors="coerce")
    d = d[np.isfinite(v) & (v > 0) & (d["standard_units"] == "nM")]
    return d[d["standard_relation"] == "="]

relaxed = curate_relaxed(acts, review, min_conf=8)
print(f"{len(relaxed)} records with confidence >= 8, versus {len(ch_quant)} with confidence 9 only "
      f"(+{len(relaxed) - len(ch_quant)})")

extra_assays = set(relaxed["assay_chembl_id"]) - set(ch_quant["assay_chembl_id"])
display(review[review["assay_chembl_id"].isin(extra_assays)]
        .nlargest(8, "n_records")[["assay_chembl_id", "n_records", "confidence_score", "description"]])

# Many of the score-8 descriptions are simply "Inhibition of CYP3A4" with no organism and no probe stated.
# ChEMBL could not confirm that the protein tested is the human CYP3A4 rather than a homologue, so it
# assigned the target by homology. For a CYP — a family of close isoforms where cross-reactivity is the
# whole problem — that is exactly the ambiguity we cannot afford. Keeping confidence 9 costs us records
# and buys us the right to say "this is human CYP3A4".
```
</details>
"""


# %% [markdown]
"""
## 4 · Exploratory analysis: what kind of molecules are these? (45–60 min)

Nine standard descriptors, computed the same way for both sources.
"""

# %%
DESCRIPTORS = {
    "MW": Descriptors.MolWt,
    "cLogP": Descriptors.MolLogP,
    "TPSA": rdMolDescriptors.CalcTPSA,
    "HBD": rdMolDescriptors.CalcNumHBD,
    "HBA": rdMolDescriptors.CalcNumHBA,
    "RotB": rdMolDescriptors.CalcNumRotatableBonds,
    "AromRings": rdMolDescriptors.CalcNumAromaticRings,
    "FracCSP3": rdMolDescriptors.CalcFractionCSP3,
    "HeavyAtoms": lambda m: m.GetNumHeavyAtoms(),
}
DESC_COLS = list(DESCRIPTORS)

def add_descriptors(df, smiles_col="smiles_std"):
    rows = []
    for s in tqdm(df[smiles_col], desc="descriptors", unit=" mol"):
        m = Chem.MolFromSmiles(s)
        rows.append({k: (f(m) if m is not None else np.nan) for k, f in DESCRIPTORS.items()})
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)

ch_struct = add_descriptors(ch_struct)
oa_struct = add_descriptors(oa_struct)

ch_struct["source"] = "ChEMBL"
oa_struct["source"] = "OpenADMET"
space = pd.concat([ch_struct[["smiles_std", "source"] + DESC_COLS],
                   oa_struct[["smiles_std", "source"] + DESC_COLS]], ignore_index=True)
space["membership"] = np.where(space["smiles_std"].isin(BOTH), "both", space["source"])
print(space.groupby("source")[DESC_COLS].median().round(2).to_string())

# %% [markdown]
"""
### Compare the distributions, one descriptor at a time

**Pick a descriptor from the dropdown.** The overlaid histograms, the medians and the effect size update together.
"""

# %%
@interact(descriptor=DESC_COLS, show_quartiles=[True, False])
def compare_descriptor(descriptor="MW", show_quartiles=True):
    a = ch_struct[descriptor].dropna()
    b = oa_struct[descriptor].dropna()
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.4), gridspec_kw={"width_ratios": [2, 1]})
    lo, hi = np.nanpercentile(np.concatenate([a, b]), [0.5, 99.5])
    bins = np.linspace(lo, hi, 45)
    axes[0].hist(a, bins=bins, alpha=.62, density=True, label=f"ChEMBL (n={len(a):,})",
                 color=SRC_COLOR["ChEMBL"])
    axes[0].hist(b, bins=bins, alpha=.62, density=True, label=f"OpenADMET (n={len(b):,})",
                 color=SRC_COLOR["OpenADMET"])
    if show_quartiles:
        for v, c in [(a.median(), SRC_COLOR["ChEMBL"]), (b.median(), SRC_COLOR["OpenADMET"])]:
            axes[0].axvline(v, color=c, ls="--", lw=2)
    axes[0].set_xlabel(descriptor); axes[0].set_ylabel("density"); axes[0].legend()
    axes[0].set_title(f"{descriptor} — dashed lines are the medians")
    sns.boxplot(data=space, x="source", y=descriptor, ax=axes[1],
                hue="source", legend=False, palette=SRC_COLOR, showfliers=False)
    axes[1].set_xlabel("")
    plt.tight_layout(); plt.show()

    pooled = np.sqrt((a.var() + b.var()) / 2)
    d = (b.mean() - a.mean()) / pooled if pooled else np.nan
    print(f"median  ChEMBL {a.median():8.2f}   OpenADMET {b.median():8.2f}   "
          f"difference {b.median() - a.median():+.2f}")
    print(f"Cohen's d (OpenADMET − ChEMBL) = {d:+.2f}  "
          f"({'negligible' if abs(d) < .2 else 'small' if abs(d) < .5 else 'medium' if abs(d) < .8 else 'large'})")

# %% [markdown]
"""
### Are the descriptors telling us nine different things?

No. Many of them measure size in different clothes. A correlation heatmap makes that obvious — and it matters for
PCA later, because correlated inputs collapse onto the first component.
"""

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
for ax, (name, frame) in zip(axes, [("ChEMBL", ch_struct), ("OpenADMET", oa_struct)]):
    sns.heatmap(frame[DESC_COLS].corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, ax=ax, cbar=False, annot_kws={"size": 7})
    ax.set_title(f"{name} (n = {len(frame):,})")
plt.tight_layout(); plt.show()

# %%
missingness = pd.DataFrame({
    "ChEMBL": ch_struct[DESC_COLS].isna().sum(),
    "OpenADMET": oa_struct[DESC_COLS].isna().sum(),
})
print("descriptor values that could not be computed:")
print(missingness.T.to_string())

# %% [markdown]
"""
### The activity distributions — side by side, never pooled

Now the part that needs the most care. We plot both, **labelled by source**, and we do **not** merge them.
"""

# %%
act = pd.concat([
    ch_quant[["pIC50"]].assign(source="ChEMBL", kind="exact (relation '=')"),
    ch_censored[["pIC50"]].assign(source="ChEMBL", kind="censored ('>' or '<') — NOT a measurement"),
    oa_struct[["pIC50"]].assign(source="OpenADMET", kind="exact (Bayesian DRC fit)"),
], ignore_index=True)

fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
bins = np.linspace(1.5, 10.5, 50)
axes[0].hist(ch_quant["pIC50"], bins=bins, alpha=.7, density=True,
             label=f"ChEMBL exact (n={len(ch_quant):,})", color=SRC_COLOR["ChEMBL"])
axes[0].hist(oa_struct["pIC50"], bins=bins, alpha=.7, density=True,
             label=f"OpenADMET (n={len(oa_struct):,})", color=SRC_COLOR["OpenADMET"])
axes[0].axvline(4, color="crimson", ls=":", lw=2)
axes[0].text(4.05, axes[0].get_ylim()[1] * .9, "pIC50 = 4", color="crimson", fontsize=9)
axes[0].set_xlabel("pIC50"); axes[0].set_ylabel("density"); axes[0].legend(fontsize=8)
axes[0].set_title("exact values only")
sns.violinplot(data=act, x="kind", y="pIC50", ax=axes[1], hue="kind", legend=False,
               palette=["#4C72B0", "#b0b0b0", "#DD8452"], cut=0)
axes[1].set_xticks(range(3))
axes[1].set_xticklabels(["ChEMBL\nexact", "ChEMBL\ncensored", "OpenADMET"], fontsize=8)
axes[1].set_xlabel(""); axes[1].set_title("including the censored records, shown apart")
plt.tight_layout(); plt.show()

print(act.groupby(["source", "kind"])["pIC50"].describe()[["count", "25%", "50%", "75%"]].round(2).to_string())

# %% [markdown]
"""
The ChEMBL distribution sits **about one log unit higher**. Three things contribute, and you cannot separate them
by looking at this plot alone:

1. **Selection.** ChEMBL contains compounds people chose to publish. A medicinal chemist writing a paper about a
   CYP3A4 liability reports the compounds that *had* one.
2. **Censoring.** The weak compounds are in ChEMBL too — but as `> 10 µM` records, which we moved to the censored
   table. The grey violin shows where they live. Removing them from the "exact" set shifts the visible distribution up.
3. **Assay.** Different probe substrates and enzyme preparations give systematically different IC50s.

OpenADMET, by contrast, screened whatever was in the library, and reports every curve it fitted — including the
flat ones. Its distribution is what an **unselected** library looks like.

> ⚠️ We deliberately do **not** draw an "active/inactive" line. Any threshold here would be arbitrary, and would
> hide exactly the difference we have just described.

### Uncertainty: the most under-used column in the file

Every OpenADMET pIC50 comes with a 95 % credible interval. **The interval gets dramatically wider as the compound
gets weaker** — which is the honest statistical statement of "we could not reach the IC50 within the tested
concentration range". Hover over the points.
"""

# %%
fig = px.scatter(oa_struct, x="pIC50", y="pIC50_ci_width", color="weak_potency_flag",
                 hover_name="Molecule_Name",
                 hover_data={"pIC50": ":.2f", "pIC50_ci_width": ":.2f", "pIC50_std": ":.3f",
                             "weak_potency_flag": False},
                 color_discrete_map={True: "#C44E52", False: "#55A868"},
                 labels={"pIC50_ci_width": "width of the 95 % credible interval (log units)",
                         "weak_potency_flag": "pIC50 < 4"},
                 title="OpenADMET: weak compounds have imprecise pIC50s", width=760, height=440)
fig.update_traces(marker=dict(size=4, opacity=.65))
fig.show()

print(f"{int(oa_struct['weak_potency_flag'].sum()):,} of {len(oa_struct):,} compounds have pIC50 < 4 "
      f"({oa_struct['weak_potency_flag'].mean():.0%})")
print(f"median CI width  : pIC50 < 4 → {oa_struct.loc[oa_struct['weak_potency_flag'], 'pIC50_ci_width'].median():.2f}"
      f"   ·   pIC50 ≥ 4 → {oa_struct.loc[~oa_struct['weak_potency_flag'], 'pIC50_ci_width'].median():.2f}"
      f" log units")

# %% [markdown]
"""
> **How to treat these.** `weak_potency_flag` is a **caution flag, not a filter**. We do not clip the values, we do
> not delete the rows, and we do not invent a censoring relation that the file does not state. We simply remember,
> when interpreting any result involving low-potency compounds, that the number is soft.

### Exercise 4.1 — explain a difference

Pick the descriptor with the largest effect size in the widget above (try **MW**, **HeavyAtoms**, **AromRings**).
Write two or three sentences explaining *why* the two sources differ on it. Then test your explanation: find the
descriptor on which they differ **least**, and say why that one is different.

Hint: think about where each set of molecules came from — published medicinal-chemistry series versus a commercial
diversity/screening library.
"""

# %%
# YOUR CODE HERE



# %% [markdown]
"""
<details><summary><b>Solution</b></summary>

```python
rows = []
for d in DESC_COLS:
    a, b = ch_struct[d].dropna(), oa_struct[d].dropna()
    pooled = np.sqrt((a.var() + b.var()) / 2)
    rows.append({"descriptor": d, "ChEMBL median": a.median(), "OpenADMET median": b.median(),
                 "Cohen's d": (b.mean() - a.mean()) / pooled})
effects = pd.DataFrame(rows).sort_values("Cohen's d")
print(effects.round(2).to_string(index=False))

# Largest differences: MW and HeavyAtoms (OpenADMET ~90 Da / 7 heavy atoms smaller), then AromRings and RotB.
#
# WHY. ChEMBL's CYP3A4 records come overwhelmingly from late-stage medicinal-chemistry papers: someone had a
# lead series, elaborated it, and measured CYP3A4 as a liability check. Elaborated leads are big, flat and
# lipophilic — that is what optimisation does to a molecule. OpenADMET screened an Enamine diversity set plus
# FDA-approved drugs: fragment-like and lead-like compounds, deliberately spread over chemical space, with no
# optimisation history. So the difference is a difference in the PURPOSE of the two collections.
#
# Smallest difference: HBD (both medians 1). Hydrogen-bond donors are tightly constrained in anything
# vendors sell as "drug-like" — both sets were filtered, explicitly or implicitly, against the same
# rule-of-five heuristics, so this descriptor cannot tell the collections apart.
#
# CAUTION: part of the gap may also be an ASSAY effect rather than a chemistry effect — big lipophilic
# compounds are exactly the ones that inhibit CYP3A4, so a literature source enriched in CYP3A4 *hits*
# is automatically enriched in big lipophilic molecules. Selection and biology are confounded here.
```
</details>
"""

# %% [markdown]
"""
## 5 · Similarity: how close is the nearest ChEMBL molecule? (60–80 min)

### Morgan fingerprints and the Tanimoto coefficient

A **Morgan fingerprint** (ECFP) looks at every atom, hashes its circular environment out to a given **radius**, and
sets the corresponding bits of a vector. We use the standard binary setting: **radius 2, 2048 bits**.

Similarity between two bit vectors is the **Tanimoto** (Jaccard) coefficient — the number of bits they share divided
by the number of bits either one has:
"""

# %% [markdown]
r"""
$$T(a,b) \;=\; \frac{|a \cap b|}{|a \cup b|} \;=\; \frac{c}{a + b - c}$$

It runs from 0 (nothing in common) to 1 (identical fingerprints). **Pick two molecules and see.**
"""

# %%
FPGEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def fp(smiles):
    m = Chem.MolFromSmiles(smiles)
    return None if m is None else FPGEN.GetFingerprint(m)

PAIRS = {
    "ketoconazole": KNOWN.loc[0, "smiles"],
    "ritonavir": KNOWN.loc[1, "smiles"],
    "midazolam": KNOWN.loc[2, "smiles"],
    "itraconazole": "CCC(C)n1ncn(-c2ccc(N3CCN(c4ccc(OCC5COC(Cn6cncn6)(c6ccc(Cl)cc6Cl)O5)cc4)CC3)cc2)c1=O",   # CHEMBL64391
    "erythromycin": "CC[C@H]1OC(=O)[C@H](C)[C@@H](O[C@H]2C[C@@](C)(OC)[C@@H](O)[C@H](C)O2)[C@H](C)[C@@H](O[C@@H]2O[C@H](C)C[C@H](N(C)C)[C@H]2O)[C@](C)(O)C[C@@H](C)C(=O)[C@H](C)[C@@H](O)[C@]1(C)O",   # CHEMBL532
    "caffeine": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
    "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
}
NAMES = list(PAIRS)

@interact(molecule_A=NAMES, molecule_B=NAMES)
def tanimoto_pair(molecule_A="ketoconazole", molecule_B="itraconazole"):
    a, b = PAIRS[molecule_A], PAIRS[molecule_B]
    t = DataStructs.TanimotoSimilarity(fp(a), fp(b))
    verdict = ("the same molecule" if t > .99 else
               "close analogues — a chemist would call them the same series" if t >= .6 else
               "related, but clearly different compounds" if t >= .4 else
               "unrelated by this measure")
    print(f"Tanimoto(ECFP4) = {t:.3f}   →   {verdict}")
    display(draw([a, b], legends=[molecule_A, molecule_B], per_row=2, size=(330, 250)))

# %% [markdown]
"""
> **A number to keep in your head.** For ECFP4 on drug-like molecules, **T ≈ 0.4 is already fairly low** and
> T ≥ 0.6–0.7 is what people usually mean by "an analogue". Two random drugs typically score around 0.1–0.2.
> These numbers depend on the fingerprint — the same pair scores differently with MACCS keys — so always say which
> fingerprint you used.

### The reference set, and the nearest-neighbour search

For every OpenADMET compound we now ask: **what is the most similar molecule ChEMBL has ever tested against CYP3A4?**

The reference set is the full ChEMBL structure table — every unique structure with any CYP3A4 IC50 record, censored
included. State its size whenever you report a coverage number: "nearest neighbour" means nothing without it.
"""

# %%
print(f"reference set: {len(ch_struct):,} unique ChEMBL structures")
print(f"query set    : {len(oa_struct):,} OpenADMET compounds")

t0 = time.time()
ch_fps = [FPGEN.GetFingerprint(Chem.MolFromSmiles(s)) for s in tqdm(ch_struct["smiles_std"], desc="ChEMBL fps")]
oa_fps = [FPGEN.GetFingerprint(Chem.MolFromSmiles(s)) for s in tqdm(oa_struct["smiles_std"], desc="OpenADMET fps")]
print(f"{len(ch_fps) + len(oa_fps):,} fingerprints in {time.time() - t0:.1f} s")

# %%
# Bulk Tanimoto: one query against the whole reference set at a time. ~25 million comparisons, a few seconds.
ch_index = {s: i for i, s in enumerate(ch_struct["smiles_std"])}

t0 = time.time()
max_sim = np.zeros(len(oa_fps))
nn_idx = np.zeros(len(oa_fps), dtype=int)
for k, (q, smi) in enumerate(zip(tqdm(oa_fps, desc="nearest neighbours"), oa_struct["smiles_std"])):
    sims = np.array(DataStructs.BulkTanimotoSimilarity(q, ch_fps))
    if smi in ch_index:
        sims[ch_index[smi]] = -1          # exclude the molecule's own identical twin
    j = int(sims.argmax())
    max_sim[k], nn_idx[k] = sims[j], j

oa_struct["max_tanimoto_to_chembl"] = max_sim
oa_struct["nn_chembl_id"] = ch_struct["molecule_chembl_id"].values[nn_idx]
oa_struct["nn_smiles"] = ch_struct["smiles_std"].values[nn_idx]
print(f"\n{len(oa_fps) * len(ch_fps) / 1e6:.0f} million comparisons in {time.time() - t0:.1f} s")

# %% [markdown]
"""
> **Exact overlap and nearest neighbours are different questions.** We answered the first in section 3: 52 molecules
> are literally present in both sets. Here we exclude those self-matches, so that `max_tanimoto_to_chembl` always
> means *"the closest ChEMBL molecule that is not this molecule"*.
"""

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 3.6))
axes[0].hist(max_sim, bins=60, color=SRC_COLOR["OpenADMET"])
axes[0].axvline(np.median(max_sim), color="k", ls="--",
                label=f"median {np.median(max_sim):.2f}")
axes[0].axvline(0.4, color="crimson", ls=":", label="T = 0.4")
axes[0].set_xlabel("max Tanimoto to the ChEMBL reference set"); axes[0].set_ylabel("OpenADMET compounds")
axes[0].legend(); axes[0].set_title("How close is the nearest ChEMBL molecule?")

thr = np.linspace(0, 1, 101)
axes[1].plot(thr, [(max_sim >= t).mean() for t in thr], lw=2, color=SRC_COLOR["OpenADMET"])
axes[1].set_xlabel("Tanimoto threshold T"); axes[1].set_ylabel("fraction with a neighbour ≥ T")
axes[1].set_title("Coverage curve"); axes[1].grid(True, alpha=.3)
plt.tight_layout(); plt.show()

print(pd.Series(max_sim).describe(percentiles=[.05, .25, .5, .75, .95]).round(3).to_string())

# %%
@interact(threshold=(0.2, 1.0, 0.05))
def coverage_at(threshold=0.4):
    covered = (max_sim >= threshold).sum()
    print(f"At Tanimoto ≥ {threshold:.2f}:")
    print(f"  {covered:5,} of {len(max_sim):,} OpenADMET compounds ({covered / len(max_sim):6.1%}) "
          f"have a ChEMBL neighbour this close")
    print(f"  {len(max_sim) - covered:5,} ({1 - covered / len(max_sim):6.1%}) do not — "
          f"ChEMBL has nothing like them")
    fig, ax = plt.subplots(figsize=(8, 1.1))
    ax.barh([0], [covered / len(max_sim)], color=SRC_COLOR["both"], label="covered")
    ax.barh([0], [1 - covered / len(max_sim)], left=[covered / len(max_sim)],
            color="#cccccc", label="not covered")
    ax.set_xlim(0, 1); ax.set_yticks([]); ax.legend(loc="center", ncol=2, fontsize=9)
    plt.tight_layout(); plt.show()

# %% [markdown]
"""
### Look at the pairs

A number is a summary; the molecules are the evidence. **Move the slider** through the OpenADMET compounds sorted
from the best-covered to the least-covered, and judge the pairs with your own eyes.
"""

# %%
ranked = oa_struct.sort_values("max_tanimoto_to_chembl", ascending=False).reset_index(drop=True)

@interact(rank=(0, len(ranked) - 1, 1))
def show_pair(rank=0):
    r = ranked.iloc[rank]
    print(f"rank {rank + 1} of {len(ranked)}   ·   max Tanimoto = {r['max_tanimoto_to_chembl']:.3f}")
    print(f"OpenADMET  {r['Molecule_Name']}   pIC50 = {r['pIC50']:.2f} "
          f"[{r['pIC50_conf_low']:.2f}, {r['pIC50_conf_high']:.2f}]")
    print(f"ChEMBL     {r['nn_chembl_id']}")
    display(draw([r["smiles_std"], r["nn_smiles"]],
                 legends=[f"{r['Molecule_Name']} (OpenADMET)", f"{r['nn_chembl_id']} (ChEMBL)"],
                 per_row=2, size=(340, 260)))

# %%
# The six best-covered and the six worst-covered, side by side.
for label, subset in [("BEST covered (closest ChEMBL neighbour)", ranked.head(6)),
                      ("WORST covered (ChEMBL has nothing like them)", ranked.tail(6))]:
    print(f"\n{label}")
    display(draw(subset["smiles_std"],
                 legends=[f"{n}  T={t:.2f}" for n, t in
                          zip(subset["Molecule_Name"], subset["max_tanimoto_to_chembl"])],
                 per_row=6, size=(200, 160)))

# %% [markdown]
"""
### T = 1.000, and yet not the same molecule

Look at the far right of the histogram. A handful of compounds have a ChEMBL neighbour at **Tanimoto 1.000** even
though we excluded their own identical twin. How?

Because **Morgan fingerprints do not encode stereochemistry** by default. Two molecules that differ only in
configuration produce *exactly the same* bit vector. This is the concrete reason we defined overlap in section 3 on
the standardised **isomeric** SMILES and not on "fingerprint similarity equals 1".

Here is every such case in this dataset. Three molecules, three *different* lessons.
"""

# %%
twins = oa_struct[oa_struct["max_tanimoto_to_chembl"] > 0.999]
print(f"{len(twins)} OpenADMET compounds have a NON-identical ChEMBL molecule at Tanimoto 1.000\n")
for _, r in twins.iterrows():
    ch_name = ch_struct.loc[ch_struct["smiles_std"] == r["nn_smiles"], "molecule_pref_name"].iloc[0]
    mo, mc = Chem.MolFromSmiles(r["smiles_std"]), Chem.MolFromSmiles(r["nn_smiles"])
    flat_same = Chem.MolToSmiles(mo, isomericSmiles=False) == Chem.MolToSmiles(mc, isomericSmiles=False)
    print(f"{r['Molecule_Name']}  ←→  {r['nn_chembl_id']} ({ch_name})")
    print(f"   InChIKey  OpenADMET {Chem.MolToInchiKey(mo)}")
    print(f"             ChEMBL    {Chem.MolToInchiKey(mc)}")
    print(f"   identical once stereochemistry is stripped: {flat_same}\n")

display(draw(list(twins["smiles_std"]) + list(twins["nn_smiles"]),
             legends=[f"{n} (OpenADMET)" for n in twins["Molecule_Name"]] +
                     [f"{c} (ChEMBL)" for c in twins["nn_chembl_id"]],
             per_row=len(twins), size=(260, 210)))

# %% [markdown]
"""
Read the InChIKeys: the **first block is the connectivity**, the **second block encodes the stereochemistry**. In
all three pairs the first block is identical and the second differs — same skeleton, different three-dimensional
molecule. And the three cases are not the same kind of problem at all:

| pair | what differs | is it the same compound? |
|---|---|---|
| **curcumin** (`CHEMBL140`) | OpenADMET omits the C=C geometry; ChEMBL specifies (*E,E*) | **Yes.** One source simply recorded less. Our strict test *under*-counts the overlap here. |
| **betamethasone / dexamethasone** (`CHEMBL632` / `CHEMBL384467`) | one stereocentre: the 16α vs 16β methyl | **No — these are two different drugs.** Both are marketed corticosteroids. The fingerprint cannot tell them apart. |
| **(S)-ibuprofen / ibuprofen** (`CHEMBL175` / `CHEMBL521`) | OpenADMET has the single (*S*) enantiomer (dexibuprofen); ChEMBL's entry is unspecified (racemic) | **It depends what you are asking.** Same substance, different specification. |

> **The lesson, in one sentence:** a fingerprint similarity of 1.0 is **not** an identity test, and identity is not
> always a yes/no question. Dexamethasone and betamethasone scoring 1.000 is exactly the failure mode that would
> have corrupted our overlap count had we used similarity instead of structure identity — and curcumin is the
> reminder that the strict test errs in the other direction. We report the strict number (52) and we show you both
> failure modes, rather than pretending a single number settles it.

### Exercise 5.1 — read a pair

Use the slider to find a pair with Tanimoto between **0.45 and 0.60** (in this dataset that band sits at roughly
**ranks 45–130** — 91 compounds fall in it).

1. What do the two molecules share, and what differs?
2. Would you expect them to have a similar CYP3A4 IC50? Look up the ChEMBL neighbour's measured value
   (`ch_quant[ch_quant.smiles_std == r.nn_smiles]`) and compare it with the OpenADMET pIC50.
3. The two numbers come from different assays. Does that change what you are willing to conclude?
"""

# %%
# YOUR CODE HERE



# %% [markdown]
"""
<details><summary><b>Solution</b></summary>

```python
band = ranked[ranked["max_tanimoto_to_chembl"].between(0.45, 0.60)]
# Only about a quarter of ChEMBL neighbours have an exact curated value; part 2 needs one of those.
comparable = band[band["nn_smiles"].isin(set(ch_quant["smiles_std"]))]
print(f"{len(band)} pairs in the 0.45–0.60 band, {len(comparable)} of them with a neighbour that has an "
      f"exact ChEMBL pIC50")
r = comparable.iloc[0]

neighbour_acts = ch_quant[ch_quant["smiles_std"] == r["nn_smiles"]]
print(f"OpenADMET {r['Molecule_Name']}: pIC50 = {r['pIC50']:.2f} "
      f"[{r['pIC50_conf_low']:.2f}, {r['pIC50_conf_high']:.2f}]")
if len(neighbour_acts):
    print(f"ChEMBL {r['nn_chembl_id']}: {len(neighbour_acts)} exact record(s), "
          f"median pIC50 = {neighbour_acts['pIC50'].median():.2f}")
    print("  assays:", neighbour_acts["assay_chembl_id"].tolist())
    print("  probes:", neighbour_acts["probe_substrate"].tolist())
else:
    print(f"{r['nn_chembl_id']} has only censored records — no exact value to compare")
display(draw([r["smiles_std"], r["nn_smiles"]], legends=["OpenADMET", "ChEMBL"], per_row=2, size=(340, 260)))

# 1. At T ≈ 0.5 the two molecules usually share a core ring system but differ in a substituent, a linker
#    or a ring fusion — enough for a chemist to see the family resemblance, not enough to call them analogues.
# 2. The similarity principle says "similar structures tend to have similar properties" — TEND TO.
#    T = 0.5 is well inside the region where it fails routinely; a single substituent can move a CYP3A4
#    IC50 by two log units (an activity cliff).
# 3. Yes, and decisively. Even if the two numbers agreed perfectly, the agreement would be partly luck,
#    because the assays differ. And if they disagree by a log unit, we cannot tell whether that is the
#    chemistry or the assay. The honest statement is about STRUCTURAL coverage only:
#    "ChEMBL contains a compound of this family" — not "we therefore know this compound's IC50".
```
</details>
"""


# %% [markdown]
"""
## 6 · Scaffolds: the skeletons underneath (80–95 min)

A **Bemis–Murcko scaffold** is what is left when you strip every side chain and keep the ring systems plus the
linkers between them. It is the coarsest useful description of "what kind of molecule is this", and it is how
chemists talk about series.
"""

# %%
def murcko(smiles):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m))

for frame in (ch_struct, oa_struct):
    frame["scaffold"] = [murcko(s) for s in tqdm(frame["smiles_std"], desc="scaffolds")]
    frame["has_ring"] = frame["scaffold"].fillna("").ne("")

# %% [markdown]
"""
### Acyclic molecules first — they have no scaffold

A molecule with no ring gives an **empty** Murcko scaffold. That is not a missing value to be imputed or an error to
be dropped silently; it is a real category. We count them, set them aside, and say so.
"""

# %%
acyclic = pd.DataFrame({
    "acyclic (no scaffold)": [int((~ch_struct["has_ring"]).sum()), int((~oa_struct["has_ring"]).sum())],
    "with a scaffold": [int(ch_struct["has_ring"].sum()), int(oa_struct["has_ring"].sum())],
}, index=["ChEMBL", "OpenADMET"])
display(acyclic)

ch_ring = ch_struct[ch_struct["has_ring"]]
oa_ring = oa_struct[oa_struct["has_ring"]]
S_CH, S_OA = set(ch_ring["scaffold"]), set(oa_ring["scaffold"])

# %%
stats = pd.DataFrame({
    "molecules with a ring": [len(ch_ring), len(oa_ring)],
    "distinct scaffolds": [len(S_CH), len(S_OA)],
    "scaffolds per molecule": [len(S_CH) / len(ch_ring), len(S_OA) / len(oa_ring)],
    "singleton scaffolds": [int((ch_ring["scaffold"].value_counts() == 1).sum()),
                            int((oa_ring["scaffold"].value_counts() == 1).sum())],
}, index=["ChEMBL", "OpenADMET"])
stats["singletons %"] = (stats["singleton scaffolds"] / stats["distinct scaffolds"] * 100).round(1)
display(stats.round(3))

shared_scaffolds = S_CH & S_OA
novel = ~oa_ring["scaffold"].isin(S_CH)
print(f"\nscaffolds shared by the two sources : {len(shared_scaffolds):,}")
print(f"OpenADMET scaffolds absent from ChEMBL: {len(S_OA - S_CH):,} of {len(S_OA):,} "
      f"({len(S_OA - S_CH) / len(S_OA):.1%} of scaffolds)")
print(f"OpenADMET MOLECULES whose scaffold ChEMBL has never seen: {int(novel.sum()):,} of {len(oa_ring):,} "
      f"({novel.mean():.1%})")

# %% [markdown]
"""
### Compare diversity fairly: matched sample sizes

"Scaffolds per molecule" is **not** comparable between sets of different sizes — the ratio mechanically rises as a
set gets smaller, because large sets have more chances to repeat a scaffold. To compare diversity honestly, draw a
ChEMBL sample of exactly the OpenADMET size, with a fixed seed, and compare like with like. Report both numbers.
"""

# %%
n_match = len(oa_ring)
reps = [ch_ring.sample(n_match, random_state=RANDOM_SEED + i)["scaffold"].nunique() for i in range(20)]
print(f"full sets          : ChEMBL {len(S_CH):,} scaffolds / {len(ch_ring):,} molecules"
      f"   ·   OpenADMET {len(S_OA):,} / {len(oa_ring):,}")
print(f"matched size ({n_match:,}): ChEMBL {np.mean(reps):.0f} ± {np.std(reps):.0f} scaffolds (20 draws)"
      f"   ·   OpenADMET {len(S_OA):,}")
print(f"\n→ at equal sample size OpenADMET is the MORE scaffold-diverse set "
      f"({len(S_OA) / np.mean(reps):.2f}× as many distinct scaffolds)")

# %% [markdown]
"""
### The most common scaffolds in each source

**Use the slider** to walk down the ranking, and watch what each source is made of.
"""

# %%
top_ch = ch_ring["scaffold"].value_counts()
top_oa = oa_ring["scaffold"].value_counts()

@interact(source=["OpenADMET", "ChEMBL"], start_rank=(0, 40, 4))
def browse_scaffolds(source="OpenADMET", start_rank=0):
    counts = top_oa if source == "OpenADMET" else top_ch
    chunk = counts.iloc[start_rank:start_rank + 8]
    other = S_CH if source == "OpenADMET" else S_OA
    legends = [f"#{start_rank + i + 1}: {c} cpds{'' if s in other else '  ★ unique to ' + source}"
               for i, (s, c) in enumerate(chunk.items())]
    print(f"{source}: scaffolds ranked {start_rank + 1}–{start_rank + len(chunk)} "
          f"of {len(counts):,}   (★ = not present in the other source)")
    display(draw(list(chunk.index), legends=legends, per_row=4, size=(230, 175)))

# %% [markdown]
"""
The contrast is stark, once you look past rank 1. In **both** sources the busiest scaffold is a trivial one — plain
benzene, `c1ccccc1`, which every monocyclic aromatic compound collapses to (133 compounds in ChEMBL, 20 in
OpenADMET). That is an artefact of how Murcko scaffolds work, not a finding.

Ranks 2 onwards are the finding. ChEMBL's are **elaborated, decorated multi-ring systems** carrying 36–84 compounds
each — the signature of published analogue series, where a single paper contributes dozens of close relatives.
OpenADMET's busiest scaffold is a plain benzenesulfonanilide with **27** compounds, and nothing else reaches 10.
**95 %** of OpenADMET's scaffolds appear exactly once, against **75 %** of ChEMBL's. That is the difference between
a diversity library and an optimisation history.

### The scaffolds the two sources share
"""

# %%
shared_counts = pd.DataFrame({
    "in ChEMBL": [top_ch.get(s, 0) for s in shared_scaffolds],
    "in OpenADMET": [top_oa.get(s, 0) for s in shared_scaffolds],
}, index=list(shared_scaffolds)).sort_values("in OpenADMET", ascending=False)
print(f"{len(shared_counts)} shared scaffolds; the 8 most populated on the OpenADMET side:")
display(shared_counts.head(8))
draw(list(shared_counts.head(8).index),
     legends=[f"ChEMBL {a} / OpenADMET {b}" for a, b in
              zip(shared_counts["in ChEMBL"].head(8), shared_counts["in OpenADMET"].head(8))],
     per_row=4, size=(230, 175))

# %% [markdown]
"""
### Browse the compounds behind a scaffold

`mols2grid` gives a searchable, sortable grid. Pick a shared scaffold and compare how each source decorated it.
"""

# %%
# Build the table first, outside the try, so the fallback below always has something to draw.
probe_scaffold = shared_counts.index[0]
members = pd.concat([
    ch_ring[ch_ring["scaffold"] == probe_scaffold].assign(
        label=lambda d: d["molecule_chembl_id"], src="ChEMBL")[["smiles_std", "label", "src", "MW", "cLogP"]],
    oa_ring[oa_ring["scaffold"] == probe_scaffold].assign(
        label=lambda d: d["Molecule_Name"], src="OpenADMET")[["smiles_std", "label", "src", "MW", "cLogP"]],
], ignore_index=True).round(1)
print(f"scaffold {probe_scaffold} — {len(members)} molecules "
      f"({(members['src'] == 'ChEMBL').sum()} ChEMBL, {(members['src'] == 'OpenADMET').sum()} OpenADMET)")

try:
    import mols2grid
    display(mols2grid.display(members, smiles_col="smiles_std",
                              subset=["img", "label", "src"], tooltip=["MW", "cLogP"],
                              size=(180, 140), n_items_per_page=12))
except Exception as e:
    print("mols2grid unavailable, drawing a static grid instead:", type(e).__name__, str(e)[:100])
    display(draw(members["smiles_std"].head(12),
                 legends=[f"{l} ({s})" for l, s in zip(members["label"].head(12), members["src"].head(12))]))

# %% [markdown]
"""
### Exercise 6.1 — find a scaffold ChEMBL has never seen

1. List the OpenADMET scaffolds that are **absent** from ChEMBL and carry at least 3 compounds.
2. Draw the top few. What ring systems are they?
3. Pick one and check the *max Tanimoto* of its member compounds. Is "novel scaffold" the same as "no similar
   molecule in ChEMBL"? Explain the difference.
"""

# %%
# YOUR CODE HERE



# %% [markdown]
"""
<details><summary><b>Solution</b></summary>

```python
novel_scaffolds = top_oa[~top_oa.index.isin(S_CH)]
multi = novel_scaffolds[novel_scaffolds >= 3]
print(f"{len(novel_scaffolds)} OpenADMET scaffolds absent from ChEMBL; {len(multi)} of them carry >= 3 compounds")
display(draw(list(multi.head(8).index),
             legends=[f"{c} compounds" for c in multi.head(8)], per_row=4, size=(230, 175)))

s = multi.index[0]
members = oa_ring[oa_ring["scaffold"] == s]
print(f"\nscaffold {s}: {len(members)} compounds, "
      f"max Tanimoto to ChEMBL = {members['max_tanimoto_to_chembl'].min():.2f}–"
      f"{members['max_tanimoto_to_chembl'].max():.2f} (median {members['max_tanimoto_to_chembl'].median():.2f})")

# They are NOT the same thing, and the numbers show it:
#
# A scaffold is an EXACT, all-or-nothing match on the ring skeleton. Add one ring, change a five-membered
# ring to a six-membered one, and the Murcko scaffold string changes completely — the match is lost even
# though a chemist would see the same family. So "novel scaffold" is a very strict criterion and 93.9 % of
# OpenADMET compounds meet it.
#
# Tanimoto is GRADED. A compound whose exact skeleton is absent from ChEMBL can still share most of its
# substructures with ChEMBL molecules, and so score T = 0.5. Typically these compounds have novel scaffolds
# but middling similarity.
#
# The compounds that deserve attention are the ones that are novel on BOTH counts: unseen scaffold AND low
# max Tanimoto. Those are the genuine blind spots — section 8 goes looking for them.
```
</details>
"""


# %% [markdown]
"""
## 7 · Chemical space: PCA and UMAP (95–110 min)

Our molecules live in a 9-dimensional descriptor space and a 2048-dimensional fingerprint space. To *look* at them we
project to two dimensions.

### The sample, and the rules we follow

Interactive plots of 13 500 points are slow and unreadable, so we sample — but **how** you sample decides what you
are allowed to conclude:

- at most **1 000 unique structures per source**, drawn with a fixed seed (42) so the figure is reproducible;
- the union is **deduplicated**, and molecules present in both sources keep the membership label **`both`**;
- the sample is **not** enriched by activity, by novelty, or by anything else. Enriching it would guarantee the
  conclusion we are trying to test.
"""

# %%
N_SAMPLE = 1000
ch_samp = ch_struct.sample(min(N_SAMPLE, len(ch_struct)), random_state=RANDOM_SEED)
oa_samp = oa_struct.sample(min(N_SAMPLE, len(oa_struct)), random_state=RANDOM_SEED)

sample = pd.concat([
    ch_samp[["smiles_std", "scaffold"] + DESC_COLS].assign(
        source="ChEMBL", label=ch_samp["molecule_chembl_id"], activity=np.nan),
    oa_samp[["smiles_std", "scaffold"] + DESC_COLS].assign(
        source="OpenADMET", label=oa_samp["Molecule_Name"], activity=oa_samp["pIC50"]),
], ignore_index=True).drop_duplicates("smiles_std").reset_index(drop=True)
sample["membership"] = np.where(sample["smiles_std"].isin(BOTH), "both", sample["source"])

print(f"{len(sample):,} unique structures in the joint sample")
print(sample["membership"].value_counts().to_dict())

# %% [markdown]
"""
### PCA on the descriptors

PCA is linear: each axis is an interpretable weighted sum of the inputs, and the explained variance tells you how
much of the spread you are actually seeing. The descriptors must be **standardised** first (MW ≈ 400, HBD ≈ 2 —
without scaling, MW would be the only thing PCA sees).
"""

# %%
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

X_desc = StandardScaler().fit_transform(sample[DESC_COLS])
pca = PCA(n_components=2).fit(X_desc)
pcs = pca.transform(X_desc)
sample["PC1"], sample["PC2"] = pcs[:, 0], pcs[:, 1]

evr = pca.explained_variance_ratio_
print(f"explained variance: PC1 {evr[0]:.1%}, PC2 {evr[1]:.1%}, together {evr.sum():.1%}")
loadings = pd.DataFrame(pca.components_.T, index=DESC_COLS, columns=["PC1", "PC2"]).round(2)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw={"width_ratios": [2, 1]})
for src, grp in sample.groupby("membership"):
    axes[0].scatter(grp["PC1"], grp["PC2"], s=9, alpha=.55, c=SRC_COLOR[src], label=f"{src} ({len(grp)})")
axes[0].set_xlabel(f"PC1 ({evr[0]:.0%})"); axes[0].set_ylabel(f"PC2 ({evr[1]:.0%})")
axes[0].legend(markerscale=2); axes[0].set_title("PCA of 9 descriptors")
loadings.plot.barh(ax=axes[1]); axes[1].set_title("loadings"); axes[1].axvline(0, c="k", lw=.8)
plt.tight_layout(); plt.show()
display(loadings)

# %% [markdown]
"""
Read the loadings: **PC1 is size** (MW, HeavyAtoms, HBA, TPSA, RotB all load positively together) and **PC2 is
lipophilicity versus saturation** (cLogP and AromRings against FracCSP3, HBD). So the left–right axis of that plot
is "small → big" and the up–down axis is "greasy and flat → polar and three-dimensional". The OpenADMET cloud sits
noticeably to the left: smaller molecules, as the descriptor histograms already told us.

### UMAP on the fingerprints

UMAP is non-linear and works on the **binary fingerprints** with the **Jaccard** (Tanimoto) distance — the same
notion of similarity we used in section 5, so this map is a picture of the same thing.

**One map, fitted once, on the union of both sources.** Fitting two separate maps and putting them side by side
would be meaningless: UMAP axes have no intrinsic meaning, and two independent fits are not comparable.
"""

# %%
import umap

X_fp = np.array([FPGEN.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in sample["smiles_std"]], dtype=bool)
t0 = time.time()
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="jaccard", random_state=RANDOM_SEED)
emb = reducer.fit_transform(X_fp)
sample["UMAP1"], sample["UMAP2"] = emb[:, 0], emb[:, 1]
print(f"UMAP on {X_fp.shape[0]:,} × {X_fp.shape[1]} fingerprints in {time.time() - t0:.0f} s")

# %%
top_shared = list(shared_counts.head(5).index)
sample["highlight_scaffold"] = np.where(sample["scaffold"].isin(top_shared),
                                        sample["scaffold"].str.slice(0, 28), "other")

@interact(colour_by=["source membership", "activity (OpenADMET only)", "shared scaffolds", "molecular weight"])
def umap_view(colour_by="source membership"):
    fig, ax = plt.subplots(figsize=(7.5, 6))
    if colour_by == "source membership":
        for src in ["ChEMBL", "OpenADMET", "both"]:
            g = sample[sample["membership"] == src]
            ax.scatter(g["UMAP1"], g["UMAP2"], s=11, alpha=.6, c=SRC_COLOR[src],
                       label=f"{src} ({len(g)})", zorder=3 if src == "both" else 2)
        ax.legend(markerscale=2)
    elif colour_by == "activity (OpenADMET only)":
        g0 = sample[sample["activity"].isna()]
        ax.scatter(g0["UMAP1"], g0["UMAP2"], s=9, c="#dddddd", label="ChEMBL (no comparable value)")
        g1 = sample[sample["activity"].notna()]
        sc = ax.scatter(g1["UMAP1"], g1["UMAP2"], s=13, c=g1["activity"], cmap="viridis")
        plt.colorbar(sc, ax=ax, label="OpenADMET pIC50"); ax.legend(markerscale=2, loc="lower left")
    elif colour_by == "shared scaffolds":
        g0 = sample[sample["highlight_scaffold"] == "other"]
        ax.scatter(g0["UMAP1"], g0["UMAP2"], s=8, c="#dddddd")
        for i, s in enumerate(sorted(set(sample["highlight_scaffold"]) - {"other"})):
            g = sample[sample["highlight_scaffold"] == s]
            ax.scatter(g["UMAP1"], g["UMAP2"], s=28, label=f"{s}… ({len(g)})",
                       c=plt.cm.tab10(i), edgecolor="k", linewidth=.3)
        ax.legend(fontsize=7, markerscale=1.2)
    else:
        sc = ax.scatter(sample["UMAP1"], sample["UMAP2"], s=11, c=sample["MW"], cmap="magma")
        plt.colorbar(sc, ax=ax, label="MW")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"Joint UMAP of ECFP4 (Jaccard) — coloured by {colour_by}")
    plt.tight_layout(); plt.show()

# %% [markdown]
"""
> ⚠️ **Do not over-read a UMAP — and this map is the perfect example of why.**
>
> Look at it honestly: the two colours are shifted relative to each other, but they *mix*. A reasonable person
> looking only at this picture would say "these two sets overlap quite a lot". Section 5 measured the same thing
> properly and found the opposite: a median nearest-neighbour Tanimoto of 0.33, and 87 % of OpenADMET compounds
> with no ChEMBL molecule within 0.4.
>
> Both statements come from the same fingerprints. The map is not lying — it is doing its job, which is to put
> every point *somewhere* in two dimensions, necessarily crushing 2048 dimensions' worth of distance to do it.
> Distances between clusters are not meaningful, the layout changes with `n_neighbors` and `min_dist`, and another
> `random_state` gives another picture.
>
> **Use a UMAP to notice things and to show things. Never to measure them.** When the picture and the
> measurement disagree, the measurement wins.

### The same map, interactive

Hover a point for its identity, its source and its activity. Use the plotly legend to switch sources on and off:
clicking "ChEMBL" off is the quickest way to see where OpenADMET goes that ChEMBL does not.
"""

# %%
plot_df = sample.copy()
plot_df["pIC50"] = plot_df["activity"].round(2)
fig = px.scatter(plot_df, x="UMAP1", y="UMAP2", color="membership",
                 color_discrete_map=SRC_COLOR, hover_name="label",
                 hover_data={"UMAP1": False, "UMAP2": False, "MW": ":.0f", "cLogP": ":.1f",
                             "pIC50": True, "membership": False},
                 title="Chemical space of the two sources — click the legend to isolate one",
                 width=820, height=620)
fig.update_traces(marker=dict(size=5, opacity=.75))
fig.update_xaxes(showticklabels=False); fig.update_yaxes(showticklabels=False)
fig.show()


# %% [markdown]
"""
## 8 · Outliers: unusual how? (105–115 min)

Two completely different questions get called "outlier detection", and keeping them apart is the point of this
section:

| question | method | what it measures |
|---|---|---|
| Does this molecule have **unusual physicochemical properties**? | Isolation Forest on the descriptors | position within its own set |
| Is this molecule **structurally new** relative to ChEMBL? | low max Tanimoto (section 5) | distance to a *reference* set |

A compound can be either, both, or neither. Peptide macrocycles are property outliers but completely familiar
chemistry; a small, perfectly drug-like molecule with an unusual heterocycle is a novelty outlier with ordinary
properties.

### Property outliers: Isolation Forest
"""

# %%
from sklearn.ensemble import IsolationForest

iso = IsolationForest(n_estimators=300, random_state=RANDOM_SEED, contamination="auto")
iso.fit(StandardScaler().fit_transform(sample[DESC_COLS]))
sample["outlier_score"] = -iso.score_samples(StandardScaler().fit_transform(sample[DESC_COLS]))

fig, axes = plt.subplots(1, 2, figsize=(13, 3.4))
for src, grp in sample.groupby("membership"):
    axes[0].hist(grp["outlier_score"], bins=40, alpha=.6, label=src, color=SRC_COLOR[src], density=True)
axes[0].set_xlabel("isolation-forest score (higher = more unusual)"); axes[0].legend()
axes[0].set_title("Property unusualness, by source")
sc = axes[1].scatter(sample["PC1"], sample["PC2"], c=sample["outlier_score"], cmap="inferno", s=11)
plt.colorbar(sc, ax=axes[1], label="outlier score")
axes[1].set_xlabel("PC1"); axes[1].set_ylabel("PC2"); axes[1].set_title("…and where it sits in PCA space")
plt.tight_layout(); plt.show()

# %% [markdown]
"""
### Novelty outliers: what ChEMBL has never seen

Now the other question, on the full OpenADMET set rather than the sample.
"""

# %%
oa_struct["novel_scaffold"] = ~oa_struct["scaffold"].isin(S_CH) & oa_struct["has_ring"]
LOW_T = 0.30
oa_struct["low_similarity"] = oa_struct["max_tanimoto_to_chembl"] < LOW_T
oa_struct["blind_spot"] = oa_struct["low_similarity"] & oa_struct["novel_scaffold"]

flags = pd.DataFrame({
    "compounds": [int(oa_struct["low_similarity"].sum()), int(oa_struct["novel_scaffold"].sum()),
                  int(oa_struct["blind_spot"].sum())],
}, index=[f"max Tanimoto < {LOW_T}", "scaffold absent from ChEMBL", "BOTH — genuine blind spots"])
flags["% of OpenADMET"] = (flags["compounds"] / len(oa_struct) * 100).round(1)
display(flags)

# %%
@interact(flag=["BOTH — genuine blind spots", "max Tanimoto < 0.30", "scaffold absent from ChEMBL",
                "property outliers (isolation forest)"],
          how_many=(4, 16, 4))
def inspect_flagged(flag="BOTH — genuine blind spots", how_many=8):
    if flag.startswith("property"):
        sub = sample.nlargest(how_many, "outlier_score")
        legends = [f"{l}  score {s:.2f}\n{m}" for l, s, m in
                   zip(sub["label"], sub["outlier_score"], sub["membership"])]
        print("Most unusual PROPERTIES within the joint sample — both sources, any membership.")
    else:
        col = {"BOTH — genuine blind spots": "blind_spot",
               "max Tanimoto < 0.30": "low_similarity",
               "scaffold absent from ChEMBL": "novel_scaffold"}[flag]
        sub = oa_struct[oa_struct[col]].nsmallest(how_many, "max_tanimoto_to_chembl")
        legends = [f"{n}  T={t:.2f}\npIC50 {p:.2f}" for n, t, p in
                   zip(sub["Molecule_Name"], sub["max_tanimoto_to_chembl"], sub["pIC50"])]
        print(f"{flag}: showing the {how_many} with the lowest similarity to ChEMBL "
              f"(of {int(oa_struct[col].sum()):,} flagged).")
    display(draw(sub["smiles_std"], legends=legends, per_row=4, size=(235, 195)))

# %%
# The inspection table: never judge a flagged compound from its score alone.
inspect = (oa_struct[oa_struct["blind_spot"]]
           .nsmallest(10, "max_tanimoto_to_chembl")
           [["Molecule_Name", "max_tanimoto_to_chembl", "pIC50", "pIC50_ci_width",
             "weak_potency_flag", "MW", "cLogP", "TPSA", "HeavyAtoms"]]
           .round(2))
display(inspect)

# %%
# Where do the blind spots sit on the joint map?
sample["is_blind_spot"] = sample["smiles_std"].isin(
    set(oa_struct.loc[oa_struct["blind_spot"], "smiles_std"]))
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, (xc, yc, title) in zip(axes, [("PC1", "PC2", "PCA (descriptors)"),
                                      ("UMAP1", "UMAP2", "UMAP (fingerprints)")]):
    ax.scatter(sample[xc], sample[yc], s=9, c="#dddddd", label="rest of the sample")
    g = sample[sample["is_blind_spot"]]
    ax.scatter(g[xc], g[yc], s=34, c="crimson", edgecolor="k", linewidth=.3,
               label=f"blind spots ({len(g)} in this sample)")
    ax.set_title(title); ax.set_xticks([]); ax.set_yticks([]); ax.legend(fontsize=8)
plt.tight_layout(); plt.show()

# %% [markdown]
"""
Notice where the red points are: **scattered through the middle of the cloud**, not out at the edges. That is the
whole lesson of this section. These compounds are not strange molecules — they sit in perfectly ordinary
property space. They are strange only *relative to ChEMBL*, and nothing but a comparison with a reference set could
have revealed them.

> ⚠️ **Never detect outliers from the 2D embedding itself.** A point that looks isolated on a UMAP may be an
> artefact of the projection. Flag in the full space (descriptors, fingerprints), then *show* the flags on the
> embedding — which is the order we used here.
>
> ⚠️ **And never delete a flagged compound automatically.** Every one of these is a real molecule with a real
> measurement. "Flagged" means *look at it*, not *drop it*.

### Exercise 8.1 — defend a flagged compound

Take the top row of the inspection table, or any compound from the property-outlier view.

1. Why was it flagged? (Which descriptor is extreme, or which comparison failed?)
2. Draw it. Is it a plausible molecule a chemist would make or buy?
3. Give one reason this compound might be **more** valuable than an average one, not less.
"""

# %%
# YOUR CODE HERE



# %% [markdown]
"""
<details><summary><b>Solution</b></summary>

```python
top_prop = sample.nlargest(5, "outlier_score")[["label", "membership", "outlier_score"] + DESC_COLS]
display(top_prop.round(2))
display(draw(sample.nlargest(5, "outlier_score")["smiles_std"],
             legends=sample.nlargest(5, "outlier_score")["label"].tolist(), per_row=5, size=(220, 180)))

z = ((sample[DESC_COLS] - sample[DESC_COLS].mean()) / sample[DESC_COLS].std())
worst = sample["outlier_score"].idxmax()
print("z-scores of the single most unusual molecule:")
print(z.loc[worst].round(1).sort_values(key=abs, ascending=False).to_string())

# Which descriptors actually drive the top of the ranking? Ask, rather than assume:
top_idx = sample.nlargest(5, "outlier_score").index
drivers = z.loc[top_idx].abs().max().sort_values(ascending=False)
print("\nlargest |z| across the five most unusual molecules:")
print(drivers.round(1).to_string())

# 1. The ranking is driven by HBD, RotB and TPSA (|z| up to ~14 on this sample), with MW and
#    HeavyAtoms following at ~+9: macrolides (erythromycin-like), cyclic peptides, aminoglycosides,
#    cardiac glycosides. One OpenADMET outlier is a long-chain quaternary ammonium with cLogP near
#    14 — a surfactant. Compute the numbers rather than quoting them: they depend on the sample,
#    which depends on the seed.
# 2. Every one of them is a real, well-known compound class. Macrolides are antibiotics, glycosides are
#    drugs, the surfactant is a catalogue chemical. Nothing here is a parsing error or a corrupt structure.
#    That is the point: "statistically unusual" and "wrong" are different words.
# 3. Several reasons, and they are not subtle:
#    - Macrolides are among the strongest CYP3A4 inhibitors known (erythromycin, clarithromycin). The
#      compounds at the extreme of property space are exactly the ones with the interesting biology.
#    - A model trained only on the dense middle of chemical space will be asked, one day, to predict one of
#      these. Removing them does not make the model better; it makes the model's failure invisible.
#    - For our actual question — does ChEMBL cover OpenADMET? — an unusual compound that ChEMBL *does*
#      contain is evidence of coverage, and one it does not is precisely the gap we are measuring.
#
#    The legitimate reasons to drop a compound are specific and checkable: an unparsable structure, a
#    documented transcription error, an assay flagged as invalid. "High outlier score" is not one of them.
```
</details>
"""


# %% [markdown]
"""
## 9 · Putting it together: one compound at a time (115–118 min)

Everything we computed, for any OpenADMET compound you choose. The shortlist mixes the best-covered, the
least-covered, the most potent and the most property-unusual compounds, so that moving the slider shows you the
full range of cases.
"""

# %%
# `ranked` was built in section 5, BEFORE section 8 added `novel_scaffold` / `blind_spot` to
# oa_struct. Rebuild it here so every row of the shortlist carries the same columns - otherwise
# pd.concat fills them with NaN, and `if NaN` is True, which would label every one of these
# compounds "novel" regardless of the truth.
ranked = oa_struct.sort_values("max_tanimoto_to_chembl", ascending=False).reset_index(drop=True)

shortlist = pd.concat([
    ranked.head(6).assign(why="best covered by ChEMBL"),
    ranked.tail(6).assign(why="worst covered by ChEMBL"),
    oa_struct.nlargest(6, "pIC50").assign(why="most potent"),
    oa_struct.nsmallest(6, "pIC50").assign(why="weakest (and least precise)"),
    oa_struct[oa_struct["smiles_std"].isin(BOTH)].head(6).assign(why="also present in ChEMBL"),
    # potent AND unseen by ChEMBL: the most interesting cell of the whole table
    oa_struct[oa_struct["blind_spot"]].nlargest(6, "pIC50").assign(why="potent, yet a ChEMBL blind spot"),
]).drop_duplicates("smiles_std").reset_index(drop=True)
print(f"{len(shortlist)} compounds in the shortlist:",
      shortlist["why"].value_counts().to_dict())

@interact(i=(0, len(shortlist) - 1, 1))
def compound_card(i=0):
    r = shortlist.iloc[i]
    in_chembl = r["smiles_std"] in BOTH
    print(f"[{i + 1}/{len(shortlist)}]  {r['Molecule_Name']}   —   selected as: {r['why']}")
    print("─" * 96)
    print(f"  CYP3A4 pIC50 (OpenADMET)  {r['pIC50']:.2f}   95 % CI [{r['pIC50_conf_low']:.2f}, "
          f"{r['pIC50_conf_high']:.2f}]   width {r['pIC50_ci_width']:.2f}"
          f"{'   ⚠ weak potency — interpret with caution' if r['weak_potency_flag'] else ''}")
    print(f"  MW {r['MW']:.0f}   cLogP {r['cLogP']:.1f}   TPSA {r['TPSA']:.0f}   "
          f"HBD {r['HBD']:.0f}   HBA {r['HBA']:.0f}   heavy atoms {r['HeavyAtoms']:.0f}")
    print(f"  nearest ChEMBL molecule   {r['nn_chembl_id']}   Tanimoto {r['max_tanimoto_to_chembl']:.3f}")
    if not r["has_ring"]:
        scaffold_note = "acyclic — it has no Murcko scaffold at all"
    elif r["novel_scaffold"]:
        scaffold_note = "no  ← a skeleton ChEMBL has never tested"
    else:
        scaffold_note = "yes"
    print(f"  exact structure in ChEMBL : {'YES' if in_chembl else 'no'}"
          f"      scaffold known to ChEMBL: {scaffold_note}")
    if in_chembl:
        rec = ch_quant[ch_quant["smiles_std"] == r["smiles_std"]]
        if len(rec):
            print(f"  ChEMBL's own value        : median pIC50 {rec['pIC50'].median():.2f} "
                  f"over {len(rec)} record(s) — DIFFERENT ASSAYS, do not average with the above")
    display(draw([r["smiles_std"], r["nn_smiles"]],
                 legends=[f"{r['Molecule_Name']} (OpenADMET)", f"{r['nn_chembl_id']} (nearest in ChEMBL)"],
                 per_row=2, size=(340, 260)))

# %% [markdown]
"""
## 10 · The answers (118–120 min)

### "How much chemical space overlaps?"

**Very little.** Four independent measurements agree:
"""

# %%
answers = pd.DataFrame([
    ("identical structures (standardised isomeric SMILES)",
     f"{len(BOTH)} of {len(oa_struct):,}", f"{len(BOTH) / len(oa_struct):.1%}"),
    ("median max Tanimoto to the ChEMBL reference set",
     f"{np.median(max_sim):.2f}", "a low value for ECFP4"),
    (f"compounds with no ChEMBL neighbour at T ≥ 0.4",
     f"{int((max_sim < 0.4).sum()):,} of {len(oa_struct):,}", f"{(max_sim < 0.4).mean():.1%}"),
    ("ring compounds whose Murcko scaffold is absent from ChEMBL",
     f"{int(novel.sum()):,} of {len(oa_ring):,}", f"{novel.mean():.1%}"),
], columns=["measurement", "value", "share"])
display(answers)

print(f"\nreference set: {len(ch_struct):,} unique ChEMBL structures with a CYP3A4 IC50 record "
      f"(ChEMBL {PROVENANCE.get('chembl_release', '?')})")
print(f"query set    : {len(oa_struct):,} OpenADMET compounds with a CYP3A4 direct-inhibition pIC50")

# %% [markdown]
"""
ChEMBL has thirty years of CYP3A4 measurements and nearly five times as many distinct structures as the challenge
set — and still, **87 % of the challenge compounds have no ChEMBL molecule within Tanimoto 0.4**, and 94 % sit on a
ring skeleton ChEMBL has never tested against this enzyme.

The reason is now familiar: the two collections were assembled for different purposes. ChEMBL is deep and narrow —
large elaborated molecules, in dense analogue series, from papers about specific projects. OpenADMET is broad and
shallow — smaller, singleton-scaffold compounds from a diversity catalogue. **More data does not mean more
coverage.**

### "Which challenge compounds have weak ChEMBL coverage?"
"""

# %%
print(f"{int(oa_struct['blind_spot'].sum()):,} compounds are flagged on both criteria "
      f"(max Tanimoto < {LOW_T} AND a scaffold ChEMBL has never seen) — "
      f"{oa_struct['blind_spot'].mean():.1%} of the set.")
display(oa_struct[oa_struct["blind_spot"]].nsmallest(8, "max_tanimoto_to_chembl")
        [["Molecule_Name", "max_tanimoto_to_chembl", "pIC50", "pIC50_ci_width", "MW", "cLogP"]].round(2))

# %% [markdown]
"""
These are where a model trained on ChEMBL would be **extrapolating**, and where it should say so. This is the
applicability-domain question, and we have answered it without training a single model.

### "Which unusual compounds deserve inspection, and why?"

Two different lists, for two different reasons — and that distinction is the most transferable thing in this
notebook:

- **Property outliers** (isolation forest on the descriptors): macrolides, glycosides, peptides, a long-chain
  surfactant. Unusual *in themselves*. Worth inspecting because extreme properties strain assays and models alike —
  and because, for CYP3A4 specifically, large lipophilic molecules are the known potent inhibitors. Valid chemistry;
  do not delete.
- **Novelty outliers** (low max Tanimoto, unseen scaffold): ordinary-looking molecules that ChEMBL simply never
  tested. Unusual *relative to a reference*. Worth inspecting because they define the limits of what literature data
  can tell you.

And one caution that applies to every number above: **we compared structures across the two sources, never their
activity values.** Only 7 of the 52 shared molecules could even be compared, and one of those seven pairs differed
by more than a log unit — different probe substrates, different enzyme preparations, different laboratories, all
measuring something they each legitimately call "CYP3A4 IC50". Structural coverage is a question we can answer with
this data; "do the two sources agree on potency" is not.

---

## Where to go next

- **Model it.** The natural sequel is a QSAR model with a declared applicability domain — and this notebook has
  already told you which test compounds fall outside it. See session 05 of the
  [chemoinformatics tutorials](https://github.com/AxelRolov/chemoinformatics_tutorials).
- **Enter the challenge.** The OpenADMET CYP challenge runs to **3 November 2026**:
  [challenge space](https://huggingface.co/spaces/openadmet/cyp-challenge).
- **Go deeper into the data.** The `tdi`, `single_concentration` and `emax` configurations of the same Hugging Face
  dataset hold time-dependent inhibition, primary-screening and Emax data — deliberately left out of this workshop.

## Further reading

- Zdrazil *et al.*, *The ChEMBL Database in 2023*, **Nucleic Acids Res.** 2024, 52, D1180.
- Landrum & Riniker, *Combining IC50 or Ki values from different sources is a source of significant noise*,
  **J. Chem. Inf. Model.** 2024, 64, 1560. — the paper behind section 3's scatter plot.
- Bemis & Murcko, *The properties of known drugs. 1. Molecular frameworks*, **J. Med. Chem.** 1996, 39, 2887.
- Rogers & Hahn, *Extended-connectivity fingerprints*, **J. Chem. Inf. Model.** 2010, 50, 742.
- Bento *et al.*, *An open source chemical structure curation pipeline using RDKit*, **J. Cheminform.** 2020, 12, 51.
- McInnes, Healy & Melville, *UMAP: Uniform Manifold Approximation and Projection*, **arXiv:1802.03426**, 2018.
- Walters, *Practical Cheminformatics* blog: <https://practicalcheminformatics.blogspot.com>.
"""
