# Credits and attribution

This workshop adapts open teaching material that others generously published. The credit for the substance
belongs to the authors listed below. If you reuse the workshop, keep these attributions and cite the original
authors.

## The course it is built on

**Chemoinformatics tutorials**, <https://github.com/AxelRolov/chemoinformatics_tutorials> (CC BY 4.0 compilation;
MIT code). The workshop reuses and adapts these sessions:

- `02_molecular_representations`: Morgan fingerprints with `rdFingerprintGenerator`, Tanimoto similarity and
  bulk nearest-neighbour search;
- `03_chemical_databases`: ChEMBL retrieval through its REST API, IC50 to pIC50 conversion and loading
  OpenADMET data from the Hugging Face Hub;
- `04_exploratory_data_analysis`: the RDKit standardisation pipeline (cleanup, largest fragment, neutralisation),
  descriptors, Bemis–Murcko scaffolds, PCA and UMAP with the Jaccard metric.

These sessions themselves adapt the following sources, which this workshop therefore also draws on:

- **TeachOpenCADD**, Volkamer lab, <https://github.com/volkamerlab/teachopencadd>, CC BY 4.0. Talktorials
  T001 (compound data acquisition from ChEMBL), T011 (querying online API web services), T033 (molecular
  representations) and T005 (compound clustering). Sydow *et al.*, J. Cheminform. **2019**, 11, 29;
  Nucleic Acids Res. **2022**, 50, W753.
- **Practical Cheminformatics Tutorials**, Patrick Walters,
  <https://github.com/PatWalters/practical_cheminformatics_tutorials>, MIT. `ChEMBL_data_curation`,
  `find_scaffolds` and `visualizing_chemical_space`, together with his blog *Practical Cheminformatics*.
- **AI for Chemistry (EPFL CH-457)**, Schwaller group, <https://github.com/schwallergroup/ai4chem_course>, MIT.
  The unsupervised-learning notebooks on dimensionality reduction.
- **MolSSI cheminformatics workshop**, Jessica A. Nash, <https://github.com/MolSSI-Education/molssi-cheminformatics>,
  MIT. Molecular similarity.
- **RDKit** documentation (`rdMolStandardize`) and the **ChEMBL structure pipeline**
  (Bento *et al.*, J. Cheminform. **2020**, 12, 51).

## Data

| data | source | licence |
|---|---|---|
| CYP3A4 IC50 activities, assays and target record (`data/chembl/`) | [ChEMBL](https://www.ebi.ac.uk/chembl/) release 37, EMBL-EBI. Zdrazil *et al.*, Nucleic Acids Res. **2024**, 52, D1180 | CC BY-SA 3.0 |
| CYP inhibition training and blinded-test sets (`data/openadmet/`) | [OpenADMET CYP inhibition blind challenge](https://huggingface.co/datasets/openadmet/cyp-challenge-train-test), data generated at Octant for OpenADMET. [Challenge announcement](https://openadmet.ghost.io/announcing-openadmets-cyp-inhibition-blind-challenge/) | Apache-2.0 (dataset card) |
| reference-drug SMILES used in the demonstrations | [PubChem](https://pubchem.ncbi.nlm.nih.gov) | public domain |

The cached ChEMBL files are redistributed unchanged, apart from compression, under CC BY-SA 3.0. Material derived
from them (curated tables in `outputs/`) is shared under the same licence.

## Software

| project | used for | licence |
|---|---|---|
| [marimo](https://marimo.io) and [molab](https://molab.marimo.io) | reactive notebook and cloud runtime | Apache-2.0 |
| [RDKit](https://www.rdkit.org) | parsing, standardisation, descriptors, fingerprints, scaffolds, drawing | BSD-3 |
| [pandas](https://pandas.pydata.org), [NumPy](https://numpy.org) | tables and arrays | BSD-3 |
| [scikit-learn](https://scikit-learn.org) | PCA, standard scaling, Isolation Forest | BSD-3 |
| [UMAP](https://github.com/lmcinnes/umap) | non-linear chemical-space map. McInnes, Healy & Melville, arXiv:1802.03426 | BSD-3 |
| [Vega-Altair](https://altair-viz.github.io) | interactive charts | BSD-3 |
| [Requests](https://requests.readthedocs.io) | ChEMBL API and file downloads | Apache-2.0 |

## Methods referenced in the notebook

- Bemis & Murcko, *The properties of known drugs. 1. Molecular frameworks*, J. Med. Chem. **1996**, 39, 2887.
- Rogers & Hahn, *Extended-connectivity fingerprints*, J. Chem. Inf. Model. **2010**, 50, 742.
- Liu, Ting & Zhou, *Isolation Forest*, ICDM **2008**.
- Landrum & Riniker, *Combining IC50 or Ki values from different sources is a source of significant noise*,
  J. Chem. Inf. Model. **2024**, 64, 1560.

## What this workshop adds

Everything of substance above comes from the projects listed. This repository adds:

- the CYP3A4 framing that matches ChEMBL with OpenADMET;
- two interchangeable ChEMBL acquisition routes that share one normalisation step, with a record-level check that
  the routes agree;
- the assay-context review that separates direct from time-dependent inhibition;
- the cross-source analyses of similarity, scaffold novelty, joint maps and the two kinds of outliers;
- the marimo interactivity, the exercises and the instructor guide.

The workshop was assembled with **Claude** (Anthropic) and reviewed by the instructor. Any errors are ours, not the
original authors'.
