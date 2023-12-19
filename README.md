# Mutants in PCM

This repository contains the open-source package `mutants_in_pcm` for the analysis of the effect of mutant data in 
bioactivity 
modelling, particularly in proteochemometric modelling (PCM). The package is written in Python 3.7 and facilitates 
the analysis of variants reported in the [ChEMBL database](https://www.ebi.ac.uk/chembl/) and the [Papyrus modelling 
dataset](https://zenodo.org/records/7821773). The full analysis is described in the paper **"A comprehensive analysis 
of the mutant landscape of bioactivity databases and its effect on modeling"** by M. Gorostiola González, 
O. Béquignon et al.

The repository contains an `analysis` folder with the main script and Jupyter notebooks used for the analysis in the 
paper. These notebooks can be directly used to quickly and easily extend the analysis to any protein of interest to the 
user. 

## Installation
The package can be installed using `pip`:
```bash
pip install git+https://github.com/CDDLeiden/mutants-in-pcm.git@master
```

## Getting started
The package contains three categories of modules:
1. Annotation: modules for the download and annotation of variants in the ChEMBL database and the Papyrus modelling 
   dataset. Includes `preprocessing.py`, `annotation.py`, and `annotation_check.py`.
2. Mutant analysis: modules for the analysis of the effect of variants in bioactivity at different levels. Includes 
   `mutant_analysis_family.py`, `mutant_analysis_accession.py`, `mutant_analysis_organism.py`, `mutant_analysis_type.py`,
    `mutant_analysis_protein.py`, `mutant_analysis_compounds.py`, `mutant_analysis_common_subsets.py`, 
   `mutant_analysis_distributions.py`, and `mutant_analysis_clustermaps.py`.
3. Modelling: modules for the modelling of bioactivity data, including the modelling of variants, and analysis of 
   the effect of variants. Includes `descriptors.py` and `modelling.py`.


## Data availability

