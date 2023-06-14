# -*- coding: utf-8 -*-


"""Bioactivity modelling."""

import json
from collections import Counter
from operator import itemgetter

import papyrus_scripts
import pandas as pd
import seaborn as sns
import os
import re
import matplotlib.pyplot as plt

import ml2json
from rdkit import Chem
from rdkit.Chem import Draw
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, GroupKFold
from papyrus_scripts.modelling import crossvalidate_model
from prodec import ProteinDescriptors, Transform, TransformType
from Mold2_pywrapper import Mold2

from preprocessing import merge_chembl_papyrus_mutants
from mutant_analysis_accession import filter_accession_data
from mutant_analysis_common_subsets import read_common_subset
from mutant_analysis_protein import calculate_average_residue_distance_to_ligand
from mutant_analysis_compounds import butina_cluster_compounds,visualize_molecular_subset_highlights
from mutant_analysis_clustermaps import extract_unique_connectivity,pivot_bioactivity_data,plot_bioactivity_heatmap,\
    plot_bioactivity_clustermap,extract_oldest_year
from mutant_analysis_type import read_mutation_distance_Epstein


def model_bioactivity_data():
    for path in ['../../data/pcm_models/', '../../data/qsar_all_models/', '../../data/qsar_separate_models/']:
        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)

    preferred_common_subset = 'all'
    common_subset_args = {'all': {'common': False,
                                  'sim': False,
                                  'sim_thres': None,
                                  'threshold': None,
                                  'variant_coverage': None},
                          'common_subset_20_sim_80': {'common': True,
                                                      'sim': True,
                                                      'sim_thres': 0.8,
                                                      'threshold': 2,
                                                      'variant_coverage': 0.2}}
    data = merge_chembl_papyrus_mutants('31', '05.5', 'nostereo', 1_000_000)

    sorted_accessions = list(zip(*sorted(Counter(data.accession).items(), key=itemgetter(1), reverse=True)))[0]

    for accession in sorted_accessions:

        # Skip if already exists
        if os.path.exists(f'../../data/pcm_models/pcm_randomsplit_{accession}.tsv'):
            continue

        accession_data = filter_accession_data(data, accession)

        # Train PCM on WT and mutant sequences
        # 1) Isolate IDs and dependent variable
        ids = accession_data[['connectivity', 'target_id', 'pchembl_value_Mean']]
        # 2) Keep unique mols and prots
        mols = accession_data[['connectivity', 'SMILES']].drop_duplicates()
        prots = accession_data[['target_id', 'sequence']].drop_duplicates()
        # 3) Obtain descriptors for unique mols, prots
        mdesc_type = Mold2(False)
        mol_descs = mdesc_type.calculate([Chem.MolFromSmiles(smiles) for smiles in mols.SMILES], show_banner=False,
                                        njobs=1)
        mol_descs = mol_descs.apply(lambda x: pd.to_numeric(x, errors='coerce'))
        pdesc_type = Transform(TransformType.AVG, ProteinDescriptors().get_descriptor('Zscale van Westen'))
        prot_descs = pdesc_type.pandas_get(prots.sequence, domains=50)
        prot_descs = prot_descs.apply(lambda x: pd.to_numeric(x, errors='coerce'))
        mol_descs = pd.concat([mols.connectivity.reset_index(drop=True),
                               mol_descs.reset_index(drop=True)
                               ], axis=1)
        prot_descs = pd.concat([prots.target_id.reset_index(drop=True),
                                prot_descs.reset_index(drop=True)
                                ], axis=1)
        # Join data
        pcm_data = (ids.merge(mol_descs, on='connectivity')
                       .merge(prot_descs, on='target_id')
                    )
        pcm_data = pcm_data.dropna()
        # Keep track of mutants ids
        target_ids = pcm_data.target_id
        pcm_data = pcm_data.drop(columns=['connectivity', 'target_id'])

        # Train PCM on random folds
        kfold = KFold(n_splits=5, shuffle=True, random_state=1234)
        model_pcm = XGBRegressor()
        ametrics, amodels = crossvalidate_model(pcm_data, model_pcm, kfold)
        # Train PCM on WT and mutants groups
        gkfolds = GroupKFold(n_splits=len(target_ids.unique()))
        model_pcm = XGBRegressor()
        gmetrics, gmodels = crossvalidate_model(pcm_data, model_pcm, gkfolds, target_ids)

        # Save performances
        ametrics.to_csv(f'../../data/pcm_models/pcm_randomsplit_{accession}.tsv', sep='\t')
        gmetrics.to_csv(f'../../data/pcm_models/pcm_mutantsplit_{accession}.tsv', sep='\t')
        for fname, models in [(f'../../data/pcm_models/pcm_randomsplit_{accession}_models.json', amodels),
                              (f'../../data/pcm_models/pcm_mutantsplit_{accession}_models.json', gmodels)]:
            with open(fname, 'w') as oh:
                json.dump({name: ml2json.serialize_model(model) for name, model in models.items()},
                          oh)
        ## QSAR
        # Train QSAR model considering all mutants as WT
        qsar_data = (ids.merge(mol_descs, on='connectivity'))
        qsar_data = qsar_data.dropna()
        target_ids = qsar_data.target_id
        qsar_data = qsar_data.drop(columns=['connectivity', 'target_id'])
        kfold = KFold(n_splits=5, shuffle=True, random_state=1234)
        model_qsar_all = XGBRegressor()
        ametrics, amodels = crossvalidate_model(qsar_data, model_qsar_all, kfold)
        gkfolds = GroupKFold(n_splits=len(target_ids.unique()))
        model_qsar_all = XGBRegressor()
        gmetrics, gmodels = crossvalidate_model(qsar_data, model_qsar_all, gkfolds, target_ids)

        # Save performances
        ametrics.to_csv(f'../../data/qsar_all_models/qsar_randomsplit_{accession}.tsv', sep='\t')
        gmetrics.to_csv(f'../../data/qsar_separate_models/qsar_mutantsplit_{accession}.tsv', sep='\t')
        for fname, models in [(f'../../data/qsar_all_models/qsar_randomsplit_{accession}_models.json', amodels),
                              (f'../../data/qsar_separate_models/qsar_mutantsplit_{accession}_models.json', gmodels)]:
            with open(fname, 'w') as oh:
                json.dump({name: ml2json.serialize_model(model) for name, model in models.items()},
                          oh)


if __name__ == '__main__':
    model_bioactivity_data()