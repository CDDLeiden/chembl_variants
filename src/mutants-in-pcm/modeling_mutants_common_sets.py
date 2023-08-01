# -*- coding: utf-8 -*-


"""Bioactivity modelling."""

import glob
import json
from itertools import chain

import pandas as pd
import os

import ml2json
from rdkit import Chem
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, GroupKFold
from papyrus_scripts.modelling import crossvalidate_model
from prodec import ProteinDescriptors, Transform, TransformType
from Mold2_pywrapper import Mold2
from tqdm.auto import tqdm


def model_common_subset_data(folder: str):
    for path in ['../../data/common_sets/pcm_models/',
                 '../../data/common_sets/qsar_all_models/',
                 '../../data/common_sets/qsar_separate_models/']:
        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)

    mol_mappings = (pd.read_csv('../../data/merged_chembl31-annotated_papyrus05.5nostereo_data_with_mutants.csv',
                               sep='\t', usecols=['connectivity', 'SMILES'])
                    .drop_duplicates(subset=['connectivity'])
                    .dropna())

    seq_mappings = (pd.read_csv('../../data/merged_chembl31-annotated_papyrus05.5nostereo_data_with_mutants.csv',
                               sep='\t', usecols=['target_id', 'sequence'])
                    .drop_duplicates(subset=['target_id'])
                    .dropna())

    datasets = [(path, path[path.find('modelling_dataset_') +18:].replace('_Thr2_Cov20_Sim80.csv', ''))
                for path in glob.glob(os.path.join(os.path.abspath(folder), '*.csv'))]
    pbar = tqdm(datasets)

    for path, accession in pbar:
        accession_data = pd.read_csv(path, sep='\t')
        if accession_data.empty:
            continue
        # Determine names of all output files
        out_file_pcm_random_tsv = f'../../data/common_sets/pcm_models/pcm_randomsplit_{accession}.tsv'
        out_file_pcm_random_json = f'../../data/common_sets/pcm_models/pcm_randomsplit_{accession}_models.json'
        out_file_pcm_mutant_tsv = f'../../data/common_sets/pcm_models/pcm_mutantsplit_{accession}.tsv'
        out_file_pcm_mutant_json = f'../../data/common_sets/pcm_models/pcm_mutantsplit_{accession}_models.json'
        out_file_pcm_nomutant_random_tsv = f'../../data/common_sets/pcm_models/pcm_nomutant_randomsplit_{accession}.tsv'
        out_file_pcm_nomutant_random_json = f'../../data/common_sets/pcm_models/pcm_nomutant_randomsplit_{accession}_models.json'
        out_file_qsar_random_tsv = f'../../data/common_sets/qsar_all_models/qsar_randomsplit_{accession}.tsv'
        out_file_qsar_random_json = f'../../data/common_sets/qsar_all_models/qsar_randomsplit_{accession}_models.json'
        out_file_qsar_mutant_tsv = f'../../data/common_sets/qsar_separate_models/qsar_mutantsplit_{accession}.tsv'
        out_file_qsar_mutant_json = f'../../data/common_sets/qsar_separate_models/qsar_mutantsplit_{accession}_models.json'
        out_file_qsar_nomutant_random_tsv = f'../../data/common_sets/qsar_all_models/qsar_nomutant_randomsplit_{accession}.tsv'
        out_file_qsar_nomutant_random_json = f'../../data/common_sets/qsar_all_models/qsar_nomutant_randomsplit_{accession}_models.json'
        out_pcm_mutants = [out_file_pcm_random_tsv, out_file_pcm_random_json,
                           out_file_pcm_mutant_tsv, out_file_pcm_mutant_json]
        out_pcm_no_mutants = [out_file_pcm_nomutant_random_tsv, out_file_pcm_nomutant_random_json]
        out_qsar_mutants = [out_file_qsar_random_tsv, out_file_qsar_random_json,
                            out_file_qsar_mutant_tsv, out_file_qsar_mutant_json]
        out_qsar_no_mutants = [out_file_qsar_nomutant_random_tsv, out_file_qsar_nomutant_random_json]
        all_files = list(chain(out_pcm_mutants, out_pcm_no_mutants, out_qsar_mutants, out_qsar_no_mutants))
        # Skip if all files already exists
        if all(os.path.exists(f) for f in all_files):
            continue
        # Set description of progress bar
        pbar.desc = accession
        pbar.refresh()
        mol_descs = None
        prot_descs = None
        # Run PCM if not any missing output file
        if not all(os.path.exists(f) for f in out_pcm_mutants):
            # Train PCM on WT and mutant sequences
            # 1) Isolate IDs and dependent variable
            ids = accession_data[['connectivity', 'target_id', 'pchembl_value_Mean']]
            # 2) Keep unique mols and prots
            mols = accession_data[['connectivity']].drop_duplicates().merge(mol_mappings, on='connectivity')
            prots = accession_data[['target_id']].drop_duplicates().merge(seq_mappings, on='target_id')
            # 3) Obtain descriptors for unique mols, prots
            mdesc_type = Mold2(False)
            mol_descs = mdesc_type.calculate([Chem.MolFromSmiles(smiles) for smiles in mols.SMILES], show_banner=False,
                                            njobs=1)
            mol_descs = mol_descs.apply(lambda x: pd.to_numeric(x, errors='coerce'))
            mol_descs = pd.concat([mols.connectivity.reset_index(drop=True),
                                   mol_descs.reset_index(drop=True)
                                   ], axis=1)
            pdesc_type = Transform(TransformType.AVG, ProteinDescriptors().get_descriptor('Zscale van Westen'))
            # Drop proteins sequences not supported by the descriptor
            validity = prots.sequence.apply(pdesc_type.Descriptor.is_sequence_valid)
            prots = prots.loc[validity, :]
            if len(prots) > 0:
                prot_descs = pdesc_type.pandas_get(prots.sequence, domains=50)
                prot_descs = prot_descs.apply(lambda x: pd.to_numeric(x, errors='coerce'))
                prot_descs = pd.concat([prots.target_id.reset_index(drop=True),
                                        prot_descs.reset_index(drop=True)
                                        ], axis=1)
                # Join data
                pcm_data = (ids.merge(mol_descs, on='connectivity')
                               .merge(prot_descs, on='target_id')
                            )
                pcm_data = pcm_data.dropna()
                pcm_target_ids = pcm_data.target_id
                pcm_data = pcm_data.drop(columns=['connectivity', 'target_id'])
                # Skip if not enough data for CV
                if pcm_data.shape[0] >= 5:  # Number of folds
                    # Train PCM on random folds
                    kfold = KFold(n_splits=5, shuffle=True, random_state=1234)
                    model_pcm = XGBRegressor()
                    ametrics, amodels = crossvalidate_model(pcm_data, model_pcm, kfold)
                    # Train PCM on WT and mutants groups
                    model_pcm = XGBRegressor()
                    if len(prots.target_id) > 1:
                        gkfolds = GroupKFold(n_splits=len(pcm_target_ids.unique()))
                        gmetrics, gmodels = crossvalidate_model(pcm_data, model_pcm, gkfolds, pcm_target_ids)
                    else:
                        gmetrics, gmodels = pd.DataFrame(), {}
                    # Save performances
                    ametrics.to_csv(out_file_pcm_random_tsv, sep='\t')
                    gmetrics.to_csv(out_file_pcm_mutant_tsv, sep='\t')
                    for fname, models in [(out_file_pcm_random_json, amodels), (out_file_pcm_mutant_json, gmodels)]:
                        with open(fname, 'w') as oh:
                            json.dump({name: ml2json.serialize_model(model) for name, model in models.items()},
                                      oh)
        # Run QSAR if not missing any output file
        if not all(os.path.exists(f) for f in out_qsar_mutants):
            if mol_descs is None:  # PCM part was skipped
                # 1) Isolate IDs and dependent variable
                ids = accession_data[['connectivity', 'target_id', 'pchembl_value_Mean']]
                # 2) Keep unique mols and prots
                mols = accession_data[['connectivity']].drop_duplicates().merge(mol_mappings, on='connectivity')
                prots = accession_data[['target_id']].drop_duplicates().merge(seq_mappings, on='target_id')
                # 3) Obtain descriptors for unique mols, prots
                mdesc_type = Mold2(False)
                mol_descs = mdesc_type.calculate([Chem.MolFromSmiles(smiles) for smiles in mols.SMILES],
                                                 show_banner=False,
                                                 njobs=1)
                mol_descs = mol_descs.apply(lambda x: pd.to_numeric(x, errors='coerce'))
                mol_descs = pd.concat([mols.connectivity.reset_index(drop=True),
                                       mol_descs.reset_index(drop=True)
                                       ], axis=1)
            # Train QSAR model considering all mutants as WT
            qsar_data = (ids.merge(mol_descs, on='connectivity'))
            qsar_data = qsar_data.dropna()
            qsar_target_ids = qsar_data.target_id
            qsar_data = qsar_data.drop(columns=['connectivity', 'target_id'])
            # Skip if not enough data for CV
            if qsar_data.shape[0] >= 5:  # Number of folds
                kfold = KFold(n_splits=5, shuffle=True, random_state=1234)
                model_qsar_all = XGBRegressor()
                ametrics, amodels = crossvalidate_model(qsar_data, model_qsar_all, kfold)
                model_qsar_all = XGBRegressor()
                if len(prots.target_id) > 1:
                    gkfolds = GroupKFold(n_splits=len(qsar_target_ids.unique()))
                    gmetrics, gmodels = crossvalidate_model(qsar_data, model_qsar_all, gkfolds, qsar_target_ids)
                else:
                    gmetrics, gmodels = pd.DataFrame(), {}
                # Save performances
                ametrics.to_csv(out_file_qsar_random_tsv, sep='\t')
                gmetrics.to_csv(out_file_qsar_mutant_tsv, sep='\t')
                for fname, models in [(out_file_qsar_random_json, amodels), (out_file_pcm_mutant_json, gmodels)]:
                    with open(fname, 'w') as oh:
                        json.dump({name: ml2json.serialize_model(model) for name, model in models.items()},
                                  oh)
        # Run PCM with mutants removed
        if not all(os.path.exists(f) for f in out_pcm_no_mutants):
            if prot_descs is None:
                # 1) Isolate IDs and dependent variable
                ids = accession_data[['connectivity', 'target_id', 'pchembl_value_Mean']]
                # 2) Keep unique mols and prots
                mols = accession_data[['connectivity']].drop_duplicates().merge(mol_mappings, on='connectivity')
                prots = accession_data[['target_id']].drop_duplicates().merge(seq_mappings, on='target_id')
                # 3) Obtain descriptors for unique mols, prots
                mdesc_type = Mold2(False)
                mol_descs = mdesc_type.calculate([Chem.MolFromSmiles(smiles) for smiles in mols.SMILES],
                                                 show_banner=False,
                                                 njobs=1)
                mol_descs = mol_descs.apply(lambda x: pd.to_numeric(x, errors='coerce'))
                mol_descs = pd.concat([mols.connectivity.reset_index(drop=True),
                                       mol_descs.reset_index(drop=True)
                                       ], axis=1)
                pdesc_type = Transform(TransformType.AVG, ProteinDescriptors().get_descriptor('Zscale van Westen'))
                # Drop proteins sequences not supported by the descriptor
                validity = prots.sequence.apply(pdesc_type.Descriptor.is_sequence_valid)
                prots = prots.loc[validity, :]
                if len(prots) > 0:
                    prot_descs = pdesc_type.pandas_get(prots.sequence, domains=50)
                    prot_descs = prot_descs.apply(lambda x: pd.to_numeric(x, errors='coerce'))
                    prot_descs = pd.concat([prots.target_id.reset_index(drop=True),
                                            prot_descs.reset_index(drop=True)
                                            ], axis=1)
            # No data to model
            if prot_descs is not None:
                # Join data
                pcm_data = (ids.merge(mol_descs, on='connectivity')
                            .merge(prot_descs, on='target_id')
                            )
                pcm_data = pcm_data.dropna()
                # Keep track of mutants ids
                pcm_target_ids = pcm_data.target_id
                # Drop mutants
                mutants_pcm_target_ids = pd.Series([target_id
                                                    for target_id in pcm_target_ids
                                                    if not target_id.endswith('WT')])
                pcm_data = pcm_data[~pcm_data.target_id.isin(mutants_pcm_target_ids)]
                pcm_data = pcm_data.drop(columns=['connectivity', 'target_id'])
                # Skip if not enough data for CV
                if pcm_data.shape[0] >= 5:  # Number of folds
                    # Train PCM on random folds
                    kfold = KFold(n_splits=5, shuffle=True, random_state=1234)
                    model_pcm = XGBRegressor()
                    ametrics, amodels = crossvalidate_model(pcm_data, model_pcm, kfold)
                    # Save performances
                    ametrics.to_csv(out_file_pcm_nomutant_random_tsv, sep='\t')
                    with open(out_file_pcm_nomutant_random_json, 'w') as oh:
                            json.dump({name: ml2json.serialize_model(model) for name, model in amodels.items()},
                                      oh)
        # Run QSAR with mutants removed
        if not all(os.path.exists(f) for f in out_qsar_no_mutants):
            if mol_descs is None:  # PCM part was skipped
                # 1) Isolate IDs and dependent variable
                ids = accession_data[['connectivity', 'target_id', 'pchembl_value_Mean']]
                # 2) Keep unique mols and prots
                mols = accession_data[['connectivity']].drop_duplicates().merge(mol_mappings, on='connectivity')
                prots = accession_data[['target_id']].drop_duplicates().merge(seq_mappings, on='target_id')
                # 3) Obtain descriptors for unique mols, prots
                mdesc_type = Mold2(False)
                mol_descs = mdesc_type.calculate([Chem.MolFromSmiles(smiles) for smiles in mols.SMILES],
                                                 show_banner=False,
                                                 njobs=1)
                mol_descs = mol_descs.apply(lambda x: pd.to_numeric(x, errors='coerce'))
                mol_descs = pd.concat([mols.connectivity.reset_index(drop=True),
                                       mol_descs.reset_index(drop=True)
                                       ], axis=1)
            # Train QSAR model considering all mutants as WT
            qsar_data = (ids.merge(mol_descs, on='connectivity'))
            qsar_data = qsar_data.dropna()
            qsar_target_ids = qsar_data.target_id
            # Drop mutants
            mutants_qsar_target_ids = pd.Series([target_id
                                                 for target_id in qsar_target_ids
                                                 if not target_id.endswith('WT')])
            qsar_data = qsar_data[~qsar_data.target_id.isin(mutants_qsar_target_ids)]
            qsar_data = qsar_data.drop(columns=['connectivity', 'target_id'])
            # Skip if not enough data for CV
            if qsar_data.shape[0] >= 5:  # Number of folds
                kfold = KFold(n_splits=5, shuffle=True, random_state=1234)
                model_qsar_all = XGBRegressor()
                ametrics, amodels = crossvalidate_model(qsar_data, model_qsar_all, kfold)
                # Save performances
                ametrics.to_csv(out_file_qsar_nomutant_random_tsv, sep='\t')
                with open(out_file_qsar_nomutant_random_json, 'w') as oh:
                    json.dump({name: ml2json.serialize_model(model) for name, model in amodels.items()},
                              oh)


if __name__ == '__main__':
    FOLDER = '../../data/test_similar_datasets/'
    model_common_subset_data(FOLDER)