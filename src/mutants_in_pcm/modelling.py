# -*- coding: utf-8 -*-


"""Bioactivity modelling."""

import glob
import json
import os
import multiprocessing
from collections import Counter
from itertools import chain
from operator import itemgetter

import ml2json
import pandas as pd
from Mold2_pywrapper import Mold2
from papyrus_scripts.modelling import crossvalidate_model
from prodec import ProteinDescriptors, Transform, TransformType
from rdkit import Chem
from sklearn.model_selection import KFold, GroupKFold
from tqdm.auto import tqdm
from xgboost import XGBRegressor

from .data_path import get_data_path
from .mutant_analysis_accession import filter_accession_data
from .preprocessing import merge_chembl_papyrus_mutants
from .mutant_analysis_common_subsets import get_variant_similar_subset
from .mutant_analysis_accession import get_statistics_across_accessions

def model_bioactivity_data(njobs: int = -1):
    """Model bioactivity data of (i) the entire set of bioactivities and of mutants only.
    Output results are stored under DATADIR/modelling

    :param njobs: number of parallel processes; -1 uses all cores, -2 uses all but one, ...
    """
    # Set number of cores
    if njobs < 0:
        njobs += multiprocessing.cpu_count() + 1
    if njobs == 0:
        raise ValueError('Number of jobs must not be 0.')
    # Obtain default path
    DATADIR = get_data_path()
    # Create output path
    path = os.path.join(DATADIR, 'modelling')
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
    # Obtain entire dataset
    merge_chembl_papyrus_mutants('31', '05.5', 'nostereo', 1_000_000, annotation_round=1)
    data = merge_chembl_papyrus_mutants('31', '05.5', 'nostereo', 1_000_000, annotation_round=2)
    # Obtain statistics per accession
    get_statistics_across_accessions('31', '05.5', 'nostereo', 1_000_000,
                                     1, DATADIR, True)
    get_statistics_across_accessions('31', '05.5', 'nostereo', 1_000_000,
                                     2, DATADIR, True)
    # Model accessions by decreasing number of datapoints
    sorted_accessions = list(zip(*sorted(Counter(data.accession).items(), key=itemgetter(1), reverse=True)))[0]
    pbar = tqdm(sorted_accessions, smoothing=0.0, ncols=90)
    for accession in pbar:
        # Determine names of all output files
        out_file_pcm_random_tsv = os.path.join(path, f'pcm_randomsplit_{accession}.tsv')
        out_file_pcm_random_json = os.path.join(path, f'pcm_randomsplit_{accession}_models.json')
        out_file_pcm_mutant_tsv = os.path.join(path, f'pcm_mutantsplit_{accession}.tsv')
        out_file_pcm_mutant_json = os.path.join(path, f'pcm_mutantsplit_{accession}_models.json')
        out_file_pcm_nomutant_random_tsv = os.path.join(path, f'pcm_nomutant_randomsplit_{accession}.tsv')
        out_file_pcm_nomutant_random_json = os.path.join(path, f'pcm_nomutant_randomsplit_{accession}_models.json')
        out_file_qsar_random_tsv = os.path.join(path, f'qsar_randomsplit_{accession}.tsv')
        out_file_qsar_random_json = os.path.join(path, f'qsar_randomsplit_{accession}_models.json')
        out_file_qsar_mutant_tsv = os.path.join(path, f'qsar_mutantsplit_{accession}.tsv')
        out_file_qsar_mutant_json = os.path.join(path, f'qsar_mutantsplit_{accession}_models.json')
        out_file_qsar_nomutant_random_tsv = os.path.join(path, f'qsar_nomutant_randomsplit_{accession}.tsv')
        out_file_qsar_nomutant_random_json = os.path.join(path, f'qsar_nomutant_randomsplit_{accession}_models.json')
        out_pcm_mutants = [out_file_pcm_random_tsv, out_file_pcm_random_json,
                           out_file_pcm_mutant_tsv, out_file_pcm_mutant_json]
        out_pcm_no_mutants = [out_file_pcm_nomutant_random_tsv, out_file_pcm_nomutant_random_json]
        out_qsar_mutants = [out_file_qsar_random_tsv, out_file_qsar_random_json,
                            out_file_qsar_mutant_tsv, out_file_qsar_mutant_json]
        out_qsar_no_mutants = [out_file_qsar_nomutant_random_tsv, out_file_qsar_nomutant_random_json]
        all_files = list(chain(out_pcm_mutants, out_pcm_no_mutants, out_qsar_mutants, out_qsar_no_mutants))
        # Skip if all output files already exists
        if all(os.path.exists(f) for f in all_files):
            pbar.update()
            continue
        # Obtain data for this accession only
        accession_data = filter_accession_data(data, accession)
        # Set description of progress bar
        pbar.desc = f'{accession} - {accession_data.shape[0]} data points'
        pbar.refresh()
        mol_descs = None
        prot_descs = None
        # Run PCM if not any missing output file
        if not all(os.path.exists(f) for f in out_pcm_mutants):
            # Train PCM on WT and mutant sequences
            # 1) Isolate IDs and dependent variable
            ids = accession_data[['connectivity', 'target_id', 'pchembl_value_Mean']]
            # 2) Keep unique mols and prots
            mols = accession_data[['connectivity', 'SMILES']].drop_duplicates(subset=['connectivity'])
            prots = accession_data[['target_id', 'sequence']].drop_duplicates()
            # 3) Obtain Mold2 descriptors for unique mols
            mdesc_type = Mold2(False)
            mol_descs = mdesc_type.calculate([Chem.MolFromSmiles(smiles) for smiles in mols.SMILES], show_banner=False,
                                            njobs=njobs)
            mol_descs = (mol_descs
                         .apply(lambda x: pd.to_numeric(x, errors='coerce'))
                         .fillna(0))
            mol_descs = pd.concat([mols.connectivity.reset_index(drop=True),
                                   mol_descs.reset_index(drop=True)
                                   ], axis=1)
            # Obtain 3 first PCs of Sandberg Zscales
            pdesc_type = Transform(TransformType.AVG, ProteinDescriptors().get_descriptor('Zscale van Westen'))
            # Drop proteins sequences not supported by the descriptor
            validity = prots.sequence.apply(pdesc_type.Descriptor.is_sequence_valid)
            prots = prots.loc[validity, :]
            if len(prots) > 0:
                # Obtain domain averages
                prot_descs = pdesc_type.pandas_get(prots.sequence, domains=50, quiet=True)
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
                    # Random cross-validation
                    kfold = KFold(n_splits=5, shuffle=True, random_state=1234)
                    model_pcm = XGBRegressor()
                    cv_metrics, cv_models = crossvalidate_model(pcm_data, model_pcm, kfold, verbose=False)
                    # Leave-one-out mutant cross-validation
                    model_pcm = XGBRegressor()
                    if len(prots.target_id) > 1:
                        gkfolds = GroupKFold(n_splits=len(pcm_target_ids.unique()))
                        loo_metrics, loo_models = crossvalidate_model(pcm_data, model_pcm, gkfolds, pcm_target_ids,
                                                                      verbose=False)
                    else:
                        loo_metrics, loo_models = pd.DataFrame(), {}
                    # Save performances
                    cv_metrics.to_csv(out_file_pcm_random_tsv, sep='\t')
                    loo_metrics.to_csv(out_file_pcm_mutant_tsv, sep='\t')
                    # Save models in json
                    for fname, models in [(out_file_pcm_random_json, cv_models),
                                          (out_file_pcm_mutant_json, loo_models)]:
                        with open(fname, 'w') as oh:
                            json.dump({name: ml2json.serialize_model(model)
                                       for name, model in models.items()},
                                      oh)
        # Run QSAR if not missing any output file
        if not all(os.path.exists(f) for f in out_qsar_mutants):
            if mol_descs is None:  # PCM part was skipped
                # 1) Isolate IDs and dependent variable
                ids = accession_data[['connectivity', 'target_id', 'pchembl_value_Mean']]
                # 2) Keep unique mols and prots
                mols = accession_data[['connectivity', 'SMILES']].drop_duplicates()
                prots = accession_data[['target_id', 'sequence']].drop_duplicates()
                # 3) Obtain descriptors for unique mols
                mdesc_type = Mold2(False)
                mol_descs = mdesc_type.calculate([Chem.MolFromSmiles(smiles) for smiles in mols.SMILES],
                                                 show_banner=False,
                                                 njobs=njobs)
                mol_descs = (mol_descs.apply(lambda x: pd.to_numeric(x, errors='coerce'))
                             .fillna(0))
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
                # Random cross-validation
                kfold = KFold(n_splits=5, shuffle=True, random_state=1234)
                model_qsar_all = XGBRegressor()
                cv_metrics, cv_models = crossvalidate_model(qsar_data, model_qsar_all, kfold, verbose=False)
                # LOO motant-cross-validation
                model_qsar_all = XGBRegressor()
                if len(prots.target_id) > 1:
                    gkfolds = GroupKFold(n_splits=len(qsar_target_ids.unique()))
                    loo_metrics, loo_models = crossvalidate_model(qsar_data, model_qsar_all, gkfolds, qsar_target_ids,
                                                                  verbose=False)
                else:
                    loo_metrics, loo_models = pd.DataFrame(), {}
                # Save performances
                cv_metrics.to_csv(out_file_qsar_random_tsv, sep='\t')
                loo_metrics.to_csv(out_file_qsar_mutant_tsv, sep='\t')
                # Save models to json
                for fname, models in [(out_file_qsar_random_json, cv_models),
                                      (out_file_qsar_mutant_json, loo_models)]:
                    with open(fname, 'w') as oh:
                        json.dump({name: ml2json.serialize_model(model)
                                   for name, model in models.items()},
                                  oh)
        # Run PCM with mutants removed
        if not all(os.path.exists(f) for f in out_pcm_no_mutants):
            if prot_descs is None:
                # 1) Isolate IDs and dependent variable
                ids = accession_data[['connectivity', 'target_id', 'pchembl_value_Mean']]
                # 2) Keep unique mols and prots
                mols = accession_data[['connectivity', 'SMILES']].drop_duplicates()
                prots = accession_data[['target_id', 'sequence']].drop_duplicates()
                # 3) Obtain descriptors for unique mols
                mdesc_type = Mold2(False)
                mol_descs = mdesc_type.calculate([Chem.MolFromSmiles(smiles) for smiles in mols.SMILES],
                                                 show_banner=False,
                                                 njobs=njobs)
                mol_descs = mol_descs.apply(lambda x: pd.to_numeric(x, errors='coerce'))
                mol_descs = pd.concat([mols.connectivity.reset_index(drop=True),
                                       mol_descs.reset_index(drop=True)
                                       ], axis=1)
                # Obtain descriptors for unique proteins
                pdesc_type = Transform(TransformType.AVG, ProteinDescriptors().get_descriptor('Zscale van Westen'))
                # Drop proteins sequences not supported by the descriptor
                validity = prots.sequence.apply(pdesc_type.Descriptor.is_sequence_valid)
                prots = prots.loc[validity, :]
                if len(prots) > 0:
                    # Obtain domain averages
                    prot_descs = pdesc_type.pandas_get(prots.sequence, domains=50, quiet=True)
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
                    # Random cross-validation
                    kfold = KFold(n_splits=5, shuffle=True, random_state=1234)
                    model_pcm = XGBRegressor()
                    cv_metrics, cv_models = crossvalidate_model(pcm_data, model_pcm, kfold, verbose=False)
                    # Save performances
                    cv_metrics.to_csv(out_file_pcm_nomutant_random_tsv, sep='\t')
                    # Save models to json
                    with open(out_file_pcm_nomutant_random_json, 'w') as oh:
                            json.dump({name: ml2json.serialize_model(model)
                                       for name, model in cv_models.items()},
                                      oh)
        # Run QSAR with mutants removed
        if not all(os.path.exists(f) for f in out_qsar_no_mutants):
            if mol_descs is None:  # PCM part was skipped
                # 1) Isolate IDs and dependent variable
                ids = accession_data[['connectivity', 'target_id', 'pchembl_value_Mean']]
                # 2) Keep unique mols and prots
                mols = accession_data[['connectivity', 'SMILES']].drop_duplicates()
                prots = accession_data[['target_id', 'sequence']].drop_duplicates()
                # 3) Obtain descriptors for unique mols
                mdesc_type = Mold2(False)
                mol_descs = mdesc_type.calculate([Chem.MolFromSmiles(smiles) for smiles in mols.SMILES],
                                                 show_banner=False,
                                                 njobs=njobs)
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
                # Random cross-validation
                kfold = KFold(n_splits=5, shuffle=True, random_state=1234)
                model_qsar_all = XGBRegressor()
                cv_metrics, cv_models = crossvalidate_model(qsar_data, model_qsar_all, kfold, verbose=False)
                # Save performances
                cv_metrics.to_csv(out_file_qsar_nomutant_random_tsv, sep='\t')
                # Save models to json
                with open(out_file_qsar_nomutant_random_json, 'w') as oh:
                    json.dump({name: ml2json.serialize_model(model)
                               for name, model in cv_models.items()},
                              oh)

def model_bioactivity_data_common_subsets(njobs: int = -1):
    """Model bioactivities of the similarity-based common subsets only.
    Output results are stored under DATADIR/modelling.
    """
    # Set number of cores
    if njobs == 0:
        raise ValueError('Number of jobs must not be 0.')
    if njobs < 0:
        njobs += multiprocessing.cpu_count() + 1
    # Obtain default path
    DATADIR = get_data_path()
    # Create output path
    path = os.path.join(DATADIR, 'modelling')
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
    # Obtain entire dataset
    data = merge_chembl_papyrus_mutants('31', '05.5', 'nostereo', 1_000_000,
                                        annotation_round=2)
    # Molecules ids
    mol_mappings = (data[['connectivity', 'SMILES']]
                    .drop_duplicates(subset=['connectivity'])
                    .dropna())
    # Protein sequences ids
    seq_mappings = (data[['target_id', 'sequence']]
                    .drop_duplicates(subset=['target_id'])
                    .dropna())
    # Obtain paths to common subsets
    datasets = [(common_set, os.path.basename(common_set)[os.path.basename(common_set).find('modelling_dataset_') +18:].replace('_Thr2_Cov20_Sim80.csv', ''))
                for common_set in glob.glob(os.path.join(DATADIR, '**Cov20_Sim80', '*.csv'))]
    if not len(datasets):
        # Obtain common subsets
        pbar = tqdm(data.accession.unique(), desc='Calculating common subsets')
        for accession in pbar:
            get_variant_similar_subset(data, accession, sim_thres=0.8, threshold=2, variant_coverage=0.2,
                                       save_dataset=True, output_dir=os.path.join(DATADIR, 'Cov20_Sim80'))
        datasets = [(common_set, os.path.basename(common_set)[os.path.basename(common_set).find('modelling_dataset_') + 18:].replace('_Thr2_Cov20_Sim80.csv', ''))
                    for common_set in glob.glob(os.path.join(DATADIR, '**Cov20_Sim80', '*.csv'))]
        if not len(datasets):
            raise RuntimeError('Error while computing common subsets.')
    # Model bioactivities of common subsets
    pbar = tqdm(datasets, smoothing=0.0, ncols=90)
    for common_set, accession in pbar:
        accession_data = pd.read_csv(common_set, sep='\t')
        if accession_data.empty:
            pbar.update()
            continue
        # Determine names of all output files
        out_file_pcm_random_tsv = os.path.join(path, f'pcm_common_subset_randomsplit_{accession}.tsv')
        out_file_pcm_random_json = os.path.join(path, f'pcm_common_subset_randomsplit_{accession}_models.json')
        out_file_pcm_mutant_tsv = os.path.join(path, f'pcm_common_subset_mutantsplit_{accession}.tsv')
        out_file_pcm_mutant_json = os.path.join(path, f'pcm_common_subset_mutantsplit_{accession}_models.json')
        out_file_pcm_nomutant_random_tsv = os.path.join(path, f'pcm_common_subset_nomutant_randomsplit_{accession}.tsv')
        out_file_pcm_nomutant_random_json = os.path.join(path, f'pcm_common_subset_nomutant_randomsplit_{accession}_models.json')
        out_file_qsar_random_tsv = os.path.join(path, f'qsar_common_subset_randomsplit_{accession}.tsv')
        out_file_qsar_random_json = os.path.join(path, f'qsar_common_subset_randomsplit_{accession}_models.json')
        out_file_qsar_mutant_tsv = os.path.join(path, f'qsar_common_subset_mutantsplit_{accession}.tsv')
        out_file_qsar_mutant_json = os.path.join(path, f'qsar_common_subset_mutantsplit_{accession}_models.json')
        out_file_qsar_nomutant_random_tsv = os.path.join(path, f'qsar_common_subset_nomutant_randomsplit_{accession}.tsv')
        out_file_qsar_nomutant_random_json = os.path.join(path, f'qsar_common_subset_nomutant_randomsplit_{accession}_models.json')
        out_pcm_mutants = [out_file_pcm_random_tsv, out_file_pcm_random_json,
                           out_file_pcm_mutant_tsv, out_file_pcm_mutant_json]
        out_pcm_no_mutants = [out_file_pcm_nomutant_random_tsv, out_file_pcm_nomutant_random_json]
        out_qsar_mutants = [out_file_qsar_random_tsv, out_file_qsar_random_json,
                            out_file_qsar_mutant_tsv, out_file_qsar_mutant_json]
        out_qsar_no_mutants = [out_file_qsar_nomutant_random_tsv, out_file_qsar_nomutant_random_json]
        all_files = list(chain(out_pcm_mutants, out_pcm_no_mutants, out_qsar_mutants, out_qsar_no_mutants))
        # Skip if all files already exists
        if all(os.path.exists(f) for f in all_files):
            pbar.update()
            continue
        # Set description of progress bar
        pbar.desc = f'{accession} - {accession_data.shape[0]} common data points'
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
            # 3) Obtain descriptors for unique mols
            mdesc_type = Mold2(False)
            mol_descs = mdesc_type.calculate([Chem.MolFromSmiles(smiles) for smiles in mols.SMILES], show_banner=False,
                                            njobs=njobs)
            mol_descs = mol_descs.apply(lambda x: pd.to_numeric(x, errors='coerce'))
            mol_descs = pd.concat([mols.connectivity.reset_index(drop=True),
                                   mol_descs.reset_index(drop=True)
                                   ], axis=1)
            # Obtain descriptors for unique proteins
            pdesc_type = Transform(TransformType.AVG, ProteinDescriptors().get_descriptor('Zscale van Westen'))
            # Drop proteins sequences not supported by the descriptor
            validity = prots.sequence.apply(pdesc_type.Descriptor.is_sequence_valid)
            prots = prots.loc[validity, :]
            if len(prots) > 0:
                # Obtain domain averages
                prot_descs = pdesc_type.pandas_get(prots.sequence, domains=50, quiet=True)
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
                    # Random cross-validation
                    kfold = KFold(n_splits=5, shuffle=True, random_state=1234)
                    model_pcm = XGBRegressor()
                    cv_metrics, cv_models = crossvalidate_model(pcm_data, model_pcm, kfold, verbose=False)
                    # LOO mutant CV
                    model_pcm = XGBRegressor()
                    if len(prots.target_id) > 1:
                        gkfolds = GroupKFold(n_splits=len(pcm_target_ids.unique()))
                        grouped_metrics, grouped_models = crossvalidate_model(pcm_data, model_pcm, gkfolds, pcm_target_ids,
                                                                              verbose=False)
                    else:
                        grouped_metrics, grouped_models = pd.DataFrame(), {}
                    # Save performances
                    cv_metrics.to_csv(out_file_pcm_random_tsv, sep='\t')
                    grouped_metrics.to_csv(out_file_pcm_mutant_tsv, sep='\t')
                    # Save models to json
                    for fname, models in [(out_file_pcm_random_json, cv_models),
                                          (out_file_pcm_mutant_json, grouped_models)]:
                        with open(fname, 'w') as oh:
                            json.dump({name: ml2json.serialize_model(model)
                                       for name, model in models.items()},
                                      oh)
        # Run QSAR if not missing any output file
        if not all(os.path.exists(f) for f in out_qsar_mutants):
            if mol_descs is None:  # PCM part was skipped
                # 1) Isolate IDs and dependent variable
                ids = accession_data[['connectivity', 'target_id', 'pchembl_value_Mean']]
                # 2) Keep unique mols and prots
                mols = accession_data[['connectivity']].drop_duplicates().merge(mol_mappings, on='connectivity')
                prots = accession_data[['target_id']].drop_duplicates().merge(seq_mappings, on='target_id')
                # 3) Obtain descriptors for unique mols
                mdesc_type = Mold2(False)
                mol_descs = mdesc_type.calculate([Chem.MolFromSmiles(smiles) for smiles in mols.SMILES],
                                                 show_banner=False,
                                                 njobs=njobs)
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
                # Random CV
                kfold = KFold(n_splits=5, shuffle=True, random_state=1234)
                model_qsar_all = XGBRegressor()
                cv_metrics, cv_models = crossvalidate_model(qsar_data, model_qsar_all, kfold, verbose=False)
                # LOO mutant CV
                model_qsar_all = XGBRegressor()
                if len(prots.target_id) > 1:
                    gkfolds = GroupKFold(n_splits=len(qsar_target_ids.unique()))
                    grouped_metrics, grouped_models = crossvalidate_model(qsar_data, model_qsar_all, gkfolds, qsar_target_ids,
                                                                          verbose=False)
                else:
                    grouped_metrics, grouped_models = pd.DataFrame(), {}
                # Save performances
                cv_metrics.to_csv(out_file_qsar_random_tsv, sep='\t')
                grouped_metrics.to_csv(out_file_qsar_mutant_tsv, sep='\t')
                # Save models to json
                for fname, models in [(out_file_qsar_random_json, cv_models),
                                      (out_file_qsar_mutant_json, grouped_models)]:
                    with open(fname, 'w') as oh:
                        json.dump({name: ml2json.serialize_model(model)
                                   for name, model in models.items()},
                                  oh)
        # Run PCM with mutants removed
        if not all(os.path.exists(f) for f in out_pcm_no_mutants):
            if prot_descs is None:
                # 1) Isolate IDs and dependent variable
                ids = accession_data[['connectivity', 'target_id', 'pchembl_value_Mean']]
                # 2) Keep unique mols and prots
                mols = accession_data[['connectivity']].drop_duplicates().merge(mol_mappings, on='connectivity')
                prots = accession_data[['target_id']].drop_duplicates().merge(seq_mappings, on='target_id')
                # 3) Obtain descriptors for unique mols
                mdesc_type = Mold2(False)
                mol_descs = mdesc_type.calculate([Chem.MolFromSmiles(smiles) for smiles in mols.SMILES],
                                                 show_banner=False,
                                                 njobs=njobs)
                mol_descs = mol_descs.apply(lambda x: pd.to_numeric(x, errors='coerce'))
                mol_descs = pd.concat([mols.connectivity.reset_index(drop=True),
                                       mol_descs.reset_index(drop=True)
                                       ], axis=1)
                # Obtain descriptors for unique proteins
                pdesc_type = Transform(TransformType.AVG, ProteinDescriptors().get_descriptor('Zscale van Westen'))
                # Drop proteins sequences not supported by the descriptor
                validity = prots.sequence.apply(pdesc_type.Descriptor.is_sequence_valid)
                prots = prots.loc[validity, :]
                if len(prots) > 0:
                    # Obtain domain averages
                    prot_descs = pdesc_type.pandas_get(prots.sequence, domains=50, quiet=True)
                    prot_descs = prot_descs.apply(lambda x: pd.to_numeric(x, errors='coerce'))
                    prot_descs = pd.concat([prots.target_id.reset_index(drop=True),
                                            prot_descs.reset_index(drop=True)
                                            ], axis=1)
            # Skip if no data to model
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
                    # Random CV
                    kfold = KFold(n_splits=5, shuffle=True, random_state=1234)
                    model_pcm = XGBRegressor()
                    cv_metrics, cv_models = crossvalidate_model(pcm_data, model_pcm, kfold, verbose=False)
                    # Save performances
                    cv_metrics.to_csv(out_file_pcm_nomutant_random_tsv, sep='\t')
                    # Save models to json
                    with open(out_file_pcm_nomutant_random_json, 'w') as oh:
                            json.dump({name: ml2json.serialize_model(model)
                                       for name, model in cv_models.items()},
                                      oh)
        # Run QSAR with mutants removed
        if not all(os.path.exists(f) for f in out_qsar_no_mutants):
            if mol_descs is None:  # PCM part was skipped
                # 1) Isolate IDs and dependent variable
                ids = accession_data[['connectivity', 'target_id', 'pchembl_value_Mean']]
                # 2) Keep unique mols and prots
                mols = accession_data[['connectivity']].drop_duplicates().merge(mol_mappings, on='connectivity')
                prots = accession_data[['target_id']].drop_duplicates().merge(seq_mappings, on='target_id')
                # 3) Obtain descriptors for unique mols
                mdesc_type = Mold2(False)
                mol_descs = mdesc_type.calculate([Chem.MolFromSmiles(smiles) for smiles in mols.SMILES],
                                                 show_banner=False,
                                                 njobs=njobs)
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
                # Random CV
                kfold = KFold(n_splits=5, shuffle=True, random_state=1234)
                model_qsar_all = XGBRegressor()
                cv_metrics, cv_models = crossvalidate_model(qsar_data, model_qsar_all, kfold, verbose=False)
                # Save performances
                cv_metrics.to_csv(out_file_qsar_nomutant_random_tsv, sep='\t')
                # Save models to json
                with open(out_file_qsar_nomutant_random_json, 'w') as oh:
                    json.dump({name: ml2json.serialize_model(model)
                               for name, model in cv_models.items()},
                              oh)

if __name__ == '__main__':
    model_bioactivity_data()
    model_bioactivity_data_common_subsets()
