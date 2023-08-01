# -*- coding: utf-8 -*-


"""Bioactivity modelling."""

import json
from collections import Counter
from itertools import chain
from operator import itemgetter

import pandas as pd
import os

import ml2json
from rdkit import Chem
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, GroupKFold
from imblearn.over_sampling import SMOTENC
from papyrus_scripts.modelling import crossvalidate_model
from prodec import ProteinDescriptors, Transform, TransformType
from Mold2_pywrapper import Mold2
from tqdm.auto import tqdm

from preprocessing import merge_chembl_papyrus_mutants
from mutant_analysis_accession import filter_accession_data
from mutant_analysis_common_subsets import read_common_subset
from mutant_analysis_protein import calculate_average_residue_distance_to_ligand
from mutant_analysis_compounds import butina_cluster_compounds,visualize_molecular_subset_highlights
from mutant_analysis_clustermaps import extract_unique_connectivity,pivot_bioactivity_data,plot_bioactivity_heatmap,\
    plot_bioactivity_clustermap,extract_oldest_year
from mutant_analysis_type import read_mutation_distance_Epstein


def model_bioactivity_data(use_smote: bool = False):
    for path in ['../../data/pcm_models/', '../../data/qsar_all_models/', '../../data/qsar_separate_models/',
                 '../../data/pcm_models_smote/', '../../data/qsar_all_models_smote/',
                 '../../data/qsar_separate_models_smote/']:
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

    pbar = tqdm(sorted_accessions)

    for accession in pbar:
        # Determine names of all output files
        out_file_pcm_random_tsv = f'../../data/pcm_models/pcm_randomsplit_{accession}.tsv'
        out_file_pcm_random_json = f'../../data/pcm_models/pcm_randomsplit_{accession}_models.json'
        out_file_pcm_mutant_tsv = f'../../data/pcm_models/pcm_mutantsplit_{accession}.tsv'
        out_file_pcm_mutant_json = f'../../data/pcm_models/pcm_mutantsplit_{accession}_models.json'
        out_file_pcm_nomutant_random_tsv = f'../../data/pcm_models/pcm_nomutant_randomsplit_{accession}.tsv'
        out_file_pcm_nomutant_random_json = f'../../data/pcm_models/pcm_nomutant_randomsplit_{accession}_models.json'
        out_file_qsar_random_tsv = f'../../data/qsar_all_models/qsar_randomsplit_{accession}.tsv'
        out_file_qsar_random_json = f'../../data/qsar_all_models/qsar_randomsplit_{accession}_models.json'
        out_file_qsar_mutant_tsv = f'../../data/qsar_separate_models/qsar_mutantsplit_{accession}.tsv'
        out_file_qsar_mutant_json = f'../../data/qsar_separate_models/qsar_mutantsplit_{accession}_models.json'
        out_file_qsar_nomutant_random_tsv = f'../../data/qsar_all_models/qsar_nomutant_randomsplit_{accession}.tsv'
        out_file_qsar_nomutant_random_json = f'../../data/qsar_all_models/qsar_nomutant_randomsplit_{accession}_models.json'
        out_pcm_mutants = [out_file_pcm_random_tsv, out_file_pcm_random_json,
                           out_file_pcm_mutant_tsv, out_file_pcm_mutant_json]
        out_pcm_no_mutants = [out_file_pcm_nomutant_random_tsv, out_file_pcm_nomutant_random_json]
        out_qsar_mutants = [out_file_qsar_random_tsv, out_file_qsar_random_json,
                            out_file_qsar_mutant_tsv, out_file_qsar_mutant_json]
        out_qsar_no_mutants = [out_file_qsar_nomutant_random_tsv, out_file_qsar_nomutant_random_json]
        all_files = list(chain(out_pcm_mutants, out_pcm_no_mutants, out_qsar_mutants, out_qsar_no_mutants))
        if use_smote:
            out_file_pcm_random_smote_tsv = f'../../data/pcm_models_smote/pcm_randomsplit_smote_{accession}.tsv'
            out_file_pcm_random_smote_json = f'../../data/pcm_models_smote/pcm_randomsplit_smote_{accession}_models.json'
            out_file_pcm_mutant_smote_tsv = f'../../data/pcm_models_smote/pcm_mutantsplit_smote_{accession}.tsv'
            out_file_pcm_mutant_smote_json = f'../../data/pcm_models_smote/pcm_mutantsplit_smote_{accession}_models.json'
            out_file_pcm_nomutant_random_smote_tsv = f'../../data/pcm_models_smote/pcm_nomutant_smote_randomsplit_{accession}.tsv'
            out_file_pcm_nomutant_random_smote_json = f'../../data/pcm_models_smote/pcm_nomutant_smote_randomsplit_{accession}_models.json'
            out_file_qsar_random_smote_tsv = f'../../data/qsar_all_models_smote/qsar_randomsplit_smote_{accession}.tsv'
            out_file_qsar_random_smote_json = f'../../data/qsar_all_models_smote/qsar_randomsplit_smote_{accession}_models.json'
            out_file_qsar_mutant_smote_tsv = f'../../data/qsar_separate_models_smote/qsar_mutantsplit_smote_{accession}.tsv'
            out_file_qsar_mutant_smote_json = f'../../data/qsar_separate_models_smote/qsar_mutantsplit_smote_{accession}_models.json'
            out_file_qsar_nomutant_random_smote_tsv = f'../../data/qsar_all_models_smote/qsar_nomutant_smote_randomsplit_{accession}.tsv'
            out_file_qsar_nomutant_random_smote_json = f'../../data/qsar_all_models_smote/qsar_nomutant_smote_randomsplit_{accession}_models.json'
            out_pcm_mutants_smote = [out_file_pcm_random_smote_tsv, out_file_pcm_random_smote_json,
                                     out_file_pcm_mutant_smote_tsv, out_file_pcm_mutant_smote_json]
            out_pcm_no_mutants_smote = [out_file_pcm_nomutant_random_smote_tsv, out_file_pcm_nomutant_random_smote_json]
            out_qsar_mutants_smote = [out_file_qsar_random_smote_tsv, out_file_qsar_random_smote_json,
                                      out_file_qsar_mutant_smote_tsv, out_file_qsar_mutant_smote_json]
            out_qsar_no_mutants_smote = [out_file_qsar_nomutant_random_smote_tsv, out_file_qsar_nomutant_random_smote_json]
            all_smote_file = list(chain(out_pcm_mutants_smote, out_pcm_no_mutants_smote,
                                        out_qsar_mutants_smote, out_qsar_no_mutants_smote))
        # Skip if all files already exists
        if all(os.path.exists(f) for f in (chain(all_files, all_smote_file) if use_smote else all_files)):
            continue
        # Set description of progress bar
        pbar.desc = accession
        pbar.refresh()
        # Obtain data for this accession only
        accession_data = filter_accession_data(data, accession)
        mol_descs = None
        prot_descs = None
        # Run PCM if not any missing output file
        if not all(os.path.exists(f) for f in out_pcm_mutants):
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
                mols = accession_data[['connectivity', 'SMILES']].drop_duplicates()
                prots = accession_data[['target_id', 'sequence']].drop_duplicates()
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
                mols = accession_data[['connectivity', 'SMILES']].drop_duplicates()
                prots = accession_data[['target_id', 'sequence']].drop_duplicates()
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
                mols = accession_data[['connectivity', 'SMILES']].drop_duplicates()
                prots = accession_data[['target_id', 'sequence']].drop_duplicates()
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
        # Run SMOTE PCM if not any missing output file
        if use_smote and not all(os.path.exists(f) for f in out_pcm_mutants_smote):
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
                # Keep track of mutants ids
                pcm_target_ids = pcm_data.target_id
                ## SMOTE

                smote = SMOTENC(pcm_data)
                # Back to standard modelling
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
        # Run SMOTE QSAR if not missing any output file
        if use_smote and not all(os.path.exists(f) for f in out_qsar_mutants_smote):
            if mol_descs is None:  # PCM part was skipped
                # 1) Isolate IDs and dependent variable
                ids = accession_data[['connectivity', 'target_id', 'pchembl_value_Mean']]
                # 2) Keep unique mols and prots
                mols = accession_data[['connectivity', 'SMILES']].drop_duplicates()
                prots = accession_data[['target_id', 'sequence']].drop_duplicates()
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
        # Run SMOTE PCM with mutants removed
        if use_smote and not all(os.path.exists(f) for f in out_pcm_no_mutants_smote):
            if prot_descs is None:
                # 1) Isolate IDs and dependent variable
                ids = accession_data[['connectivity', 'target_id', 'pchembl_value_Mean']]
                # 2) Keep unique mols and prots
                mols = accession_data[['connectivity', 'SMILES']].drop_duplicates()
                prots = accession_data[['target_id', 'sequence']].drop_duplicates()
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
        # Run SMOTE QSAR with mutants removed
        if use_smote and not all(os.path.exists(f) for f in out_qsar_no_mutants_smote):
            if mol_descs is None:  # PCM part was skipped
                # 1) Isolate IDs and dependent variable
                ids = accession_data[['connectivity', 'target_id', 'pchembl_value_Mean']]
                # 2) Keep unique mols and prots
                mols = accession_data[['connectivity', 'SMILES']].drop_duplicates()
                prots = accession_data[['target_id', 'sequence']].drop_duplicates()
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
    model_bioactivity_data(True)