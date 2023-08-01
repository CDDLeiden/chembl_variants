# -*- coding: utf-8 -*-


"""Data sourcing and pre-processing."""

from typing import Union, Iterator

from tqdm.auto import tqdm
import papyrus_scripts
import chembl_downloader
from rdkit.Chem import PandasTools
from rdkit import Chem

import pandas as pd
import os
from pandas.io.parsers import TextFileReader as PandasTextFileReader
from papyrus_scripts.utils import IO as papyrusIO

from data_path import get_data_path
data_dir = get_data_path()

def obtain_chembl_data(chembl_version: str, chunksize: int = None, data_folder: str = None):
    """Obtain assay descriptions and bioactivities annotated for mutants from ChEMBL using chembl-downloader.

    :param chembl_version: version of chembl to work with
    :param chunksize: size of chunks of data to be used (default: None)
    :param data_folder: path to the folder in which the ChEMBL
    SQLite database is located or will be downloaded (default:
    pystow's default directory)
    """
    if data_folder is not None:
        os.environ['PYSTOW_HOME'] = data_folder

    chembl_file = os.path.join(data_dir, 'chembl_data.csv')
    if not os.path.isfile(chembl_file):

        query = """
            SELECT assays.description,assays.assay_id,assays.variant_id,assays.chembl_id as 'assay_chembl_id',
                assays.assay_organism,
                docs.year,docs.abstract,
                variant_sequences.mutation,
                activities.activity_id,activities.pchembl_value,activities.standard_type,activities.activity_comment,
                molecule_dictionary.chembl_id,compound_structures.canonical_smiles,
                component_sequences.accession,component_sequences.sequence,component_sequences.organism
            FROM assays
                LEFT JOIN activities USING (assay_id)
                LEFT JOIN variant_sequences USING (variant_id)
                LEFT JOIN docs USING (doc_id)
                INNER JOIN molecule_dictionary
                    ON activities.molregno = molecule_dictionary.molregno
                INNER JOIN compound_structures
                    ON molecule_dictionary.molregno = compound_structures.molregno
                INNER JOIN target_dictionary
                        ON assays.tid = target_dictionary.tid
                INNER JOIN target_components
                    ON target_dictionary.tid = target_components.tid
                INNER JOIN component_sequences
                    ON target_components.component_id = component_sequences.component_id
            WHERE
                activities.standard_relation = '='
            """

        chembl_assays = chembl_downloader.query(query, version=chembl_version,
                                              prefix=['mutants-in-pcm', 'chembl'])

        chembl_assays.to_csv(chembl_file, sep='\t', index=None)

    else:
        chembl_assays = pd.read_csv(chembl_file, sep='\t')

    return chembl_assays


def obtain_papyrus_data(papyrus_version: str, flavor: str, chunksize: int = None, data_folder: str = None) -> Union[pd.DataFrame, PandasTextFileReader]:
    """Obtain mutant data from Papyrus using Papyrus-scripts.

    :param papyrus_version: version of chembl to work with
    :param flavor: flavor of the Papyrus dataset to work with; one of {nostereo_pp, nostereo, stereo}
    :param chunksize: size of chunks of data to be used (default: None)
    param data_folder: path to the folder in which the Papyrus
    dataset is located or will be downloaded (default:
    pystow's default directory)
    :return: a Pandas dataframe if chunksize is None, a pandas.io.parsers.TextFileReader otherwise
    """
    # 1) Verify version is available online
    online_versions = papyrusIO.get_online_versions()
    if papyrus_version not in online_versions:
        raise ValueError(f'Papyrus version must be one of [{", ".join(online_versions)}]')

    # Verify flavor
    flavor = flavor.lower()
    if flavor not in ['nostereo_pp', 'nostereo', 'stereo']:
        raise ValueError('Flavor of the Papyrus data must be one of {\'nostereo_pp\', \'nostereo\', \'stereo\'}')

    # Validate version number
    papyrus_version = papyrusIO.process_data_version(papyrus_version, data_folder)

    # 2) Identify local files
    papyrus_files = papyrus_scripts.IO.get_downloaded_papyrus_files(data_folder)
    # Filter for files of interest
    papyrus_files = papyrus_files[papyrus_files.short_name.isin(
        ['papyrus++', '2D_papyrus', '3D_papyrus']) & (papyrus_files.version == papyrus_version)]

    # Download files if not available
    if flavor == 'nostereo_pp' and not papyrus_files[papyrus_files.short_name == 'papyrus++'].downloaded.item():
        papyrus_scripts.download_papyrus(data_folder, papyrus_version,
                                         nostereo=True, stereo=False, only_pp=True,
                                         structures=False, descriptors='none', disk_margin=0, progress=True)
    elif flavor == 'nostereo' and not papyrus_files[papyrus_files.short_name == '2D_papyrus'].downloaded.item():
        papyrus_scripts.download_papyrus(data_folder, papyrus_version,
                                         nostereo=True, stereo=False, only_pp=False,
                                         structures=False, descriptors='none', disk_margin=0, progress=True)
    elif flavor == 'stereo' and not papyrus_files[papyrus_files.short_name == '3D_papyrus'].downloaded.item():
        papyrus_scripts.download_papyrus(data_folder, papyrus_version,
                                         nostereo=False, stereo=True, only_pp=False,
                                         structures=False, descriptors='none', disk_margin=0, progress=True)

    # 3) Filter mutant data
    only_pp = flavor.endswith('pp')
    stereo = flavor.startswith('stereo')

    papyrus_data = papyrus_scripts.read_papyrus(is3d=stereo, version=papyrus_version, plusplus=only_pp,
                                                chunksize=chunksize, source_path=data_folder)

    return _keep_targets_with_mutants(papyrus_data, source=f'Papyrus{papyrus_version}')


def _keep_targets_with_mutants(data: pd.DataFrame, source:str, predefined_variants: bool = False):
    """Keep bioactivity data related to proteins with at least one mutant defined
    :param data: ChEMBL data (raw or preprocessed)
    :param predefined_variants: whether to use ChEMBL pre-defined variants (True)
                                or use annotated variants (False)
    """
    if isinstance(data, (PandasTextFileReader, Iterator)):
        generator_with_mutants = _chunked_keep_papyrus_mutants(data,source)
        data_with_mutants = pd.concat(generator_with_mutants, ignore_index=True)

    else:
        if predefined_variants and ('Papyrus' not in source):
            def revert_annotated_mutation(x):
                if not isinstance(x['mutation'], str):
                    target_id = f"{x['accession']}_WT"
                else:
                    target_id = x['target_id']
                return target_id

            # Revert annotated sequence to WT sequence
            from annotation import mutate_sequence
            data['sequence'] = mutate_sequence(data, 'sequence', 'target_id', revert = True)
            # Revert target_id annotation, keeping only pre-defined variants in ChEMBL
            data['target_id'] = data.apply(revert_annotated_mutation, axis=1)

        # Count number of variants per accession
        variants_count = data.drop_duplicates(subset=['target_id']).groupby(['accession'])[
            'target_id'].count().to_dict()
        # Keep data for targets with variants defined otehr than WT
        targets_with_mutants = [accession for accession, num_variants in variants_count.items() if num_variants > 1]
        data_with_mutants = data[data['accession'].isin(targets_with_mutants)]

    return data_with_mutants

def _chunked_keep_papyrus_mutants(data: Union[PandasTextFileReader, Iterator], source:str) -> Iterator:
    """Keep Papyrus data related to mutant proteins only

    :param data: Papyrus data (raw or preprocessed)
    """
    for chunk in data:
        yield _keep_targets_with_mutants(chunk,source)


def combine_chembl_papyrus_mutants(chembl_version: str, papyrus_version: str, papyrus_flavor: str, chunksize:int,
                                   annotation_round:int, predefined_variants: bool = False):
    """
    Combine datasets with ChEMBL and Papyrus mutants. Include also WT data for targets with at least one variant defined.
    Filter out only targets with no variants defined.
    :param chembl_version: ChEMBL version
    :param papyrus_version: Papyrus version
    :param papyrus_flavor: Papyrus flavor (nostereo_pp, nostereo, stereo)
    :param chunksize: number of rows to read at a time
    :param annotation_round: round of annotation following further curation
    :param predefined_variants: whether to use ChEMBL pre-defined variants
    """
    if predefined_variants:
        file_name = os.path.join(data_dir,f'chembl{chembl_version}_papyrus{papyrus_version}' \
                    f'{papyrus_flavor}_data_with_mutants_round{annotation_round}.csv')
    else:
        file_name = os.path.join(data_dir,f'chembl{chembl_version}-annotated_papyrus{papyrus_version}' \
                    f'{papyrus_flavor}_data_with_mutants_round{annotation_round}.csv')

    if not os.path.exists(file_name):
        from annotation import chembl_annotation
        # Get ChEMBL data and extract mutants
        chembl_annotated = chembl_annotation(chembl_version, annotation_round)
        chembl_annotated['source'] = f'ChEMBL{chembl_version}'
        chembl_with_mutants = _keep_targets_with_mutants(chembl_annotated, f'ChEMBL{chembl_version}', predefined_variants)
        # Rename columns so they match Papyrus
        dict_rename = {'chembl_id':'CID','assay_id':'AID','canonical_smiles':'SMILES','year':'Year'}
        chembl_with_mutants.rename(dict_rename, axis=1, inplace=True)
        # Add connectivity ID to identify compounds
        PandasTools.AddMoleculeColumnToFrame(chembl_with_mutants, 'SMILES', 'Molecule', includeFingerprints=False)
        chembl_with_mutants['connectivity'] = chembl_with_mutants['Molecule'].apply(lambda x: Chem.MolToInchiKey(x).split('-')[0])

        # Keep common subset of columns
        chembl_with_mutants = chembl_with_mutants[['CID', 'connectivity', 'target_id', 'AID', 'accession',
                                                   'pchembl_value_Mean', 'SMILES', 'sequence', 'source',
                                                   'Activity_class', 'Year']]

        # Get Papyrus data with annotated mutants
        papyrus_with_mutants = obtain_papyrus_data(papyrus_version, papyrus_flavor, chunksize)
        papyrus_with_mutants['source'] = papyrus_with_mutants['source'].apply(lambda x: f'Papyrus{papyrus_version}_{x}')
        # Make sure that all mutations in a variant are ordered by position
        papyrus_with_mutants['target_id'] = papyrus_with_mutants['target_id'].apply(
            lambda x: f"{x.split('_')[0]}_{'_'.join(sorted(x.split('_')[1:], key=lambda y: y[1:-1], reverse=False))}")
        # Add sequence data from protein data file
        sequences = papyrus_scripts.read_protein_set(version=papyrus_version)[['target_id', 'Sequence']].rename(
            {'Sequence': 'sequence'}, axis=1)
        papyrus_with_mutants = pd.merge(papyrus_with_mutants, sequences, on='target_id')
        # Filter out all ChEMBL related data from Papyrus set to avoid duplicates
        papyrus_with_mutants_nochembl = papyrus_with_mutants[~papyrus_with_mutants['source'].str.contains('ChEMBL')]
        # Keep common subset of columns
        papyrus_with_mutants_nochembl = papyrus_with_mutants_nochembl[['CID', 'connectivity','target_id', 'AID',
                                                                       'accession', 'pchembl_value_Mean', 'SMILES',
                                                                       'sequence', 'source', 'Activity_class', 'Year']]

        # Concatenate ChEMBL and papyrus data and write file
        chembl_papyrus_with_mutants = pd.concat((chembl_with_mutants, papyrus_with_mutants_nochembl), axis=0, ignore_index=True)

        chembl_papyrus_with_mutants.to_csv(file_name, sep='\t', index=False)

    else:
        chembl_papyrus_with_mutants = pd.read_csv(file_name, sep='\t')

    return chembl_papyrus_with_mutants

def annotate_uniprot_metadata(data: pd.DataFrame, papyrus_version: str):
    """
    Add Uniprot metadata to bioactivity dataset (Organism, gene name)
    :param data: bioactivity dataset with accession column
    :param papyrus_version: Papyrus version
    :return: pd.DataFrame annotated
    """
    # Read Papyrus protein dataset
    papyrus_proteins = papyrus_scripts.read_protein_set(version=papyrus_version)

    # Map Uniprot metadata based on accession (not target_id because not all mutants are annotated in Papyrus)
    papyrus_proteins['accession'] = papyrus_proteins['target_id'].apply(lambda x: x.split('_')[0])
    papyrus_proteins.drop_duplicates(subset='accession', inplace=True)
    mapping_df = papyrus_proteins[['accession', 'UniProtID', 'Organism', 'HGNC_symbol']]

    data_mapped = data.merge(mapping_df, how='left', on='accession')

    return data_mapped

def calculate_mean_activity_chembl_papyrus(data: pd.DataFrame):
    """
    From a dataset with concatenated ChEMBL and Papyrus entries, compute mean pchembl_value for the same target_id-connectivity pair
    :param data: DataFrame with activity data
    :return: DataFrame with unique activity datapoints per target_id - connectivity pair
    """
    def agg_functions_variant_connectivity(x):
        d ={}
        d['pchembl_value_Mean'] = x['pchembl_value_Mean'].mean()
        d['Activity_class_consensus'] = x['Activity_class'].mode()
        d['source'] = ';'.join(list(set(list(x['source']))))
        d['SMILES'] = list(x['SMILES'])[0]
        d['CID'] = list(x['CID'])[0]
        d['accession'] = list(x['accession'])[0]
        d['sequence'] = list(x['sequence'])[0]
        d['Year'] = min(x['Year']) # Keep first year when the compound was tested

        return pd.Series(d, index=['pchembl_value_Mean', 'Activity_class_consensus', 'source', 'SMILES', 'CID',
                                   'accession', 'sequence', 'Year'])

    agg_activity_data = data.groupby(['target_id','connectivity'], as_index=False).apply(agg_functions_variant_connectivity)

    return agg_activity_data

def merge_chembl_papyrus_mutants(chembl_version: str, papyrus_version: str, papyrus_flavor: str, chunksize:int,
                                 annotation_round: int,predefined_variants: bool = False):
    """
    Create a dataset with targets with at least one annotated variant from ChEMBL and Papyrus. Merge datasets for
    connectivity-target_id pairs if data available from both sources.
    :param chembl_version: ChEMBL version
    :param papyrus_version: Papyrus version
    :param papyrus_flavor: Papyrus flavor (nostereo_pp, nostereo, stereo)
    :param chunksize: number of rows to read at a time
    :param annotation_round: round of annotation following further curation
    :param predefined_variants: whether to use ChEMBL pre-defined variants
    """
    if predefined_variants:
        file_name = os.path.join(data_dir,f'merged_chembl{chembl_version}_papyrus{papyrus_version}' \
                    f'{papyrus_flavor}_data_with_mutants_round{annotation_round}.csv')
    else:
        file_name = os.path.join(data_dir,f'merged_chembl{chembl_version}-annotated_papyrus{papyrus_version}' \
                    f'{papyrus_flavor}_data_with_mutants_round{annotation_round}.csv')

    if not os.path.exists(file_name):
        chembl_papyrus_with_mutants = combine_chembl_papyrus_mutants(chembl_version, papyrus_version, papyrus_flavor,
                                                                     chunksize, annotation_round, predefined_variants)

        agg_activity_data_not_annotated = calculate_mean_activity_chembl_papyrus(chembl_papyrus_with_mutants)

        # Annotate Uniprot metadata
        agg_activity_data = annotate_uniprot_metadata(agg_activity_data_not_annotated, papyrus_version)

        agg_activity_data.to_csv(file_name, sep='\t', index=False)

    else:
        agg_activity_data = pd.read_csv(file_name, sep='\t')

    return agg_activity_data

if __name__ == "__main__":
    merge_chembl_papyrus_mutants('31', '05.5', 'nostereo', 1_000_000, annotation_round=1, predefined_variants=False)
