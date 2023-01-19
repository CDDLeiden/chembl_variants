# -*- coding: utf-8 -*-


"""Data sourcing and pre-processing."""

from typing import Union, Iterator

from tqdm.auto import tqdm
import papyrus_scripts
import chembl_downloader

import pandas as pd
import os
from pandas.io.parsers import TextFileReader as PandasTextFileReader
from papyrus_scripts.utils import IO as papyrusIO


def obtain_chembl_data(chembl_version: str, chunksize: int = None, data_folder: str = None):
    """Obtain assay descriptions and bioactivities annotated for mutants from ChEMBL using chembl-downloader.

    :param chembl_version: version of chembl to work with
    :param chunksize: size of chunks of data to be used (default: None)
    param data_folder: path to the folder in which the ChEMBL
    SQLite database is located or will be downloaded (default:
    pystow's default directory)
    """
    if data_folder is not None:
        os.environ['PYSTOW_HOME'] = data_folder

    chembl_file = '../../data/chembl_data.csv'
    if not os.path.isfile(chembl_file):

        query = """
            SELECT assays.description,assays.assay_id,assays.variant_id,assays.chembl_id,assays.assay_organism,
                variant_sequences.mutation,
                activities.activity_id,activities.pchembl_value,activities.standard_type,
                molecule_dictionary.chembl_id,compound_structures.canonical_smiles,
                component_sequences.accession,component_sequences.sequence,component_sequences.organism
            FROM assays
                LEFT JOIN activities USING (assay_id)
                LEFT JOIN variant_sequences USING (variant_id)
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
            """

        chembl_assays = chembl_downloader.query(query, version=chembl_version,
                                              prefix=['mutants-in-pcm', 'chembl'])#,
                                              # chunksize=chunksize)
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


def combine_chembl_papyrus_mutants(chembl_version: str, papyrus_version: str, papyrus_flavor: str, chunksize:int, predefined_variants: bool = False):
    """
    Combine datasets with ChEMBL and Papyrus mutants. Include also WT data for targets with at least one variant defined.
    Filter out only targets with no variants defined.
    :param chembl_mutants:
    :param papyrus_mutants:
    :return:
    """
    if predefined_variants:
        file_name = f'chembl{chembl_version}_papyrus{papyrus_version}{papyrus_flavor}_bioactivity_targets_with_mutants.txt'
    else:
        file_name = f'chembl{chembl_version}-annotated_papyrus{papyrus_version}{papyrus_flavor}_bioactivity_targets_with_mutants.txt'

    if not os.path.exists(file_name):
        from annotation import chembl_annotation
        # Get ChEMBL data and extract mutants
        chembl_annotated = chembl_annotation(chembl_version)
        chembl_annotated['source'] = f'ChEMBL{chembl_version}'
        chembl_with_mutants = _keep_targets_with_mutants(chembl_annotated, f'ChEMBL{chembl_version}', predefined_variants)
        # Rename columns so they match Papyrus
        dict_rename = {'chembl_id':'CID','assay_id':'AID','canonical_smiles':'SMILES'}
        # Keep common subset of columns
        chembl_with_mutants = chembl_with_mutants[['CID', 'target_id', 'AID', 'accession', 'pchembl_value_Mean', 'SMILES', 'sequence', 'source']]

        # Get Papyrus data with annotated mutants
        papyrus_with_mutants = obtain_papyrus_data(papyrus_version, papyrus_flavor, chunksize)
        papyrus_with_mutants['source'] = papyrus_with_mutants['source'].apply(lambda x: f'Papyrus{papyrus_version}_{x}')
        # TODO: add sequences from protein data file
        # TODO: filter chembl data from papyrus
        # Keep common subset of columns
        papyrus_with_mutants = papyrus_with_mutants[['CID', 'target_id', 'AID', 'accession', 'pchembl_value_Mean', 'SMILES', 'sequence', 'source']]

        # Concatenate chembl and papyrus data and write file

if __name__ == "__main__":
    combine_chembl_papyrus_mutants('31', '05.5', 'nostereo', 1_000_000)
