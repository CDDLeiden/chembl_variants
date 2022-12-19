# -*- coding: utf-8 -*-


"""Data sourcing and pre-processing."""

import os
import re
from typing import Optional, Union
from functools import partial
from itertools import chain

import pystow
import chembl_downloader
import papyrus_scripts

import pandas as pd
from rdkit import Chem
from pandas.io.parsers import TextFileReader as PandasTextFileReader


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
    :param flavor: flavor of the Papyrus dataset to work with
    :param chunksize: size of chunks of data to be used (default: None)
    param data_folder: path to the folder in which the Papyrus
    dataset is located or will be downloaded (default:
    pystow's default directory)
    :return: a Pandas dataframe if chunksize is None, a pandas.io.parsers.TextFileReader otherwise
    """
    # 1) Verify version is available online
    # 2) Verify version is available locally
    # 3) If not download
    # 4) Filter mutant data
    return


if __name__ == "__main__":
    chembl_data = obtain_chembl_data('31', 100_000)
    papyrus_data = obtain_papyrus_data('05.6', '++', 100_000)