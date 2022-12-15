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
    """Obtain mutant data from ChEMBL using chembl-downloader.

    :param chembl_version: version of chembl to work with
    :param chunksize: size of chunks of data to be used (default: None)
    param data_folder: path to the folder in which the ChEMBL
    SQLite database is located or will be downloaded (default:
    pystow's default directory)
    """
    query = """ """
    chembl_data = chembl_downloader.query(query, version=chembl_version,
                                          prefix=[data_folder],
                                          chunksize=chunksize)
    return chembl_data


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