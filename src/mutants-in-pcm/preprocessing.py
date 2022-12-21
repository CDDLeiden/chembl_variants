# -*- coding: utf-8 -*-


"""Data sourcing and pre-processing."""

from typing import Union, Iterator

from tqdm.auto import tqdm
import papyrus_scripts
import chembl_downloader

import pandas as pd
from pandas.io.parsers import TextFileReader as PandasTextFileReader
from papyrus_scripts.utils import IO as papyrusIO


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
    # Define arguments for parsing and download
    only_pp = flavor.endswith('pp')
    stereo = flavor.startswith('stereo')

    # 2) Ensure version is available locally
    # This either downloads or verifies file hashes
    papyrus_scripts.download_papyrus(data_folder, papyrus_version,
                                     nostereo=(not stereo), stereo=stereo, only_pp=only_pp,
                                     structures=False, descriptors='none', disk_margin=0)
    # Make sure version is correct (transform 'latest' to a version number)
    papyrus_version = papyrusIO.process_data_version(papyrus_version, data_folder)

    # 3) Filter mutant data
    papyrus_data = papyrus_scripts.read_papyrus(is3d=stereo, version=papyrus_version, plusplus=only_pp,
                                                chunksize=chunksize, source_path=data_folder)

    return _keep_papyrus_mutants(papyrus_data)


def _keep_papyrus_mutants(data: Union[PandasTextFileReader, pd.DataFrame]) -> Union[Iterator, pd.DataFrame]:
    """Keep Papyrus data related to mutant proteins only

    :param data: Papyrus data (raw or preprocessed)
    """
    if isinstance(data, (PandasTextFileReader, Iterator)):
        return _chunked_keep_papyrus_mutants(data)
    else:
        return data[data.Protein_Type != 'WT']


def _chunked_keep_papyrus_mutants(data: Union[PandasTextFileReader, Iterator]) -> Iterator:
    """Keep Papyrus data related to mutant proteins only

    :param data: Papyrus data (raw or preprocessed)
    """
    for chunk in data:
        yield _keep_papyrus_mutants(chunk)


if __name__ == "__main__":
    # chembl_data = obtain_chembl_data('31', 100_000)
    papyrus_data = obtain_papyrus_data('05.6', 'nostereo', 1_000_000)
    print(sum(len(chunk.index) for chunk in tqdm(papyrus_data)))
