# -*- coding: utf-8 -*-


"""Mutant annotation."""
import pandas as pd

from preprocessing import obtain_chembl_data

def filter_assay_data(chembl_df: pd.DataFrame):
    """
    Filter from a dataframe with ChEMBL data the necessary columns for amino acid change annotation.
    :param chembl_df: DataFrame withe ChEMBL data of interest. It must contain columns of interest:
                    ['assay_id', 'description', 'accession', 'sequence']
    :return: DataFrame containing only columns of interest with duplicates dropped.
    """

def extract_aa_change(assays_df: pd.DataFrame):
    """
    Use regular expressions to extract potential amino acid changes from assay descriptions.
    :param assays_df: DataFrame containing (ChEMBL) assay descriptions in a column named 'description'
    :return: the input DataFrame with a new column 'aa_change' with the potential amino acid changes extracted in a list
    """

def define_aa_change_exceptions(assays_df_annotated: pd.DataFrame):
    """
    Query ChEMBL assay annotations to exclude False amino acid changes extracted using regular expressions.
    :param assays_df_annotated: DataFrame containing amino acid change annotations in a column named 'aa_change' and
                                wild type target sequences in a column named 'sequence'
    :return: the input DataFrame with a new column 'exception_flag' containing a list with the reasons to define the
            extracted aa_change values as exceptions.
    """
    # Fetch type of cells, organism, anything that is reported and match the extracted aa_change
    # Return type of data that raised the flag (e.g. cell_type)


def validate_aa_change(assays_df_annotated: pd.DataFrame, known_exceptions: str, automatic_exceptions: bool = True):
    """
    Validate amino acid (aa) changes extracted with regular expression by comparing the wild type aa to its position in
    the sequence.
    :param assays_df_annotated: DataFrame containing amino acid change annotations in a column named 'aa_change' and
                                wild type target sequences in a column named 'sequence'
    :param known_exceptions: Path to a dictionary containing known exceptions where the sequence validation would return
                            a false positive. Dictionary keys are targets annotated as 'accession' (Uniprot accession).
    :param automatic_exceptions: Call function `define_aa_change_exceptions` to automatically flag potential False
                                annotations.
    :return: the input DataFrame with a new column 'aa_change_validation' with a list of Booleans that defined whether
            the extracted aa change is validated by its position in the wild type sequence
    """

def create_target_id(assays_df_validated: pd.DataFrame, clean_df: bool = True):
    """
    Create column 'target_id' matching Papyrus dataset style based on annotated and validated 'aa_change' from assay
    descriptions in ChEMBL.
    :param assays_df_validated: DataFrame containing amino acid change annotations in a column named 'aa_change' and a
                                column 'aa_change_validation' with a list of Booleans that defined whether the extracted
                                aa change is validated by its position in the wild type sequence
    :param clean_df: Whether to strip the output DataFrame of annotation and validation columns, i.e.
                    ['aa_change', 'aa_change_validation', 'exception_flag']
    :return: the input DataFrame with a new column 'target_id' consisting of the target accession code followed by as
            many mutations separated by underscores as annotated and validated amino acid changes.
    """
    # Write df to file to later join to bioactivity data for modelling

if __name__ == '__main__':
    chembl_data = obtain_chembl_data(chembl_version='31', chunksize= 100_000)
    chembl_assays = filter_assay_data(chembl_data)
    chembl_assays_extracted = extract_aa_change(chembl_assays)
    chembl_assays_validated = validate_aa_change(chembl_assays_extracted, 'path_to_dict')
    chembl_assays_annotated = create_target_id(chembl_assays_validated)
