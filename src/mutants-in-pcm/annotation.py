# -*- coding: utf-8 -*-


"""Mutant annotation."""
import json

import pandas as pd
import numpy as np
import re
import chembl_downloader

from preprocessing import obtain_chembl_data

def filter_assay_data(chembl_df: pd.DataFrame):
    """
    Filter from a dataframe with ChEMBL data the necessary columns for amino acid change annotation.
    :param chembl_df: DataFrame with ChEMBL data of interest. It must contain columns of interest:
                    ['assay_id', 'description', 'accession', 'sequence', 'mutation']
    :return: DataFrame containing only columns of interest with duplicates dropped.
    """
    assay_df = chembl_df[['assay_id', 'description', 'accession', 'sequence', 'mutation']]
    # Drop duplicates inherited from assay-activity link
    assay_df = assay_df.drop_duplicates(subset=['assay_id', 'accession'])

    return assay_df


def extract_aa_change(assays_df: pd.DataFrame):
    """
    Use regular expressions to extract potential amino acid changes from assay descriptions.
    :param assays_df: DataFrame containing (ChEMBL) assay descriptions in a column named 'description'
    :return: the input DataFrame with a new column 'aa_change' with the potential amino acid changes extracted in a list
    """
    # Extract aa_change in format W123M or Trp123Met
    regex_expression = '[A-Z][a-z]{2,}\d+[A-Z][a-z]{2,}|[A-Z]\d+[A-Z]'
    assays_df_extracted = assays_df.copy(deep=True)
    assays_df_extracted['aa_change'] = assays_df_extracted.description.str.findall(regex_expression)

    # Define dictionary for three-letter aa code to one-letter aa code
    aas = {'Ala': 'A',
           'Gly': 'G',
           'Ile': 'I',
           'Leu': 'L',
           'Pro': 'P',
           'Val': 'V',
           'Phe': 'F',
           'Trp': 'W',
           'Tyr': 'Y',
           'Asp': 'D',
           'Glu': 'E',
           'Arg': 'R',
           'His': 'H',
           'Lys': 'K',
           'Ser': 'S',
           'Thr': 'T',
           'Cys': 'C',
           'Met': 'M',
           'Asn': 'N',
           'Gln': 'Q',
           'Sec': 'U',  # Selenocysteine,
           'Pyl': 'O',  # Pyrrolysine
           'Asx': 'B',  # Aspartic acid or Asparagine
           'Glx': 'Z',  # Glutamic acid or Glutamine
           'Xaa': 'X',  # Any amino acid
           'Xle': 'J'  # Leucine or Isoleucine
           }

    # Convert three-letter codes to one-letter codes
    assays_df_extracted['aa_change'] = assays_df_extracted['aa_change'].map(lambda x: ','.join(x))
    assays_df_extracted['aa_change'] = assays_df_extracted.aa_change.replace(aas, regex=True)

    # Convert aa_change column back to list
    assays_df_extracted['aa_change'] = assays_df_extracted['aa_change'].map(lambda x: x.split(sep=','))

    # Remove wrongly assigned empty strings in amino acid lists
    assays_df_extracted['aa_change'] = assays_df_extracted['aa_change'].apply(lambda x: [i for i in x if i != ""])

    return assays_df_extracted
    
    
def define_aa_change_exceptions(assays_df_extracted: pd.DataFrame, chembl_version: str):
    """
    Query ChEMBL assay annotations to exclude False amino acid changes extracted using regular expressions.
    :param assays_df_extracted: DataFrame containing amino acid change annotations in a column named 'aa_change' and
                                wild type target sequences in a column named 'sequence'
    :param chembl_version: version of ChEMBL to query
    :return: the input DataFrame with a new column 'exception_flag' containing a list with the reasons to define the
            extracted aa_change values as exceptions.
    """

    # Extract from ChEMBL information about assays that could be confused with a mutation
    query = """
            SELECT assays.assay_id,assays.assay_cell_type,
                component_synonyms.component_synonym,
                cell_dictionary.cell_name,
                target_dictionary.pref_name
            FROM assays
                INNER JOIN target_dictionary
                    ON assays.tid = target_dictionary.tid
                INNER JOIN target_components
                    ON target_dictionary.tid = target_components.tid
                INNER JOIN component_sequences
                    ON target_components.component_id = component_sequences.component_id
                LEFT JOIN component_synonyms
                    ON component_sequences.component_id = component_synonyms.component_id
                LEFT JOIN cell_dictionary
                    ON assays.cell_id = cell_dictionary.cell_id
            """

    chembl_assay_information = chembl_downloader.query(query, version=chembl_version,
                                                       prefix=['mutants-in-pcm', 'chembl'])

    # Keep one entry per assay id by aggregating and minimizing the information
    agg_functions = {'assay_cell_type': list, 'component_synonym': list, 'cell_name': list, 'pref_name': list}
    chembl_assay_information_grouped = chembl_assay_information.groupby(chembl_assay_information['assay_id'],
                                                                        as_index=False).aggregate(agg_functions)
    chembl_assay_information_grouped['assay_cell_type'] = chembl_assay_information_grouped['assay_cell_type'].apply(
        lambda x: list(set([i for i in x if i is not None])))
    chembl_assay_information_grouped['component_synonym'] = chembl_assay_information_grouped['component_synonym'].apply(
        lambda x: list(set([i for i in x if i is not None])))
    chembl_assay_information_grouped['cell_name'] = chembl_assay_information_grouped['cell_name'].apply(
        lambda x: list(set([i for i in x if i is not None])))
    chembl_assay_information_grouped['pref_name'] = chembl_assay_information_grouped['pref_name'].apply(
        lambda x: list(set([i for i in x if i is not None])))

    # Flag assays in input set with additional assay information
    assays_df_exceptions = pd.merge(assays_df_extracted, chembl_assay_information_grouped, how='left', on='assay_id')

    # Look for matches between extracted aa_changed and assay information
    def flag_regex_exceptions(row):
        flags = [False for i in row['aa_change']]
        reasons = [None for i in row['aa_change']]

        for i, aa_change in enumerate(row['aa_change']):
            for col in ['assay_cell_type', 'component_synonym', 'cell_name', 'pref_name']:
                match = [s for s in row[col] if aa_change.upper() in s.upper()]
                if len(match) > 0:
                    flags[i] = True
                    reasons[i] = col

        return flags, reasons

    assays_df_exceptions[['exception_flags', 'exception_reasons']] = assays_df_exceptions.apply(flag_regex_exceptions,
                                                                                                  axis=1,
                                                                                                  result_type='expand')
    # Drop information columns
    assays_df_exceptions.drop(['assay_cell_type', 'component_synonym', 'cell_name', 'pref_name'], axis=1, inplace=True)

    return assays_df_exceptions


def validate_aa_change(assays_df_extracted: pd.DataFrame,
                       known_exceptions: str = '../../data/known_regex_exceptions.json',
                       automatic_exceptions: bool = True,
                       clean_df: bool = True,
                       **kwargs):
    """
    Validate amino acid (aa) changes extracted with regular expression by comparing the wild type aa to its position in
    the sequence.
    :param assays_df_extracted: DataFrame containing amino acid change annotations in a column named 'aa_change' and
                                wild type target sequences in a column named 'sequence'
    :param known_exceptions: Path to a dictionary containing known exceptions where the sequence validation would return
                            a false positive. Dictionary keys are targets annotated as 'accession' (Uniprot accession).
    :param automatic_exceptions: Call function `define_aa_change_exceptions` to automatically flag potential False
                                annotations.
    :param clean_df: Whether to strip the output DataFrame of annotation and validation columns, i.e.
                    ['exception_flags', 'exception_reasons', 'seq_flags', 'seq_flags_fixed', 'aa_change_val1']
    :return: the input DataFrame with a new column 'mutants' with a list of validated extracted aa changes
    """
    # Define exceptions based on additional assay information from ChEMBL query
    if automatic_exceptions:
        try:
            chembl_version = kwargs['chembl_version']
        except:
            print('Kwarg "chembl_version" needed for automatic exception check')

        assays_df_exceptions = define_aa_change_exceptions(assays_df_extracted, chembl_version)
    else:
        assays_df_exceptions = assays_df_extracted.copy(deep=True)
        assays_df_exceptions['exception_flags'] = assays_df_exceptions['aa_change'].apply(lambda x: [False for i in x])
        assays_df_exceptions['exception_reasons'] = assays_df_exceptions['aa_change'].apply(lambda x: [None for i in x])


    # Validation step 1: Filter out extracted mutations if there is an exception found
    assays_df_validation1 = assays_df_exceptions.copy(deep=True)
    assays_df_validation1['aa_change_val1'] = \
        assays_df_validation1.apply(lambda x: [i for i,n in zip(x['aa_change'], x['exception_flags']) if n == False], axis=1)

    # Define validation flags based on position of wild type amino acid in the sequence
    def sequence_validation(row,aa_change_col,sequence_col):
        flags = [False for i in row[aa_change_col]]

        for i, aa_change in enumerate(row[aa_change_col]):
            aa_wt = re.search('[A-Z]', aa_change).group(0)
            res = int(re.search('\d+', aa_change).group(0))

            if res < len(str(row[sequence_col])):
                if str(row[sequence_col])[res - 1] == aa_wt:  # residue number 1-based; while python index 0-based
                    flags[i] = True

        return flags

    assays_df_validation2 = assays_df_validation1.copy(deep=True)
    assays_df_validation2['seq_flags'] = assays_df_validation2.apply(sequence_validation,
                                                                     args=('aa_change_val1','sequence'), axis=1)

    # Define exception (rescue) flags from file of known exceptions
    if known_exceptions is not None:
        with open(known_exceptions) as json_file:
            dict_known_exceptions = json.load(json_file)
            false_pos = dict_known_exceptions['false_positive']
            false_neg = dict_known_exceptions['false_negative']

        # Re-define sequence validation flags based on known exceptions and rescues
        def redefine_seq_flags(row):
            flags_fixed = row['seq_flags'][:]

            for i, mut in enumerate(row['aa_change_val1']):
                # Revert False positives based on accession
                if row['accession'] in false_pos['accession'].keys():
                    if mut in false_pos['accession'][row['accession']]:
                        flags_fixed[i] = False
                # Revert False positives based on assay_id
                if str(row['assay_id']) in false_pos['assay_id'].keys():
                    if mut in false_pos['assay_id'][str(row['assay_id'])]:
                        flags_fixed[i] = False
                # Revert False positives based on assay description matches
                description_match = [n for n, s in enumerate(false_pos['description'].keys()) if
                                     s in row['description']]
                if bool(description_match):
                    if mut in list(false_pos['description'].values())[description_match[0]]:
                        flags_fixed[i] = False

                # Revert False negatives based on accession
                if row['accession'] in false_neg['accession'].keys():
                    if mut in false_neg['accession'][row['accession']]:
                        flags_fixed[i] = True
                # Revert False negatives based on assay_id
                if str(row['assay_id']) in false_neg['assay_id'].keys():
                    if mut in false_neg['assay_id'][str(row['assay_id'])]:
                        flags_fixed[i] = True
                # Revert False negatives based on assay description matches
                description_match = [n for n, s in enumerate(false_neg['description'].keys()) if
                                     s in row['description']]
                if bool(description_match):
                    if mut in list(false_neg['description'].values())[description_match[0]]:
                        flags_fixed[i] = True

            return flags_fixed

        assays_df_validation2['seq_flags_fixed'] = assays_df_validation2.apply(redefine_seq_flags, axis=1)

    else:
        assays_df_validation2['seq_flags_fixed'] = assays_df_validation2['seq_flags']

    # Validation step 2: Filter out mutations where wild type amino acid does not match at the right location
    assays_df_validation2['aa_change_val2'] = \
        assays_df_validation2.apply(lambda x: [i for i, n in zip(x['aa_change_val1'], x['seq_flags_fixed'])
                                               if n == True], axis=1)

    # Remove columns created for validation purposes and make validated changes a new variable called 'mutants'
    assays_df_validated = assays_df_validation2.copy(deep=True)
    if clean_df:
        assays_df_validated.drop(['exception_flags', 'exception_reasons', 'seq_flags', 'seq_flags_fixed', 'aa_change_val1'],
                                  axis=1, inplace=True)
    assays_df_validated.rename(columns={"aa_change_val2":"mutants"}, inplace=True)

    return assays_df_validated



def create_papyrus_columns(assays_df_validated: pd.DataFrame):
    """
    Create column 'target_id' and 'Protein_Type' matching Papyrus dataset style based on annotated and validated
    'aa_change' from assay descriptions in ChEMBL.
    :param assays_df_validated: DataFrame containing amino acid change annotations in a column named 'aa_change' and a
                                column 'aa_change_validation' with a list of Booleans that defined whether the extracted
                                aa change is validated by its position in the wild type sequence
    :return: the input DataFrame with two new columns: 'target_id' consisting of the target accession code followed by
            '_WT or 'as many mutations separated by underscores as annotated and validated amino acid changes; and
            'Protein_Type', where WT or mutants are defined.
    """
    # Remove duplicated extracted mutations (sometimes multiple times in the assay description) and order by residue
    assays_df_validated['mutants'] = assays_df_validated['mutants'].apply(lambda x: list(set(x)))

    def num_sort(mut):
        return list(map(int, re.findall(r'\d+', str(mut))))[0]

    assays_df_validated['mutants'] = assays_df_validated['mutants'].apply(lambda x: sorted(x, key=num_sort))

    # Make target_id identifier
    assays_df_validated['target_id'] = assays_df_validated.apply(
        lambda x: f'{x["accession"]}_{"_".join(x["mutants"])}' if len(x["mutants"]) > 0 else f'{x["accession"]}_WT',
        axis=1)

    # Make Protein_Type column
    assays_df_validated['Protein_Type'] = assays_df_validated.apply(
        lambda x: ";".join(x["mutants"]) if len(x["mutants"]) > 0 else 'WT', axis=1)

    return assays_df_validated

def mutate_sequence(df: pd.DataFrame, sequence_col: str, target_id_col: str, revert: bool = False):
    """
    Replaces the column with the target sequence with the mutant sequence based on mutations defined in the target_id
    :param df: DataFrame containing a column with sequences and another with target IDs annotated with mutations
    :param sequence_col: name of the column containing sequences
    :param target_id_col: name of the column containing the annotated target IDs
    :param revert: if True, mutant sequence is reverted to WT
    :return: the input dataframe with a modified sequence column with mutations introduced or reverted
    """
    def replace_aa(row):
        sequence = row[sequence_col]
        for mut in row[target_id_col].split('_')[1:]:
            if mut != 'WT':
                aa_wt = mut[0]
                aa_mut = mut[-1]
                res = int(re.search('\d+', mut).group(0))
                index = res -1 # residue number 1-based; while python index 0-based
                if not revert:
                    sequence = sequence[:index] + aa_mut + sequence[index + 1:]
                else:
                    sequence = sequence[:index] + aa_wt + sequence[index + 1:]

        return sequence

    df[sequence_col] = df.apply(replace_aa, axis=1)

    return df

def map_activity_mutations(chembl_df: pd.DataFrame, assays_df_validated: pd.DataFrame):
    """
    Join mutation annotations to dataframe with ChEMBL bioactivity for modelling. Aggregate activity values for the same
    chembl_id-target_id pair by calculating the mean pchembl_value.
    :param chembl_df: DataFrame with ChEMBL bioactivity data of interest. It must contain columns of interest:
                    ['assay_id', 'accession', 'pchembl_value', 'chembl_id', 'canonical_smiles']
    :param assays_df_validated:DataFrame with annotated and validated mutations from assay descriptions. Must contain:
                                ['assay_id', 'accession', 'target_id', 'sequence']
    :return: DataFrame with one row per chembl_id-target_id pair with an aggregated pchembl_value_Mean
    """
    # Mutate sequences based on extracted mutations
    assays_df_validated = mutate_sequence(assays_df_validated, 'sequence', 'target_id')

    # Keep columns of interest before joining dataframes
    chembl_df = chembl_df[['assay_id', 'accession', 'pchembl_value', 'chembl_id', 'canonical_smiles']]
    assays_df_validated = assays_df_validated[['assay_id', 'accession', 'target_id', 'sequence']]

    # Map mutations to bioactivity entries based on assay_id and accession
    chembl_mutations_df = pd.merge(chembl_df, assays_df_validated, how='left', on=['assay_id', 'accession'])

    # Keep only assays with pchembl value defined
    chembl_bioactivity_df = chembl_mutations_df[chembl_mutations_df['pchembl_value'].notna()]

    # Aggregate bioactivity per compound-target(mutant) pair. Calculate mean pchembl_value if needed
    def agg_functions(x):
        d = {}
        d['assay_id'] = list(x['assay_id'])
        d['accession'] = x['accession'].iloc[0]
        d['pchembl_value'] = list(x['pchembl_value'])
        d['pchembl_value_Mean'] = x['pchembl_value'].mean()
        d['canonical_smiles'] = x['canonical_smiles'].iloc[0],
        d['sequence'] = x['sequence'].iloc[0]
        return pd.Series(d, index=['assay_id', 'accession', 'pchembl_value', 'pchembl_value_Mean', 'canonical_smiles',
                                   'sequence'])

    chembl_bioactivity_agg_df = chembl_mutations_df.groupby(['chembl_id', 'target_id'], as_index=False).apply(
        agg_functions)

    return chembl_bioactivity_agg_df


def annotation(chembl_version: str):
    """
    Obtain ChEMBL bioactivity data and annotate for validated mutants. If multiple assays are available per mutant-compound pair,
    calculate mean pchembl value.
    :param chembl_version: Version of ChEMBL to obtain data from
    :return: pd.DataFrame with one entry per target_id (mutant) - chembl_id (compound) with mean pchembl value
    """
    chembl_data = obtain_chembl_data(chembl_version='31')
    chembl_assays = filter_assay_data(chembl_data)
    chembl_assays_extracted = extract_aa_change(chembl_assays)
    chembl_assays_validated = validate_aa_change(chembl_assays_extracted, chembl_version='31')
    chembl_assays_annotated = create_papyrus_columns(chembl_assays_validated)
    chembl_bioactivity_dataset = map_activity_mutations(chembl_data, chembl_assays_annotated)

    return chembl_bioactivity_dataset
