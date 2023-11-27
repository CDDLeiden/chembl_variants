# -*- coding: utf-8 -*-


"""Second round of mutant annotation comparing to ChEMBL annotation."""

import numpy as np
import pandas as pd
import ast

from mutants_in_pcm.preprocessing import obtain_chembl_data,combine_chembl_papyrus_mutants,merge_chembl_papyrus_mutants
from mutants_in_pcm.annotation import *

from .data_path import get_data_path

def read_assay_annotations(chembl_version: str, annotation_round: int):
    """Read the annotation results for a particular round of annotations.

    :param chembl_version: ChEMBL version
    :param annotation_round: annotation round
    :return: assay annotations
    """
    data_dir = get_data_path()

    # Read round 1 annotated assays making sure that columns containing lists are read as lists
    def convert_to_list(s):
        return ast.literal_eval(s)

    converters = {'aa_change': convert_to_list,
                  'mutants': convert_to_list}

    assay_data = pd.read_csv(
        os.path.join(data_dir, f'chembl{chembl_version}_annotated_assays_round{annotation_round}.csv'),
        sep='\t', converters=converters)

    return assay_data

def get_assay_info(assay_data:pd.DataFrame, chembl_version: str):
    """
    Get additional assay information for assays to check where differences in local
    and ChEMBL annotations might come from
    """
    # Extract from ChEMBL information about assays
    query = """
            SELECT *
            FROM assays
                INNER JOIN target_dictionary
                    ON assays.tid = target_dictionary.tid
                INNER JOIN target_components
                    ON target_dictionary.tid = target_components.tid
                INNER JOIN component_sequences
                    ON target_components.component_id = component_sequences.component_id
            """

    chembl_assay_information = chembl_downloader.query(query, version=chembl_version,
                                                       prefix=['mutants-in-pcm', 'chembl'])

    # Merge assay information to annotated assays
    assay_data_ExtraInfo = pd.merge(assay_data, chembl_assay_information, how='left',
                                    on=['assay_id','accession'])

    return assay_data_ExtraInfo

def filter_positive_annotations(assay_data: pd.DataFrame, annotation_type: str, undefined_mutations: bool = False):
    """
    Return assays that were annotated automatically and not in ChEMBL for manual inspection.

    :param assay_data: annotated assays (with additional information)
    :param annotation_type: "new" for completely new annotations or "rescued" for annotations that were
                            originally 'UNDEFINED MUTATION'
    :param undefined_mutations: whether to consider undefined annotations as positive annotations
    :return: assays to check for false positive annotations
    """
    # Filter undefined mutations
    if not undefined_mutations:
        assay_data = assay_data[~assay_data['target_id'].str.contains('_MUTANT')]

    # Filter newly annotated assays (not previously annotated on ChEMBL)
    if annotation_type == 'new':
        return assay_data[~assay_data['target_id']
            .str.contains('_WT') & assay_data['mutation'].isnull() ]

    # Filter rescued annotations (previously annotated as 'UNDEFINED MUTATION' on ChEMBL)
    elif annotation_type == 'rescued':
        return assay_data[~assay_data['target_id']
            .str.contains('_WT') & (assay_data['mutation'] == 'UNDEFINED MUTATION')]

    else:
        raise ValueError('annotation_type must be either "new" or "rescued"')

def export_positive_annotations(chembl_version: str, annotation_round: int, annotation_dir: str):
    """
    Write assays that were annotated in the round of annotation and not in ChEMBL
    for manual inspection. These assays are potential false positive annotations and can be
    classified manually given the additional information.

    :param chembl_version: ChEMBL version
    :param annotation_round: annotation round
    :param annotation_dir: directory with annotation analysis results
    """
    positive_file = os.path.join(annotation_dir, f'chembl{chembl_version}_new_annotations.xlsx')

    # If file exists, read it
    if os.path.exists(positive_file):
        pass

    else:
        positive_assays = []
        # Read the annotation results
        assay_data = read_assay_annotations(chembl_version, annotation_round)

        # Extract additional information regarding assay annotation
        assay_data_ExtraInfo = get_assay_info(assay_data, chembl_version)

        # Filter new and rescued annotations
        new_annotations = filter_positive_annotations(assay_data_ExtraInfo, 'new')
        positive_assays.append(new_annotations)

        rescued_annotations = filter_positive_annotations(assay_data_ExtraInfo, 'rescued')
        positive_assays.append(rescued_annotations)

        # Concatenate new and rescued annotations
        positive_assays = pd.concat(positive_assays)

        # Write positive assays to xlsx file
        positive_assays.to_excel(positive_file, index=False)

def print_manual_annotation_instructions(chembl_version: str, annotation_round: int, annotation_dir: str):
    """
    Print instructions for manual annotation of positive assays

    :param chembl_version: ChEMBL version
    :param annotation_round: annotation round
    :param annotation_dir: directory with annotation analysis results
    """
    data_dir = get_data_path()
    print(f'Please check the file {annotation_dir}/chembl{chembl_version}_new_annotations.xlsx')
    print(f'for assays that were annotated automatically in round {annotation_round} and not in ChEMBL.')
    print('These assays are potential false positive annotations and can be classified manually')
    print('given the additional information.')
    print('For each assay, please add a column "reason" with the reason why the annotation is incorrect.')
    print('If the annotation is correct, please add "correct" to the column "reason".')
    print('For each reason, add also a column "group_reason" with a more general reason for the incorrect annotation.')
    print('Finally, export only the incorrect annotations to a new tab-separated .csv file in the data directory:')
    print(os.path.join(data_dir,f'chembl{chembl_version}_wrong_annotated_assays_round{annotation_round}.csv'))

def read_manual_positive_annotations(chembl_version: str, annotation_round: int):
    """
    Check if there is a file created with manual annotations based on the positive annotations
    :param chembl_version: ChEMBL version
    :param annotation_round: annotation round
    :return: manual annotations dataframe
    """
    data_dir = get_data_path()

    manual_positive_file = os.path.join(data_dir,
                                        f'chembl{chembl_version}_wrong_annotated_assays_round{annotation_round}.csv')
    if os.path.exists(manual_positive_file):
        manual_positive_assays = pd.read_csv(manual_positive_file, sep='\t')
        print('File with manual annotations for false positives found. Continuing...')
        return manual_positive_assays
    else:
        return False

def check_manual_positive_annotations(chembl_version: str, annotation_round: int):
    """
    Check if there is a file created with manual annotations based on the positive annotations and if it has the
    correct format

    :param chembl_version: ChEMBL version
    :param annotation_round: annotation round
    """
    manual_positive_assays = read_assay_annotations(chembl_version, annotation_round)

    # If file exists, read it and check that it has a column with manual annotations
    if manual_positive_assays:
        print('Checking manual annotation format...')
        if 'reason' not in manual_positive_assays.columns:
            raise ValueError('Must contain a column "reason" with manual annotations')
        elif 'group_reason' not in manual_positive_assays.columns:
            raise ValueError('Must contain a column "group_reason" with manual annotations')
        elif 'correct' in manual_positive_assays['reason'].unique():
            raise ValueError('Must contain only incorrect annotations')
        elif 'correct' in manual_positive_assays['group_reason'].unique():
            raise ValueError('Must contain only incorrect annotations')
        else:
            print('Correct manual annotation format. Continuing...')
            return manual_positive_assays
    else:
        raise ValueError('No file with manual annotations for false positives')

def filter_negative_annotations(assay_data: pd.DataFrame, undefined_mutations: bool = False):
    """
    Return assays that were originally annotated in ChEMBL but not in the round
    of local annotation (i.e. they were rejected by our annotation validation pipeline).
    These assays are potential false negative annotations and can be classified automatically
    given the additional information.

    :param assay_data: dataframe with assay data annotated
    :param undefined_mutations: whether to consider undefined annotations as negative annotations
    :return: assays to check for false negative annotations
    """
    # Get the assays that were originally annotated on ChEMBL with a defined mutation
    assay_data_original = assay_data[assay_data['mutation'] != 'UNDEFINED MUTATION'].dropna(subset=['mutation'])

    # Filter the assays that were not annotated in the round of annotation
    def check_original_valid(row):
        original_mutations = row['mutation'].split(',')
        valid_mutations = row['target_id'].split('_')[1:]
        if not undefined_mutations:
            if valid_mutations == ['MUTANT']:
                return True
            else:
                if all(item in valid_mutations for item in original_mutations):
                    return True
                else:
                    return False
        else:
            if all(item in valid_mutations for item in original_mutations):
                return True
            else:
                return False

    assay_data_original_rejected = assay_data_original[
        ~assay_data_original.apply(check_original_valid, axis=1)]

    return assay_data_original_rejected

def give_rejection_flag(assay_row):
    """
    Classify the reason why an assay was rejected by the annotation validation pipeline.
    :param assay_row: Row in the assay dataframe
    :return: reason for rejection
    """
    original_mutations = assay_row['mutation'].split(',')
    extracted_mutations = list(assay_row['aa_change'])
    valid_mutations = list(assay_row['mutants'])
    # Category 1: no amino acid change was extracted from the description with the regular expression
    if len(extracted_mutations) == 0:
        # Category 1A: original mutation is a deletion
        if any(['del' in om for om in original_mutations]):
            return 'original_deletion'
        # Category 1B: original mutation is undefined (THIS SHOULD BE ZERO BECAUSE WE FILTER THEM OUT)
        elif any(['UNDEFINED MUTATION' in om for om in original_mutations]):
            return 'original_undefined'
        # Category 1C: other reasons (e.g. tricky definition in description)
        else:
            return 'no_extraction'
    # Category 2: amino acid was extracted but it is not valid
    elif len(extracted_mutations) > 0:
        # Category 2A: original mutation contains a deletion
        if any(['del' in om for om in original_mutations]):
            return 'original_deletion'
        # Category 2A: original and extracted mutations match in all but sequence position (e.g. possible sequence
        # renumbering of which ChEMBL was aware of)
        elif (all(f'{om[0]}{om[-1]}' in [f'{em[0]}{em[-1]}' for em in extracted_mutations] for om in
        original_mutations)) and not (all(om[1:-1] in [em[1:-1] for em in extracted_mutations] for om in
        original_mutations)) and (len(valid_mutations) < len(original_mutations)):
            return 'original_shift_exception'
        # Category 2B: original and extracted mutations match, but they are not valid (e.g. wrong accession)
        elif (any(om in extracted_mutations for om in original_mutations)) and (len(valid_mutations) < len
            (original_mutations)):
            if assay_row['target_type'] == 'PROTEIN FAMILY':
                return 'protein_family'
            else:
                return 'original_not_valid'
        # Category 2C: original and extracted mutations do not match at all (e.g. tricky definition in description
        # that was known in ChEMBL as an exception)
        elif all(om not in extracted_mutations for om in original_mutations):
            return f'original_exception_{assay_row["curated_by"]}'
    else:
        return 'other'

def classify_negative_annotations(chembl_version: str, annotation_round: int):
    """
    Classify the reason why assays were rejected by the annotation validation pipeline.
    :param chembl_version: ChEMBL version
    :param annotation_round: Annotation round
    :return: dataframe with the reason for rejection
    """
    data_dir = get_data_path()
    false_negative_file = os.path.join(data_dir,
                                       f'chembl{chembl_version}_rejected_assays_round{annotation_round}.csv')

    # If file exists, read it
    if os.path.exists(false_negative_file):
        negative_annotations = pd.read_csv(false_negative_file, sep='\t')

    else:
        # Read the annotation results
        assay_data = read_assay_annotations(chembl_version, annotation_round)

        # Extract additional information regarding assay annotation
        assay_data_ExtraInfo = get_assay_info(assay_data, chembl_version)

        # Filter negative annotations
        negative_annotations = filter_negative_annotations(assay_data_ExtraInfo)

        # Classify the reason for rejection
        negative_annotations['rejection_flag'] = negative_annotations.apply(give_rejection_flag, axis=1)

        # Write out file with false negatives
        negative_annotations.to_csv(false_negative_file, sep='\t', index=False)

    return negative_annotations