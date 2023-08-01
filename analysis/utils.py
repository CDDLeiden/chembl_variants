import json
import os

def get_directories(directories_file: str):
    """
    Reads a text file with directories for input and output of the analysis.
    :param directories_file: .txt file with {mutant_analysis_dir, modeling_dir, and modeling_analysis_dir} specified in
    the format: md_dir = 'C:\path\to\directory'
    :return: paths to directories
    """
    try:
        with open(directories_file, 'r') as f:
            directories_dict = json.load(f)
    except FileNotFoundError:
        print('Define a .json file with the five directories needed for the analysis:')
        print('md_dir, md_analysis_dir, desc_dir, desc_dir_dend and pcm_dir')
        print('The format of the file must be a dictionary.')

    try:
        return directories_dict['annotation_analysis_dir'],directories_dict['mutant_analysis_dir'], directories_dict[
            'modeling_dir'], directories_dict[
            'modeling_analysis_dir']
    except KeyError:
        print('The following directories need to be defined: mutant_analysis_dir, modeling_dir, and modeling_analysis_dir')

def update_directories(annotation_round: int, directory_path: str):
    """
    Updates the directories for the analysis of a specific annotation round.
    :param annotation_round:
    :param directory_path:
    :return:
    """
    annotation_tag = f'round_{annotation_round}'
    return os.path.join(directory_path, annotation_tag)

def get_distance_path(directories_file):
    """
    Returns the path to the distance directory.
    :param directories_file:
    :return:
    """
    annotation_analysis_dir, mutant_analysis_dir, modeling_dir, modeling_analysis_dir = get_directories(
        directories_file)
    return os.path.join(mutant_analysis_dir, '2_mutation_type', 'distance')

def get_annotation_analysis_path(directories_file: str, annotation_round: int):
    """
    Returns the path to the analysis directory for a specific annotation round.
    :param directories_file:
    :param annotation_round:
    :return:
    """
    annotation_analysis_dir,mutant_analysis_dir, modeling_dir, modeling_analysis_dir = get_directories(
        directories_file)
    return update_directories(annotation_round, annotation_analysis_dir)

def get_mutant_analysis_path(directories_file: str, analysis_step: str, annotation_round: int):
    """
    Returns the path to the analysis directory for a specific analysis step and annotation round.
    :param directories_file:
    :param analysis_step:
    :param annotation_round:
    :return:
    """
    annotation_analysis_dir, mutant_analysis_dir, modeling_dir, modeling_analysis_dir = get_directories(
        directories_file)

    if analysis_step == 'family':
        analysis_step_path = os.path.join(mutant_analysis_dir, '0_family_stats')
    elif analysis_step == 'accession':
        analysis_step_path = os.path.join(mutant_analysis_dir, '1_target_stats')
    elif analysis_step == 'type':
        analysis_step_path = os.path.join(mutant_analysis_dir, '2_mutation_type')
    elif analysis_step == 'common':
        analysis_step_path = os.path.join(mutant_analysis_dir, '3_common_subset')
    elif analysis_step == 'compound':
        analysis_step_path = os.path.join(mutant_analysis_dir, '4_compound_clusters')
    elif analysis_step == 'bioactivity':
        analysis_step_path = os.path.join(mutant_analysis_dir, '5_bioactivity_distribution')
    else:
        print('The analysis step is not defined. Please check the analysis_step argument.')

    return update_directories(annotation_round, analysis_step_path)
