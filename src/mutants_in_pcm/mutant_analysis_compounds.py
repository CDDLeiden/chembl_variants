# -*- coding: utf-8 -*-

"""Mutant statistics analysis. Part 3"""
"""Analyze characteristics of compound sets"""

import pandas as pd
import seaborn as sns
import os
import re
import json
import matplotlib.pyplot as plt
import chembl_downloader

from mycolorpy import colorlist as mcp
import numpy as np
from matplotlib.patches import Patch

import rdkit
from rdkit import Chem
from rdkit import RDConfig
from rdkit import DataStructs
from rdkit.Chem import rdFMCS
from rdkit.Chem import rdMolHash
from rdkit.Chem import PandasTools
from rdkit.Chem import Draw
from rdkit.Chem import rdFingerprintGenerator
from rdkit.ML.Cluster import Butina
from rdkit.Chem.Draw import IPythonConsole
IPythonConsole.molSize=(450,350)
from rdkit.Chem import rdRGroupDecomposition
from rdkit.Chem import rdqueries
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit import Geometry
rdDepictor.SetPreferCoordGen(True)

from IPython.display import SVG,Image
from ipywidgets import interact

from .data_path import get_data_path
from .preprocessing import merge_chembl_papyrus_mutants
from .mutant_analysis_accession import filter_accession_data
from .mutant_analysis_common_subsets import compute_variant_activity_distribution, get_variant_common_subset, define_consistent_palette


def downlaod_chembl_compounds(chembl_version: str, data_folder: str = None):
    """Download ChEMBL compounds.

    :param chembl_version: version of chembl to work with
    :param data_folder: path to the folder in which the ChEMBL
    SQLite database is located or will be downloaded (default:
    pystow's default directory)
    """
    if data_folder is not None:
        os.environ['PYSTOW_HOME'] = data_folder

    data_dir = get_data_path()
    chembl_file = os.path.join(data_dir, f'chembl{chembl_version}_compound_info.csv')

    if not os.path.isfile(chembl_file):
        # Query ChEMBL for compound descriptions
        query = """
                    SELECT molecule_dictionary.molregno,molecule_dictionary.chembl_id,molecule_dictionary.pref_name,
                    molecule_dictionary.max_phase,
                    molecule_dictionary.therapeutic_flag,molecule_dictionary.natural_product,
                    molecule_dictionary.molecule_type,molecule_dictionary.first_approval,molecule_dictionary.oral,
                    molecule_dictionary.parenteral,molecule_dictionary.topical,molecule_hierarchy
                    .parent_molregno,compound_structures.canonical_smiles,
                    compound_structures.standard_inchi,compound_structures.standard_inchi_key,drug_mechanism
                    .mechanism_of_action,drug_mechanism.action_type,drug_mechanism
                    .mechanism_comment,drug_mechanism
                    .selectivity_comment,component_sequences.accession,variant_sequences.mutation,drug_indication
                    .max_phase_for_ind,drug_indication.mesh_heading
                    FROM molecule_dictionary
                        LEFT JOIN molecule_hierarchy
                            ON molecule_dictionary.molregno = molecule_hierarchy.molregno
                        LEFT JOIN compound_structures USING (molregno)
                        LEFT JOIN drug_mechanism
                            ON molecule_dictionary.molregno = drug_mechanism.molregno
                        LEFT JOIN drug_indication
                            ON molecule_dictionary.molregno = drug_indication.molregno
                        LEFT JOIN target_dictionary
                            ON drug_mechanism.tid = target_dictionary.tid
                        LEFT JOIN target_components
                            ON target_dictionary.tid = target_components.tid
                        LEFT JOIN component_sequences
                            ON target_components.component_id = component_sequences.component_id
                        LEFT JOIN variant_sequences
                            ON drug_mechanism.variant_id = variant_sequences.variant_id

                    """

        chembl_compounds = chembl_downloader.query(query, version=chembl_version,
                                                   prefix=['mutants-in-pcm', 'chembl'])

        # Check whether the compound is the parent molecule
        def is_parent(x):
            if x['parent_molregno'] == x['molregno']:
                return True
            else:
                return False

        chembl_compounds['is_parent'] = chembl_compounds.apply(is_parent, axis=1)
        # Make indications column with mesh and max phase for the indication to facilitate merging later
        chembl_compounds['indications'] = chembl_compounds.apply(lambda x: f'{x.mesh_heading} ({x.max_phase_for_ind})',
                                                                 axis=1)

        # Save results
        chembl_compounds.to_csv(chembl_file, sep='\t', index=None)

    else:
        print('ChEMBL compound file already exists. Reading it.')
        chembl_compounds = pd.read_csv(chembl_file, sep='\t')

    return chembl_compounds


def group_unique_df(df, groupby_cols):
    """Group a dataframe by a list of columns and return a dataframe with the unique values of each column.
    """
    grouped_df = df.groupby(groupby_cols, dropna=False).agg(['unique'])  # Important: keep NAs
    # Drop the multiindex
    grouped_df.columns = grouped_df.columns.droplevel(1)
    # Convert the unique values to strings
    grouped_df = grouped_df.applymap(lambda x: x[0] if len(x) == 1 else ';'.join(str(i) for i in x))
    # Reset the index
    grouped_df = grouped_df.reset_index()

    return grouped_df


def has_parent_moa(x):
    """Check whether a compound is a parent molecule with a mechanism of action."""
    if (x['is_parent'] == True) and (str(x['accession']) not in ['NaN', 'nan', 'None']):
        return True
    else:
        return False


def fix_parent_moa(chembl_version: str, data_folder: str = None):
    """Fix the mechanism of action of parent molecules.

    :param chembl_version: version of chembl to work with
    :param data_folder: path to the folder in which the ChEMBL
    """
    if data_folder is not None:
        os.environ['PYSTOW_HOME'] = data_folder

    data_dir = get_data_path()
    fixed_chembl_file = os.path.join(data_dir, f'chembl{chembl_version}_compound_info_fixed_MOA.csv')

    df = downlaod_chembl_compounds(chembl_version, data_folder)

    if not os.path.isfile(fixed_chembl_file):
        # Check if the parent compound has a MOA (accession defined)
        df['has_parent_moa'] = df.apply(has_parent_moa, axis=1)

        # Create dataframe to check which compounds need to be fixed (add a MOA to the parent compound)
        check_parent_moa = df.groupby(['parent_molregno'], dropna=False).agg({'accession': 'nunique',
                                                                              'chembl_id': 'nunique',
                                                                              'has_parent_moa': 'unique'})

        # Keep only the parent compounds that need to be fixed and those that can help fix the parent compound
        # (i.e. the parent compound does not have a MOA and it has at least one child compound with a MOA)
        to_fix = check_parent_moa[(check_parent_moa['chembl_id'] > 1) & (check_parent_moa['accession'] > 0) &
                                  (check_parent_moa['has_parent_moa'].apply(lambda x: True not in x))].index.to_list()

        # Check which parent compounds need to be fixed
        parents_to_fix = df[(df['parent_molregno'].isin(to_fix)) & (df['is_parent'] == True)]
        # Keep only one row per parent compound (merging all unique indication information)
        parents_to_fix_unique = group_unique_df(parents_to_fix, ['chembl_id'])

        # Check which child compounds can help fix the parent compound
        children_to_fix = df[(df['parent_molregno'].isin(to_fix)) & (df['is_parent'] == False)]
        # Keep only one row per child compound - accession pair (merging all unique indication information)
        children_to_fix_unique = group_unique_df(children_to_fix, ['chembl_id', 'accession'])
        # Rename all columns to avoid confusion when merging
        children_to_fix_unique.columns = (children_to_fix_unique.
                                          columns.map(lambda x: f'{x}_child' if x != 'parent_molregno' else x))
        # Keep columns of interest
        children_to_fix_unique = children_to_fix_unique[['parent_molregno', 'chembl_id_child', 'pref_name_child',
                                                         'max_phase_child',
                                                         'accession_child', 'mutation_child',
                                                         'mechanism_of_action_child',
                                                         'action_type_child', 'mechanism_comment_child',
                                                         'selectivity_comment_child',
                                                         'indications_child']]

        # Merge parent and child compounds to annotate parent with the child information
        parents_fixed = pd.merge(parents_to_fix_unique, children_to_fix_unique, on='parent_molregno', how='left')
        # Keep one row per parent compound with unique information
        parents_fixed_unique = group_unique_df(parents_fixed, ['chembl_id'])

        # Merge the fixed parent compounds with the original dataframe
        # First keep also one row per original compound with unique information
        df_unique = group_unique_df(df, ['chembl_id'])
        print(f'Number of compounds before fixing: {len(df_unique)}')
        # Remove compounds that need to be fixed
        df_unique = df_unique[~df_unique['chembl_id'].isin(parents_to_fix_unique['chembl_id'].tolist())]
        print(f'Number of unique compounds before fixing: {len(df_unique)}')
        # Merge the fixed parent compounds with the original dataframe
        df_fixed = pd.concat([df_unique, parents_fixed_unique], ignore_index=True)
        print(f'Number of unique compounds after fixing: {len(df_fixed)}')

        # Save results
        df_fixed.to_csv(fixed_chembl_file, sep='\t', index=None)

    else:
        print('Fixed MOA file already exists. Reading it.')
        df_fixed = pd.read_csv(fixed_chembl_file, sep='\t')

    return df_fixed


def map_chembl_compounds(chembl_version: str, papyrus_version: str, papyrus_flavor: str,
                         chunksize: int, annotation_round: int, data_folder: str = None):
    """Obtain compound descriptions from ChEMBL whenever possible.

    :param chembl_version: version of chembl to work with
    :param papyrus_version: version of papyrus to work with
    :param papyrus_flavor: flavor of papyrus to work with
    :param chunksize: size of chunks of data to be used (default: None)
    :param annotation_round: round of annotation to work with
    :param data_folder: path to the folder in which the ChEMBL
    SQLite database is located or will be downloaded (default:
    pystow's default directory)
    """
    if data_folder is not None:
        os.environ['PYSTOW_HOME'] = data_folder

    data_dir = get_data_path()
    project_compounds_file = os.path.join(data_dir, f'chembl{chembl_version}_papyrus{papyrus_version}' \
                                                    f'{papyrus_flavor}_round{annotation_round}_compound_mapping.csv')

    if not os.path.isfile(project_compounds_file):
        # Read unique compounds from ChEMBL/Papyrus annotated mutant data
        annotated_data = merge_chembl_papyrus_mutants(chembl_version, papyrus_version, papyrus_flavor,
                                                      chunksize, annotation_round)

        # Keep compounds with ChEMBL IDs
        annotated_data = annotated_data[annotated_data['CID'].str.contains('CHEMBL')]
        unique_compounds = annotated_data[['connectivity', 'CID']].drop_duplicates().sort_values('connectivity')

        # Download ChEMBL compound descriptions and path parent MOA with child compounds if needed
        chembl_compounds_fixed = fix_parent_moa(chembl_version, data_folder)

        # Merge ChEMBL compound descriptions with unique compounds
        mapped_compounds = pd.merge(unique_compounds, chembl_compounds_fixed, left_on='CID',
                                    right_on='chembl_id', how='left').fillna('NA')

        # Save results
        mapped_compounds.to_csv(project_compounds_file, sep='\t', index=None)

    else:
        print('ChEMBL/Papyrus compound mapping file already exists. Reading it.')
        mapped_compounds = pd.read_csv(project_compounds_file, sep='\t')

    return mapped_compounds

def GetRingSystems(mol, includeSpiro: bool = False):
    """
    Extract ring systems in molecule
    :param mol: rdkit molecule object
    :param includeSpiro: whether to include Spiro links as rings
    :return: list of atoms forming ring systems in the molecule
    """
    ri = mol.GetRingInfo()
    systems = []
    for ring in ri.AtomRings():
        ringAts = set(ring)
        nSystems = []
        for system in systems:
            nInCommon = len(ringAts.intersection(system))
            if nInCommon and (includeSpiro or nInCommon>1):
                ringAts = ringAts.union(system)
            else:
                nSystems.append(system)
        nSystems.append(ringAts)
        systems = nSystems
    return systems

def visualize_molecular_subset_highlights(accession: str, accession_data: pd.DataFrame, subset: list, subset_alias: str,
                                          match_type: str, substructure_to_match: dict, save: bool, output_dir: str):
    """
    Plot 2D molecualr representation of compounds in a subset, highlighting a specific substructure in each molecule 
    for easier comparison
    :param accession: target Uniprot accession code 
    :param accession_data: dataframe with bioactivity data for the target of interest 
    :param subset: list of compounds in the form of connectivities 
    :param subset_alias: alias to identify the subset of interest in the output file
    :param match_type: type of substructure to highlight. Options include: 'Murcko' (highlight  Murcko 
    scaffold), 'ring' (highlight biggest ring system in each molecule), 'MCS' (highlight maximum 
    common substructure accross the subset), 'SMILES' (highlight an exact match of a SMILES pattern defined in 
    substructure_to_match), 'SMARTS' (highlight a match for a SMARTS pattern defined in substructure_to_match)
    :param substructure_to_match: dictionary with the SMILES or SMARTS pattern to match for the accession of interest
    :param save: whether to save the figure
    :param output_dir: path to output directory 
    :return: Figure 
    """""
    subset_df = accession_data[accession_data['connectivity'].isin(subset)]
    # Keep first occurence
    subset_df.drop_duplicates(subset='connectivity', keep='first', inplace=True, ignore_index=True)

    # Compute molecule from smiles
    PandasTools.AddMoleculeColumnToFrame(subset_df,'SMILES','Molecule',includeFingerprints=True)

    # Extract RDkit molecular objects
    mMols = subset_df['Molecule'].tolist()
    subset_legend = subset_df['connectivity'].tolist() # Just in case the order does not fully match with the
    # original subset list

    if match_type == 'Murcko':
        murckoList = [Chem.Scaffolds.MurckoScaffold.GetScaffoldForMol(mMol) for mMol in mMols]
        highlight_match = [mMol.GetSubstructMatch(murcko) for mMol, murcko in zip(mMols, murckoList)]

    elif match_type == 'MCS':
        mcs = rdFMCS.FindMCS(mMols)
        match_mol = Chem.MolFromSmarts(mcs.smartsString)
        highlight_match = [mMol.GetSubstructMatch(match_mol) for mMol in mMols]

    elif match_type == 'ring':
        # Extract ring systems for each molecule
        ringSys_list = [GetRingSystems(mMol) for mMol in mMols]
        # Keep the biggest ring system for each molecule
        highlight_match = [max(ringSys, key=len) for ringSys in ringSys_list]

    elif match_type == 'SMILES':
        match_mol = Chem.MolFromSmiles(substructure_to_match[match_type])
        highlight_match = [mMol.GetSubstructMatch(match_mol) for mMol in mMols]

    elif match_type == 'SMARTS':
        match_mol = Chem.MolFromSmarts(substructure_to_match[match_type])
        highlight_match = [mMol.GetSubstructMatch(match_mol) for mMol in mMols]


    # Draw molecules in grid with the highlight defined
    img = Draw.MolsToGridImage(mMols, legends=subset_legend,
                               highlightAtomLists=highlight_match,
                               subImgSize=(500, 500), useSVG=False, molsPerRow=5, returnPNG=False)

    # Save image
    if save:
        img.save(os.path.join(output_dir, f'{accession}_{subset_alias}_highlight_{match_type}.png'))

    return img

def tanimoto_distance_matrix(fp_list: list):
    """
    Calculate distance matrix for fingerprint list
    :param fp_list: list of morgan fingerprints
    :return: dissimilarity matrix
    """
    dissimilarity_matrix = []
    # Notice how we are deliberately skipping the first and last items in the list
    # because we don't need to compare them against themselves
    for i in range(1, len(fp_list)):
        # Compare the current fingerprint against all the previous ones in the list
        similarities = DataStructs.BulkTanimotoSimilarity(fp_list[i], fp_list[:i])
        # Since we need a distance matrix, calculate 1-x for every element in similarity matrix
        dissimilarity_matrix.extend([1 - x for x in similarities])
    return dissimilarity_matrix

def butina_cluster_compounds(accession: str, accession_data: pd.DataFrame, subset: list, subset_alias: str,
                             output_dir: str, cutoff: float = 0.2):
    """
    Cluster compounds with Butina algorithm.
    :param accession: target Uniprot accession code
    :param accession_data: dataframe with bioactivity data for the target of interest
    :param subset: list of compounds in the form of connectivities
    :param subset_alias: alias to identify the subset of interest in the output file
    :param output_dir: path to output directory
    :param cutoff: distance cutoff to the cluster central molecule for molecule inclusion in cluster
    :return: clusters: cluster object
            compounds: list of (connectivity,molecule object) tuples that were clustered
            connectivity_cluster_dict: dictionary of connectivities and their assigned cluster
    """
    subset_df = accession_data[accession_data['connectivity'].isin(subset)]
    # Keep first occurence
    subset_df.drop_duplicates(subset='connectivity', keep='first', inplace=True, ignore_index=True)

    # Compute molecule from smiles
    PandasTools.AddMoleculeColumnToFrame(subset_df,'SMILES','Molecule',includeFingerprints=True)

    # Extract RDkit molecular objects
    mMols = subset_df['Molecule'].tolist()

    compounds = []
    for _, connectivity, mol in subset_df[["connectivity", "Molecule"]].itertuples():
        compounds.append((mol, connectivity))

    # Create fingerprints for all molecules
    rdkit_gen = rdFingerprintGenerator.GetRDKitFPGenerator(maxPath=5)
    fplist = [rdkit_gen.GetFingerprint(mol) for mol in mMols]

    # Calculate Tanimoto distance matrix
    distance_matrix = tanimoto_distance_matrix(fplist)

    # Now cluster the data with the implemented Butina algorithm:
    clusters = Butina.ClusterData(distance_matrix, len(fplist), cutoff, isDistData=True)
    clusters = sorted(clusters, key=len, reverse=True)

    # Give a short report about the numbers of clusters and their sizes
    num_clust_g1 = sum(1 for c in clusters if len(c) == 1)
    num_clust_g5 = sum(1 for c in clusters if len(c) > 5)
    num_clust_g25 = sum(1 for c in clusters if len(c) > 25)
    num_clust_g100 = sum(1 for c in clusters if len(c) > 100)

    print("total # clusters: ", len(clusters))
    print("# clusters with only 1 compound: ", num_clust_g1)
    print("# clusters with >5 compounds: ", num_clust_g5)
    print("# clusters with >25 compounds: ", num_clust_g25)
    print("# clusters with >100 compounds: ", num_clust_g100)

    # Plot the size of the clusters
    fig, ax = plt.subplots(figsize=(15, 4))
    ax.set_xlabel("Cluster index")
    ax.set_ylabel("Number of molecules")
    ax.set_title(f"Threshold: {cutoff:3.1f}")
    ax.bar(range(1, len(clusters) + 1), [len(c) for c in clusters], lw=5)

    # Save plot
    plt.savefig(os.path.join(output_dir,f'{accession}_{subset_alias}_ButinaClusters_{cutoff}.png'))

    # Make dictionary mapping cluster index to connectivity
    connectivity_cluster_dict = {}
    for i,(mol,connectivity) in enumerate(compounds):
        for j,cluster in enumerate(clusters):
            if i in cluster:
                connectivity_cluster_dict[connectivity] = j + 1 # cluster numbers from 1 on (not zero)
    with open(os.path.join(output_dir,f'{accession}_{subset_alias}_ButinaClusters_{cutoff}.json'), 'w') as out_file:
        json.dump(connectivity_cluster_dict, out_file)

    # Visualize molecules in clusters with more than 1 member of strictly common subset, highlighting the MCS in each
    # cluster
    for cluster in [i + 1 for i, c in enumerate(clusters) if len(c) > 1]:
        cluster_connectivities = [k for k, v in connectivity_cluster_dict.items() if v == cluster]
        visualize_molecular_subset_highlights(accession, accession_data, cluster_connectivities,
                                              f'{subset_alias}_{cutoff}cluster{cluster}',
                                              'MCS', {}, True, output_dir)

    return clusters,compounds, connectivity_cluster_dict

def get_clustering_stats(accession: str, output_dir: str, subset_alias:str, cutoff: float = 0.2):
    """
    Get statistics for clustering of compounds with Butina algorithm from the output json file.
    :param accession: target Uniprot accession code
    :param subset_alias: alias to identify the subset of interest in the output file
    :param output_dir: path to output directory
    :param cutoff: distance cutoff to the cluster central molecule for molecule inclusion in cluster
    """
    with open(os.path.join(output_dir, accession, f'{accession}_{subset_alias}_ButinaClusters_{cutoff}.json')) as \
            in_file:
        connectivity_cluster_dict = json.load(in_file)

    # Get cluster sizes
    clusters = sorted(list(set(connectivity_cluster_dict.values())))

    cluster_sizes = [len([k for k, v in connectivity_cluster_dict.items() if v == cluster]) for cluster in clusters]
    print(f'Number of clusters: {len(clusters)}')
    print(f'Number of compounds in clusters: {sum(cluster_sizes)}')

    print(f'Number of compounds per cluster:')
    for c,s in zip(clusters,cluster_sizes):
        print(f'Cluster {c}: {s} compounds')

    return connectivity_cluster_dict


def plot_bioactivity_distribution_cluster_subset(accession: str, annotation_round:str, output_dir: str,
                                                 replot: bool = False):
    """
    Plot bioactivity distribution of compounds in clusters of the common subset. In this case,
    the common subset is very lax and includes all compounds that have been tested in at least
    two variants.

    :param accession: Uniprot accession code
    :param annotation_round: round of annotation following further curation
    :param output_dir: path to write the results to
    :param replot: whether to replot existing bioactivity distribution plotting results
    """
    # Load data
    accession_data = filter_accession_data(merge_chembl_papyrus_mutants('31', '05.5', 'nostereo', 1_000_000,
                                                                        annotation_round),accession)

    # Create directory for the accession of interest
    if not os.path.exists(os.path.join(output_dir, accession)):
        os.mkdir(os.path.join(output_dir, accession))

    # Get common subset (compounds tested in at least two variants)
    common_data, coverage_dict = get_variant_common_subset(accession_data, accession, True, 2,
                                                           None, False, os.path.join(output_dir, accession))
    # Extract list of compounds in the common subset
    accession_data_common = accession_data[accession_data['connectivity'].
        isin(common_data['connectivity'].unique().tolist())]

    # Check if clustering was already done, if so read in results
    if os.path.exists(os.path.join(output_dir, accession, f'{accession}_full_set_ButinaClusters_0.5.json')):
        with open(os.path.join(output_dir, accession, f'{accession}_full_set_ButinaClusters_0.5.json'), 'r') as in_file:
            connectivity_cluster_dict = json.load(in_file)
            print(f'Loaded {len(connectivity_cluster_dict)} clusters for {accession}')
    else:
        # Butina cluster compounds in the common subset
        clusters, compounds, connectivity_cluster_dict = \
            butina_cluster_compounds(accession, accession_data_common,
                                     accession_data_common.connectivity.unique().tolist(),
                                     'full_set', os.path.join(output_dir, accession), 0.5)

    # Define consistent color palette that includes all variants
    palette_dict = define_consistent_palette(accession_data_common, accession)

    # Plot bioactivity distribution for 10 largest clusters
    for i in range(1, 11):
        # Extract compounds in cluster
        cluster_connectivity = [k for k, v in connectivity_cluster_dict.items() if v == i]
        # Extract data for compounds in cluster
        data_cluster = accession_data_common[accession_data_common['connectivity'].isin(cluster_connectivity)]
        # Create directory for cluster
        cluster_dir = os.path.join(output_dir, accession, f'cluster_{i}')
        if not os.path.exists(cluster_dir):
            os.mkdir(cluster_dir)
        # Plot distribution
        compute_variant_activity_distribution(data_cluster, accession, common=False, sim=False, sim_thres=None,
                                              threshold=None, variant_coverage=None, plot=True, hist=False,
                                              plot_mean=True, color_palette=palette_dict, save_dataset=False,
                                              output_dir=cluster_dir, replot=replot)

def annotate_cluster_compounds(connectivity_cluster_dict: dict,
                                  mapped_compounds: pd.DataFrame):
    """
    Annotate compounds in clusters with additional information for further analysis
    :param connectivity_cluster_dict: Dictionary with connectivity and cluster assignment
    :param mapped_compounds: Compound information dataframe
    :return: Compound information dataframe for compounds in clusters
    """
    # Extract compounds in dictionary from dataframe
    cluster_df = mapped_compounds[mapped_compounds['connectivity'].isin(connectivity_cluster_dict.keys())]

    # Add cluster column by mapping dictionary on connectivity
    cluster_df['cluster'] = cluster_df['connectivity'].map(connectivity_cluster_dict)
    cluster_df.sort_values('connectivity', inplace=True)

    # Keep one row per connectivity
    cluster_df_unique = group_unique_df(cluster_df, 'connectivity')

    return cluster_df_unique

def check_moa(x, accession):
    """
    Check whether a compound is linked to the accession in its MOA or otherwise if its child is
    :param x: row
    :param accession: accession of interest
    """
    parent_moa = False
    child_moa = False
    if accession in str(x['accession']):
        parent_moa = True
    elif accession in str(x['accession_child']):
        child_moa = True

    return parent_moa, child_moa

def check_mutation(x):
    """
    Check whether a compound is linked to a mutation in its MOA or otherwise if its child is
    :param x: row
    :param accession: accession of interest
    """
    parent_mutation = False
    child_mutation = False
    if not pd.isna(x['mutation']):
        parent_mutation = True
    elif not pd.isna(x['mutation_child']):
        child_mutation = True

    return parent_mutation, child_mutation

def check_approval(x):
    """
    Check whether a compound is approved or otherwise if its child is approved
    :param x: row
    """
    approved = False
    approved_child = False

    max_phase_connectivity = max([int(i) for i in str(x['max_phase']).split(';')])
    try:
        max_phase_connectivity_child = max([int(i) for i in str(x['max_phase_child']).split(';')])
    except ValueError:  # Some compounds don't have a child defined and will throw an error
        max_phase_connectivity_child = 0

    if max_phase_connectivity == 4:
        approved = True
    elif max_phase_connectivity_child == 4:
        approved_child = True
    return approved, approved_child
def explore_cluster_compound_info(cluster_df_unique: pd.DataFrame,
                                  accession: str,
                                  analysis_type: str,
                                  sort: str = 'cluster'):
    """
    Explore compound information in clusters
    :param cluster_df_unique: dataframe with compound information for compounds in clusters
    :param accession: Uniprot accession code of interest
    :param analysis_type: Type of analysis to perform. Options include 'MOA' and 'approval'
    :return: statistics dataframe with cluster information
    """
    if analysis_type == 'MOA':
        # Check if compounds are linked to the analysis accession in their MOA
        cluster_df_unique[[f'{accession}_MOA', f'{accession}_MOA_child']] = cluster_df_unique.apply(
            lambda
                x:
            check_moa
            (x, accession=accession), axis=1, result_type='expand')

        # Compute statistics
        stats = cluster_df_unique.groupby('cluster').agg({f'{accession}_MOA':np.sum,
                                                          f'{accession}_MOA_child':np.sum})
        # Add column with sum of MOA and MOA child
        stats[f'{accession}_MOA_total'] = stats[f'{accession}_MOA'] + stats[f'{accession}_MOA_child']

    elif analysis_type == 'mutation':
        # Check if compounds are linked to the analysis accession in their MOA
        cluster_df_unique[[f'{accession}_mutation', f'{accession}_mutation_child']] = cluster_df_unique.apply(
            lambda x: check_mutation(x), axis=1, result_type='expand')

        # Compute statistics
        stats = cluster_df_unique.groupby('cluster').agg({f'{accession}_mutation':np.sum,
                                                          f'{accession}_mutation_child':np.sum})
        # Add column with sum of MOA and MOA child
        stats[f'{accession}_mutation_total'] = stats[f'{accession}_mutation'] + stats[f'{accession}_mutation_child']

    elif analysis_type == 'approval':
        # Check if compounds are approved drugs
        cluster_df_unique[['approved', 'approved_child']] = cluster_df_unique.apply(lambda x: check_approval(x), axis=1,
                                                                                    result_type='expand')
        # Compute statistics
        stats = cluster_df_unique.groupby('cluster').agg({'approved':np.sum, 'approved_child':np.sum})
        # Add column with sum of approved and approved child
        stats['approved_total'] = stats['approved'] + stats['approved_child']

    if sort == 'cluster':
        satisfying_condition =['parent', len(stats[stats[stats.columns[0]] > 0]), (len(stats[stats[stats.columns[0]]
                                                                                             > 0]) / len(stats))*100]
        pass
    elif sort == 'parent':
        satisfying_condition = ['parent', len(stats[stats[stats.columns[0]] > 0]),
                                len(stats[stats[stats.columns[0]] > 0]) / len(stats)]
        # Sort in descending order of values in the first column
        stats.sort_values(stats.columns[0], ascending=False, inplace=True)

    elif sort == 'child':
        satisfying_condition = ['child', len(stats[stats[stats.columns[1]] > 0]),
                                len(stats[stats[stats.columns[1]] > 0]) / len(stats)]
        # Sort in descending order of values in the second column
        stats.sort_values(stats.columns[1], ascending=False, inplace=True)
    elif sort == 'both':
        satisfying_condition = ['parent or child', len(stats[stats[stats.columns[2]] > 0]),
                                len(stats[stats[stats.columns[2]] > 0]) / len(stats)]
        # Sort in descending order of values in the third column and then in descending order of values in the first
        # column
        stats.sort_values([stats.columns[2], stats.columns[0]], ascending=False, inplace=True)

    # Print how many clusters have at least one compound that satisfies the condition
    print(f'Number of clusters with at least one ({satisfying_condition[0]}) compound satisfying the condition:'
          f' {satisfying_condition[1]} ({satisfying_condition[2]:.2f}%)')

    return stats

if __name__ == '__main__':
    annotation_round = 1
    output_dir = f'C:\\Users\\gorostiolam\\Documents\\Gorostiola Gonzalez, ' \
             f'Marina\\PROJECTS\\6_Mutants_PCM\\DATA\\2_Analysis\\1_mutant_statistics\\4_compound_clusters\\round' \
                 f'_{annotation_round}'

    # Plot distributions of bioactivities in most populated Butina clusters for targets with > 90 compounds in common
    # subsets
    for accession in ['P00533', 'Q72547', 'O75874','O60885','P00519','P07949','P10721','P13922','P15056','P22607',
    'P30613','P36888','Q15910','Q5S007','Q9UM73']:
        plot_bioactivity_distribution_cluster_subset(accession, annotation_round, output_dir)

    # Plot distribution of bioactivities in most populated Butina clusters for targets with => 50% mutant bioactivity
    # ratio
    for accession in ['P15056', 'P23443', 'O75874', 'P13922', 'P30613', 'P01116', 'Q6P988', 'Q86WV6', 'P48735',
                      'Q9P2K8', 'P21146', 'P48065', 'Q81R22', 'P07753', 'Q62120', 'Q15022', 'C1KIQ2', 'P36873',
                      'Q5NGQ3', 'Q9QUR6', 'D5F1R0', 'P02511', 'P11678', 'P0DOF9', 'P56690', 'Q05320', 'P13738',
                      'Q9NZN5', 'P15682', 'Q9NPD8']:
        plot_bioactivity_distribution_cluster_subset(accession, annotation_round, output_dir)