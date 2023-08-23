# -*- coding: utf-8 -*-

"""Mutant statistics analysis. Part 3"""
"""Analyze characteristics of compound sets"""

import pandas as pd
import seaborn as sns
import os
import re
import json
import matplotlib.pyplot as plt

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

from .preprocessing import merge_chembl_papyrus_mutants
from .mutant_analysis_accession import filter_accession_data
from .mutant_analysis_common_subsets import compute_variant_activity_distribution, get_variant_common_subset, define_consistent_palette


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
    img = Draw.MolsToGridImage(mMols, legends=subset,
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