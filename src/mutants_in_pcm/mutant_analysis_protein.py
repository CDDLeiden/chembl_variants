# -*- coding: utf-8 -*-


"""Mutant statistics analysis. Part 3"""
"""Analyze characteristics of mutated protein structures"""

import numpy as np
import os
from math import sqrt
from UniProtMapper import UniProtIDMapper
from pypdb import get_info
import Bio
from Bio.PDB import PDBList
import json

def map_accession_to_pdb(accession:str):
    """
    Fetch all PDB codes for a Uniprot accession code of interest.
    Returns: pdb_list (list): List of PDB codes
    """
    mapper = UniProtIDMapper()
    result, failed = mapper.mapIDs(
        ids=[accession], from_db="UniProtKB_AC-ID", to_db="PDB"
    )

    try:
        pdb_list = result["to"].tolist()
    except KeyError:
        pdb_list = []

    return pdb_list

def get_pdb_ligand(pdb_code:str):
    """
    Extract ligand name for a particular PDB code.
    Returns:
        comp_id (str): Ligand name.
    """
    info = get_info(pdb_code)

    try:
        comp_id = info['rcsb_binding_affinity'][0]['comp_id']
    except KeyError:
        comp_id = None

    return comp_id

def COG(coordinates: list):
    """
    Calculates center of geometry of a list of [x,y,z] coordinates.
    Returns:
        center (list): List of float coordinates [x,y,z] that represent the
        center of geometry (precision 3).
    """
    center = [None, None, None]
    # calculate center of geometry
    center = [sum([coordinates[i][j] / (len(coordinates))
                   for i in range(len(coordinates))]) for j in range(3)]
    center = [round(center[i], 3) for i in range(3)]

    return center


def get_pdb_ligand_chain(pdbfile:str, hetname:str):
    """
    Extracts the chain ID where the ligand sit. If multiple, returns the first one.
    Returns:
        chain_id (str): Name of the chain (e.g. 'A')
    """
    ligand_chains = []

    with open(pdbfile) as pdb:
        for line in pdb:
            # Ligand lines
            if line.startswith('HETATM'):
                if (hetname != None) and (line[17:20] == hetname):
                    # Extract chain ID
                    chain = line[21:22]
                    if chain not in ligand_chains:
                        ligand_chains.append(chain)
    try:
        return ligand_chains[0]
    except IndexError:
        return ''


def get_pdb_protein_residues(pdbfile: str, chain: str):
    """
    Extracts the protein residue numbers from a pdb file.
    Returns:
        resn_list (list): List of residues in the protein.
    """
    resn_list = []

    with open(pdbfile) as pdb:
        for line in pdb:
            # Select the right chain
            if line[21:22] == chain:
                if line.startswith('ATOM'):
                    try:
                        if int(line[22:27]) not in resn_list:
                            resn_list.append(int(line[22:27]))
                    # Account for alternative residue atom positions
                    except ValueError:
                        continue

    return resn_list

def get_pdb_coordinates(pdbfile: str, chain: str, hetname: str, resn: list):
    """
    Extracts list of coordinates of a ligand and residue numbers of interest from a pdb file.
    Returns:
        ligand_coordinates (list): Nested list of float coordinates [x,y,z] for all atoms in the ligand.
        residue_coordinates (dict): Dictionary containing a nested list of float coordinates for all atoms in each
                                    of the residues of interest.
    """
    ligand_coordinates = []
    residue_coordinates = {}

    with open(pdbfile) as pdb:
        # extract coordinates [ [x1,y1,z1], [x2,y2,z2], ... ]

        # Define residues to compute
        for res in resn:
            residue_coordinates[res] = []


        for line in pdb:
            # Select the right chain
            if line[21:22] == chain:
                # Ligand coordinates
                if line.startswith('HETATM'):
                    if (hetname != None) and (line[17:20] == hetname):
                        ligand_coordinates.append([float(line[30:38]),    # x_coord
                                                   float(line[38:46]),    # y_coord
                                                   float(line[46:54])     # z_coord
                        ])

                # Residue coordinates
                for res in resn:
                    try:
                        if int(line[22:27]) == res:
                            residue_coordinates[res].append([float(line[30:38]),  # x_coord
                                                            float(line[38:46]),  # y_coord
                                                            float(line[46:54])  # z_coord
                                                            ])
                    except ValueError:
                        continue


    return ligand_coordinates, residue_coordinates

def distance_between_coordinates(coord1: list, coord2: list):
    """
    Calculates Euclidean distance (Angström) between two sets of [x,y,z] coordinates.
    Returns
        distance (float): Distance between the two coordinates (precision 3).
    """
    x1,y1,z1 = coord1
    x2,y2,z2 = coord2
    distance = sqrt((x1-x2)**2+(y1-y2)**2+(z1-z2)**2)

    distance = round(distance, 3)

    return distance

def calculate_distance_between_ligand_and_residues_COG(pdbfile:str, chain:str, hetname: str, resn: list):
    """
    Calculates Euclidean distance between the center of geometry (COG) of a ligand and the COG of a list of protein
    residues of interest (or all protein residues, if residues not specified).
    Returns
        distance_dict (dict): Dictionary with distances (A) from each residue of interest COG to ligand COG (
        precision 3).
    """
    distance_dict = {}

    # If no residues are defined, extract all residue numbers for the protein
    if len(resn) == 0:
        resn = get_pdb_protein_residues(pdbfile,chain)

    # Extract coordinates for ligand and residues
    ligand_coordinates, residue_coordinates = get_pdb_coordinates(pdbfile, chain, hetname, resn)

    # Calculate COG for ligand and residue coordinates
    ligand_COG = COG(ligand_coordinates)
    residue_COG = {}

    for res in resn:
        residue_COG[res] = COG(residue_coordinates[res])

        # Calculate distances from ligand COG to each residues COG
        distance_dict[res] = distance_between_coordinates(ligand_COG, residue_COG[res])

    return distance_dict

def calculate_average_residue_distance_to_ligand(accession:str, resn:list, common:bool, pdb_dir:str, output_dir:str):
    """
    Calculates average distance from ligand COG to protein residues COG across PDB structures for a Uniprot accession of
    interest. Calculation is done for all residues in the protein, unless a list of residues is passed with resn.
    Returns:
        accession_average_dict (dict): Dictionary with average distances from each of the residues' COG to ligand's COG
    """

    output_file = os.path.join(output_dir, f'ligand_protein_distances_{accession}.json')

    if os.path.exists(output_file):
        with open(output_file) as json_file:
            print('Reading pre-computed distance results...')
            log_dict = json.load(json_file)
            # print(log_dict)

            accession_average_dist = log_dict[f'{accession}_average']['distance']
            # print(accession_average_dist)

    else:
        log_dict = {}

        accession_average_dist = {}
        pdb_dist_list = []

        # Get PDB codes for accession
        pdb_codes = map_accession_to_pdb(accession)
        pdbl = PDBList()

        # Get ligand ID for each PDB code. If there is a ligand present, download PDB and start analysis
        for pdb_code in pdb_codes:
            ligand_name = get_pdb_ligand(pdb_code)
            if ligand_name != None:

                # Start filling log dictionary (to write out)
                log_dict[pdb_code] = {}
                log_dict[pdb_code]['ligand'] = ligand_name

                # Download PDB file
                pdb_file_new = os.path.join(pdb_dir, f'{pdb_code}.pdb')
                if os.path.exists(pdb_file_new):
                    print(f'PDB structure {pdb_file_new} exists ...')
                else:
                    pdb_file_old = pdbl.retrieve_pdb_file(pdb_code, file_format='pdb', pdir=pdb_dir)
                    # Rename PDB file
                    os.rename(pdb_file_old, pdb_file_new)

                # Extract chain containing the ligand
                chain = get_pdb_ligand_chain(pdb_file_new, ligand_name)
                log_dict[pdb_code]['chain'] = chain

                # Calculate distance dictionary (calculate for all residues for reporting)
                pdb_dist = calculate_distance_between_ligand_and_residues_COG(pdb_file_new, chain, ligand_name, [])
                pdb_dist_list.append(pdb_dist)

                log_dict[pdb_code]['distance'] = pdb_dist

        if common:
            # Get common residues between structures (if more than one)
            common_res = set(pdb_dist_list[0].keys())
            if len(pdb_dist_list) > 1:
                for d in pdb_dist_list[1:]:
                    common_res.intersection_update(set(d.keys()))

            # Calculate mean distance for each residue across PDB files
            for res in common_res:
                accession_average_dist[str(res)] = round(sum(d[res] for d in pdb_dist_list) / len(pdb_dist_list),3)

        else:
            # Get all residues across structures (if more than one)
            all_res = set()
            if len(pdb_dist_list) > 1:
                for d in pdb_dist_list:
                    all_res.update(d.keys())

            # Calculate mean distance for each residue across PDB files
            for res in all_res:
                res_list = []
                for d in pdb_dist_list:
                    if res in d.keys():
                        res_list.append(d[res])
                res_avg = round(sum(res_list)/len(res_list), 3)
                accession_average_dist[str(res)] = res_avg

        # Write ouytput to json file
        log_dict[f'{accession}_average'] = {}
        log_dict[f'{accession}_average']['n_structures'] = len(pdb_dist_list)
        log_dict[f'{accession}_average']['distance'] = accession_average_dist

        with open(output_file, 'w') as outfile:
            json.dump(log_dict, outfile)

    # Return distance for residues of interest
    if len(resn) > 0:
        accession_average_dist_resn = dict((str(k), accession_average_dist[str(k)]) for k in resn if str(k) in
                                           accession_average_dist)

        return accession_average_dist_resn

    else:
        return accession_average_dist


