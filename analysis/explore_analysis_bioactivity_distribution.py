from utils import *
from mutants_in_pcm import data_path

# Define ChEMBL and Papyrus versions
chembl_version = '31'
papyrus_version = '05.5'
papyrus_flavor = 'nostereo'

# Define data path
data_path.data_dir = '../data'

# Define annotation round
annotation_round = 2

# Define accession for analysis
accession_list = ['Q72547','P00533','O60885','O75874']

# Define directories for this annotation round
common_analysis_dir = get_mutant_analysis_path('1_mutant_statistics', 'common', annotation_round)

from mutants_in_pcm.mutant_analysis_distributions import *

# Define common subset options
common_subset_args = {'all':{'common':False,
                             'sim':False,
                             'sim_thres':None,
                             'threshold':None,
                             'variant_coverage':None},

                      'common_subset_20_sim_80':{'common':True,
                                                 'sim':True,
                                                 'sim_thres':0.8,
                                                 'threshold':2,
                                                 'variant_coverage':0.2}}

# Compute Wasserstein distances between WT and each mutant for all proteins
for common_subset_name, common_subset_arg in common_subset_args.items():
    print(f'Computing Wasserstein distances for common subset {common_subset_name}...')
    wasserstein_distance_statistics(output_dir=common_analysis_dir, **common_subset_arg)

# Compute Wasserstein distances for each variant between two subsets for all proteins
wasserstein_distance_dual_statistics(common_subset_args['all'],
                                     common_subset_args['common_subset_20_sim_80'],
                                     common_analysis_dir)
