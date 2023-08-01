
import os

import pandas as pd

from preprocessing import merge_chembl_papyrus_mutants
from mutant_analysis_common_subsets import compute_stats_per_accession, extract_relevant_targets, compute_variant_activity_distribution, get_variant_similar_subset


if __name__ == '__main__':
    from tqdm.auto import tqdm
    pd.options.display.width = 0
    # Define output directory for mutant statistical analysis
    output_dir = r'C:\Users\ojbeq\Documents\GitHub\mutants-in-pcm\data'

    # Get data with mutants
    data_with_mutants = merge_chembl_papyrus_mutants('31', '05.5', 'nostereo', 1_000_000)

    for accession in tqdm(data_with_mutants['accession']):
        get_variant_similar_subset(data_with_mutants, accession, sim_thres=0.8, threshold=2, variant_coverage=0.2,
                                   save_dataset=True, output_dir=os.path.join(output_dir,'test_similar_datasets'))
