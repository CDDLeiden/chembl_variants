# -*- coding: utf-8 -*-


"""Mutant statistics analysis. Part 1"""
"""Analyze mutant data available per family"""

import os
import chembl_downloader
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from textwrap import wrap
import math
from math import floor,ceil

from .preprocessing import merge_chembl_papyrus_mutants
from .annotation import chembl_annotation
from .data_path import get_data_path


def obtain_chembl_family(chembl_version: str, chunksize: int = None, data_folder: str = None):
    """Obtain family classifications (levels L1-L5) from ChEMBL using chembl-downloader.

    :param chembl_version: version of chembl to work with
    :param chunksize: size of chunks of data to be used (default: None)
    param data_folder: path to the folder in which the ChEMBL
    SQLite database is located or will be downloaded (default:
    pystow's default directory)
    :return: dataframe with ChEMBL L1-l5 family level annotations per accession
    """
    if data_folder is not None:
        os.environ['PYSTOW_HOME'] = data_folder

    data_dir = get_data_path()
    chembl_file = os.path.join(data_dir,f'chembl{chembl_version}_families.csv')
    if not os.path.isfile(chembl_file):

        query = """
            SELECT component_sequences.accession,
            protein_family_classification.l1,protein_family_classification.l2,protein_family_classification.l3,
            protein_family_classification.l4,protein_family_classification.l5
            FROM component_sequences
                LEFT JOIN component_class USING (component_id)
                LEFT JOIN protein_family_classification USING (protein_class_id)
            """

        chembl_families = chembl_downloader.query(query, version=chembl_version,
                                              prefix=['mutants-in-pcm', 'chembl'])

        # Make sure every accession appears only once. If it does not, keep the most annotated entry
        nan_counts = chembl_families.drop('accession', axis=1).isna().sum(axis=1)
        chembl_families['nan_counts'] = nan_counts

        chembl_families_unique = chembl_families.sort_values('nan_counts', ascending=True).drop_duplicates('accession')
        chembl_families_unique.drop('nan_counts', axis=1, inplace=True)

        chembl_families_unique.to_csv(chembl_file, sep='\t', index=None)

    else:
        chembl_families_unique = pd.read_csv(chembl_file, sep='\t')

    return chembl_families_unique

def group_families(chembl_families: pd.DataFrame):
    """
    Group smaller families in bigger families to avoid cluttering the plot
    :param chembl_families: dataframe with ChEMBL L1-l5 family level annotations per accession
    :return: updated dataframe with ChEMBL family level annotations per accession
    """
    # Define levels as "Other" if not available
    chembl_families[['l1', 'l2', 'l3', 'l4', 'l5']] = chembl_families[['l1', 'l2', 'l3', 'l4', 'l5']].fillna('Other')

    # Group small L1 families into "Other" category
    chembl_families['l1'] = chembl_families['l1'].apply(
        lambda x: 'Other' if any([match in x for match in ['Other', 'Auxiliary', 'Unclassified', 'Structural',
                                                           'Surface']])
        else x)

    # Group all GPCRs into a single L2 family
    chembl_families['l2'] = chembl_families['l2'].apply(lambda x: 'GPCR' if any([match in x for match in
                                                                                 ['G protein-coupled']]) else x)

    # Group small L2 families into "Other" category
    chembl_families['l2'] = chembl_families['l2'].apply(lambda x: 'Other' if any([match in x for match in
                                                                                  ['Primary active',
                                                                                   'Other', 'Ligase',
                                                                                   'Isomerase', 'Writer']]) else x)

    return chembl_families

def link_bioactivity_to_family(data: pd.DataFrame, chembl_families: pd.DataFrame):
    """
    Merge bioactivity data to its ChEMBL family level L1-L5 based on accession code
    :param data: dataframe with bioactivity data and accession codes
    :param chembl_families: dataframe with ChEMBL family level annotations per accession
    :return: dataframe with bioactivity data and its ChEMBL family level annotations
    """
    annotated_data_families = pd.merge(data, chembl_families, how='inner', on='accession')

    return annotated_data_families

def plot_circular_barplot_families(annotated_data_families: pd.DataFrame, family_level: str, output_dir: str,
                                   subset_level: str = None, subset_family: str = None, save: bool = False,
                                   figure_panel: bool = False):
    """
    Plot circular barplots representing the amount of bioactivity datapoints available per family within a specific
    ChEMBL family level (l1-l5)
    :param annotated_data_families: dataframe with bioactivity data and its ChEMBL family level annotations
    :param family_level: Family level (l1-l5) to consider for plotting
    :param output_dir: Path to output directory
    :param subset_level: Family level to filter on
    :param subset_family: Family name from level defined in "subset_level" to filter on
    :param save: whether to save the dataframe and figure
    :param figure_panel: Whether the figure is to be a panel (i.e. make labels bigger)
    :return: dataframe with family statistics for plotting and plot (saved as .svg)
    """
    # Define subset (e.g. l2 = Kinase)
    if subset_level != None:
        annotated_data_families = annotated_data_families[annotated_data_families[subset_level] == subset_family]
        file_tag = f'_subset_{subset_level}_{subset_family.replace(" ","-")}'
    else:
        file_tag = ''

    # Define options for making figure a panel (bigger font size)
    if figure_panel:
        label_size = 22
        medium_label_size = 18
        small_label_size = 16
        label_pad = -60
        cbar_x, cbar_y, cbar_w, cbar_h = 0.175, 0.1, 0.70, 0.02
        figure_tag = file_tag + '_panel'
    else:
        label_size = 12
        medium_label_size = 10
        small_label_size = 10
        label_pad = -40
        cbar_x, cbar_y, cbar_w, cbar_h = 0.325, 0.1, 0.35, 0.01
        figure_tag = file_tag

    # Count number of bioactivity datapoints in total and in non-WT variants
    # Here undefined mutants (_MUTANT) are considered mutants
    activity_mut = annotated_data_families[~annotated_data_families['target_id'].str.contains('WT')].groupby([family_level]).count()[['CID']].rename(columns={'CID':'activity_mut'})

    activity_all = annotated_data_families.groupby([family_level]).count()[['CID']].rename(columns={'CID':'activity_all'})

    summary_all = pd.concat([activity_mut,activity_all],axis=1)

    # Calculate ratio of mutated datapoints respect to all datapoints
    summary_all['mut_ratio'] = summary_all['activity_mut']/summary_all['activity_all']
    if save:
        summary_all.to_csv(os.path.join(output_dir, f'family_stats_{family_level}{file_tag}.csv'), sep='\t')
    print(summary_all)

    # Calculate maximum log exponent that will be plotted from the data and the range leading to it
    max_value = max(summary_all['activity_all'])
    max_log_exp = int("{:e}".format(max_value).split('+')[1])
    log_exps = range(0,max_log_exp+1,1)

    # Bars are sorted
    df_sorted = summary_all.sort_values("activity_all", ascending=False)
    # Values for the x axis
    ANGLES = np.linspace(0.05, 2 * np.pi - 0.05, len(df_sorted), endpoint=False)
    # Activity values
    ACTIVITIES = df_sorted["activity_all"].values
    # Mutant activity values
    MUTANT_ACTIVITIES = df_sorted["activity_mut"].values
    # Family label
    FAMILY = df_sorted.index.tolist()
    # Mutant ratio
    RATIO = df_sorted["mut_ratio"]

    GREY12 = "#1f1f1f"

    # Set default font to Bell MT
    if not figure_panel:
        plt.rcParams.update({"font.family": "Bell MT"})
        sns.set_context("notebook")
    else:
        plt.rcParams.update({"font.family": "sans-serif"})
        sns.set_context("paper")

    # Set default font color to GREY12
    plt.rcParams["text.color"] = GREY12

    # The minus glyph is not available in Bell MT
    # This disables it, and uses a hyphen
    plt.rc("axes", unicode_minus=False)

    # Colors
    COLORS = ["#6C5B7B","#C06C84","#F67280","#F8B195"]

    # Colormap
    cmap = mpl.colors.LinearSegmentedColormap.from_list("my color", COLORS, N=256)

    # Normalizer
    def round_half_up(n, decimals=0):
        multiplier = 10 ** decimals

        return math.floor(n*multiplier + 0.5) / multiplier
    norm = mpl.colors.Normalize(vmin=floor(RATIO.min()), vmax=round_half_up(RATIO.max(),1))

    # Normalized colors. Each number of tracks is mapped to a color in the
    # color scale 'cmap'
    COLORS = cmap(norm(RATIO))

    # Some layout stuff ----------------------------------------------
    # Initialize layout in polar coordinates
    fig, ax = plt.subplots(figsize=(9, 12.6), subplot_kw={"projection": "polar"})
    plt.yscale('symlog') # Symmetric logaritmic scale

    # Set background color to white, both axis and figure.
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.set_theta_offset(1.2 * np.pi / 2)

    # Add geometries to the plot -------------------------------------
    # See the zorder to manipulate which geometries are on top

    # Add bars to represent the total activity datapoints
    ax.bar(ANGLES, ACTIVITIES, color=COLORS, alpha=0.9, width=0.52, zorder=10)

    # Add dashed vertical lines. These are just references
    ax.vlines(ANGLES, 0, 10**max_log_exp, color=GREY12, ls=(0, (4, 4)), zorder=11)

    # Add dots to represent the mutated activity datapoints
    ax.scatter(ANGLES, MUTANT_ACTIVITIES, s=60, color=GREY12, zorder=11)


    # Add labels for the families -------------------------------------
    # Note the 'wrap()' function.
    # The '5' means we want at most 5 consecutive letters in a word,
    # but the 'break_long_words' means we don't want to break words
    # longer than 5 characters.
    FAMILY = ["\n".join(wrap(r, 15, break_long_words=False)) for r in FAMILY]
    FAMILY

    # Set the labels
    ax.set_xticks(ANGLES)
    ax.set_xticklabels(FAMILY, size=label_size)

    # Customize guides and annotations
    # Remove lines for polar axis (x)
    ax.xaxis.grid(False)

    # Put grid lines for radial axis (y) at 1, 10, 100, 1000, and 10000 (...) depending on the data
    ax.set_yticklabels([])
    y_ticks = [10**exp for exp in log_exps]
    ax.set_yticks(y_ticks)

    # Remove spines
    ax.spines["start"].set_color("none")
    ax.spines["polar"].set_color("none")


    # Adjust padding of the x axis labels ----------------------------
    # This is going to add extra space around the labels for the
    # ticks of the x axis.
    XTICKS = ax.xaxis.get_major_ticks()
    for tick in XTICKS:
        tick.set_pad(10)

    # Add custom annotations -----------------------------------------
    # The following represent the heights in the values of the y axis
    for log_exp in log_exps:
        PAD = 0
        POS = 10**log_exp
        ax.text(-0.2 * np.pi / 2, POS + PAD, f"10$^{log_exp}$", ha="center", size=medium_label_size)

    # Add text to explain the meaning of the height of the bar and the
    # height of the dot
    ax.text(ANGLES[0]-0.042, max_value, "Total bioactivity\npoints (Log scale)", rotation=21,
            ha="center", va="top", size=small_label_size, zorder=12)
    ax.text(ANGLES[0]+ 0.012, (10**(max_log_exp-2)), "Mutant bioactivity points\n(Log scale)", rotation=-69,
            ha="center", va="center", size=small_label_size, zorder=12)

    # Add legend -----------------------------------------------------

    # First, make some room for the legend and the caption in the bottom.
    fig.subplots_adjust(bottom=0.175)

    # Create an inset axes.
    # Width and height are given by the (cbar_w, cbar_h) in the
    # bbox_to_anchor
    cbaxes = inset_axes(
        ax,
        width="100%",
        height="100%",
        loc="center",
        bbox_to_anchor=(cbar_x, cbar_y, cbar_w, cbar_h),
        bbox_transform=fig.transFigure # Note it uses the figure.
    )

    # Create the colorbar
    cb = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        cax=cbaxes, # Use the inset_axes created above
        orientation = "horizontal"
    )

    # Remove the outline of the colorbar
    cb.outline.set_visible(False)

    # Remove tick marks
    cb.ax.xaxis.set_tick_params(size=0, labelsize=medium_label_size)

    # Set legend label and move it to the top (instead of default bottom)
    cb.set_label("Mutant ratio", size=label_size, labelpad=label_pad)

    # Add annotations ------------------------------------------------

    # Make some room for the title and subtitle above.
    fig.subplots_adjust(top=0.8)

    # Define title, subtitle, and caption
    title = "\nBioactivity datapoints distribution\naccross protein families"
    subtitle = "\n".join([
        f"Level of protein family definition (ChEMBL) = {family_level}.\n",
        "The following data was calculated for the subset:",
        f"Family level {subset_level} = {subset_family}."
    ])
    # caption = "Insert caption here"

    # And finally, add them to the plot.
    fig.text(0.1, 0.93, title, fontsize=25, weight="bold", ha="left", va="baseline")
    fig.text(0.1, 0.9, subtitle, fontsize=14, ha="left", va="top")
    # fig.text(0.5, 0.025, caption, fontsize=10, ha="center", va="baseline")

    if save:
        # Save figure
        fig.savefig(os.path.join(output_dir, f'family_stats_{family_level}{figure_tag}.svg'))

    plt.close()

    if save:
        return os.path.join(output_dir, f'family_stats_{family_level}{figure_tag}.svg')

def plot_circular_barplot_families_newannotations(annotated_data_families: pd.DataFrame, family_level: str,
                                                  output_dir: str, subset_level: str = None, subset_family: str = None,
                                                  save: bool = False, figure_panel: bool = False):
    """
        Plot circular barplots representing the amount of newly annotated mutant bioactivity datapoints available per
        family within a specific ChEMBL family level (l1-l5)
        :param annotated_data_families: dataframe with bioactivity data and its ChEMBL family level annotations
        :param family_level: Family level (l1-l5) to consider for plotting
        :param output_dir: Path to output directory
        :param subset_level: Family level to filter on
        :param subset_family: Family name from level defined in "subset_level" to filter on
        :param save: whether to save the dataframe and figure
        :param figure_panel: Whether the figure is to be a panel (i.e. make labels bigger)
        :return: dataframe with family statistics for plotting and plot (saved as .svg)
        """

    # Define subset (e.g. l2 = Kinase)
    if subset_level != None:
        annotated_data_families = annotated_data_families[annotated_data_families[subset_level] == subset_family]
        file_tag = f'_subset_{subset_level}_{subset_family.replace(" ","-")}'
    else:
        file_tag = ''

    # Define options for making figure a panel (bigger font size)
    if figure_panel:
        label_size = 22
        medium_label_size = 18
        small_label_size = 16
        label_pad = -60
        cbar_x, cbar_y, cbar_w, cbar_h = 0.175, 0.1, 0.70, 0.02
        figure_tag = file_tag + '_panel'
    else:
        label_size = 12
        medium_label_size = 10
        small_label_size = 10
        label_pad = -40
        cbar_x, cbar_y, cbar_w, cbar_h = 0.325, 0.1, 0.35, 0.01
        figure_tag = file_tag

    # Count number of WT bioactivity datapoints in total and those that were not previously defined in ChEMBL
    # Here undefined mutants (_MUTANT) are not considered as new annotations as they are not precise enough
    mut_new = annotated_data_families[~(annotated_data_families['target_id'].str.contains('WT') |
                                        annotated_data_families['target_id'].str.contains('MUTANT')) &
                                      (annotated_data_families['mutation'].isna() |
                                       annotated_data_families['mutation'].str.contains('UNDEFINED_MUTANT'))].groupby([
        family_level]).count()[['chembl_id']].rename(columns={'chembl_id':'mut_new'})

    mut_all = annotated_data_families[~(annotated_data_families['target_id'].str.contains('WT'))
                                      | annotated_data_families['target_id'].str.contains('MUTANT')].\
        groupby([family_level]).count()[['chembl_id']].rename(columns={'chembl_id':'mut_all'})

    summary_all = pd.concat([mut_new,mut_all],axis=1)

    # Calculate ratio of mutated datapoints respect to all datapoints
    summary_all['mut_ratio'] = summary_all['mut_new']/summary_all['mut_all']
    summary_all['mut_ratio'] = summary_all['mut_ratio'].fillna(value=0)
    if save:
        summary_all.to_csv(os.path.join(output_dir, f'family_stats_ChEMBLNewAnnotations_{family_level}{file_tag}.csv'), sep='\t')
    print(summary_all)

    # Calculate maximum log exponent that will be plotted from the data and the range leading to it
    max_value = max(summary_all['mut_all'])
    max_log_exp = int("{:e}".format(max_value).split('+')[1])
    log_exps = range(0,max_log_exp+1,1)

    # Bars are sorted
    df_sorted = summary_all.sort_values("mut_all", ascending=False)
    # Values for the x axis
    ANGLES = np.linspace(0.05, 2 * np.pi - 0.05, len(df_sorted), endpoint=False)
    # Mutant activity values
    MUTANT_ACTIVITIES = df_sorted["mut_all"].values
    # Newly annotated mutant activity values
    NEW_MUTANT_ACTIVITIES = df_sorted["mut_new"].values
    # Family label
    FAMILY = df_sorted.index.tolist()
    # Mutant ratio
    RATIO = df_sorted["mut_ratio"]

    GREY12 = "#1f1f1f"

    # Set default font to Bell MT
    if not figure_panel:
        plt.rcParams.update({"font.family": "Bell MT"})
        sns.set_context("notebook")
    else:
        plt.rcParams.update({"font.family": "sans-serif"})
        sns.set_context("paper")

    # Set default font color to GREY12
    plt.rcParams["text.color"] = GREY12

    # The minus glyph is not available in Bell MT
    # This disables it, and uses a hyphen
    plt.rc("axes", unicode_minus=False)

    # Colors
    COLORS = sns.color_palette("mako", as_cmap=False)[1:]

    # Colormap
    cmap = mpl.colors.LinearSegmentedColormap.from_list("my color", COLORS, N=256)

    # Normalizer
    def round_half_up(n, decimals=0):
        multiplier = 10 ** decimals

        return floor(n*multiplier + 0.5) / multiplier
    norm = mpl.colors.Normalize(vmin=floor(RATIO.min()), vmax=round_half_up(RATIO.max(),1))

    # Normalized colors. Each number of tracks is mapped to a color in the
    # color scale 'cmap'
    COLORS = cmap(norm(RATIO))

    # Some layout stuff ----------------------------------------------
    # Initialize layout in polar coordinates
    fig, ax = plt.subplots(figsize=(9, 12.6), subplot_kw={"projection": "polar"})
    plt.yscale('symlog') # Symmetric logaritmic scale

    # Set background color to white, both axis and figure.
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.set_theta_offset(1.2 * np.pi / 2)

    # Add geometries to the plot -------------------------------------
    # See the zorder to manipulate which geometries are on top

    # Add bars to represent the mutant activities
    ax.bar(ANGLES, MUTANT_ACTIVITIES, color=COLORS, alpha=0.9, width=0.52, zorder=10)

    # Add dashed vertical lines. These are just references
    ax.vlines(ANGLES, 0, 10**max_log_exp, color=GREY12, ls=(0, (4, 4)), zorder=11)

    # Add dots to represent the newly annotated mutant activities
    ax.scatter(ANGLES, NEW_MUTANT_ACTIVITIES, s=60, color=GREY12, zorder=11)


    # Add labels for the families -------------------------------------
    # Note the 'wrap()' function.
    # The '5' means we want at most 5 consecutive letters in a word,
    # but the 'break_long_words' means we don't want to break words
    # longer than 5 characters.
    FAMILY = ["\n".join(wrap(r, 15, break_long_words=False)) for r in FAMILY]
    FAMILY

    # Set the labels
    ax.set_xticks(ANGLES)
    ax.set_xticklabels(FAMILY, size=label_size)

    # Customize guides and annotations
    # Remove lines for polar axis (x)
    ax.xaxis.grid(False)

    # Put grid lines for radial axis (y) at 1, 10, 100, 1000, and 10000 (...) depending on the data
    ax.set_yticklabels([])
    y_ticks = [10**exp for exp in log_exps]
    ax.set_yticks(y_ticks)

    # Remove spines
    ax.spines["start"].set_color("none")
    ax.spines["polar"].set_color("none")


    # Adjust padding of the x axis labels ----------------------------
    # This is going to add extra space around the labels for the
    # ticks of the x axis.
    XTICKS = ax.xaxis.get_major_ticks()
    for tick in XTICKS:
        tick.set_pad(10)

    # Add custom annotations -----------------------------------------
    # The following represent the heights in the values of the y axis
    for log_exp in log_exps:
        PAD = 0
        POS = 10**log_exp
        ax.text(-0.2 * np.pi / 2, POS + PAD, f"10$^{log_exp}$", ha="center", size=medium_label_size)

    # Add text to explain the meaning of the height of the bar and the
    # height of the dot
    ax.text(ANGLES[0]-0.042, max_value, "Total mutant bioactivity\npoints (Log scale)", rotation=21,
            ha="center", va="top", size=small_label_size, zorder=12)
    ax.text(ANGLES[0]+ 0.012, (10**(max_log_exp-2)), "New mutant bioactivity points\n(Log scale)", rotation=-69,
            ha="center", va="center", size=small_label_size, zorder=12)

    # Add legend -----------------------------------------------------

    # First, make some room for the legend and the caption in the bottom.
    fig.subplots_adjust(bottom=0.175)

    # Create an inset axes.
    # Width and height are given by the (0.35 and 0.01) in the
    # bbox_to_anchor
    cbaxes = inset_axes(
        ax,
        width="100%",
        height="100%",
        loc="center",
        bbox_to_anchor=(cbar_x, cbar_y, cbar_w, cbar_h),
        bbox_transform=fig.transFigure # Note it uses the figure.
    )

    # Create the colorbar
    cb = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        cax=cbaxes, # Use the inset_axes created above
        orientation = "horizontal"
    )

    # Remove the outline of the colorbar
    cb.outline.set_visible(False)

    # Remove tick marks
    cb.ax.xaxis.set_tick_params(size=0, labelsize=medium_label_size)

    # Set legend label and move it to the top (instead of default bottom)
    cb.set_label("New mutant annotation ratio", size=label_size, labelpad=label_pad)

    # Add annotations ------------------------------------------------

    # Make some room for the title and subtitle above.
    fig.subplots_adjust(top=0.8)

    # Define title, subtitle, and caption
    title = "\nChEMBL mutant bioactivity datapoints\ndistribution accross protein families"
    subtitle = "\n".join([
        f"Level of protein family definition (ChEMBL) = {family_level}.\n",
        "The following data was calculated for the subset:",
        f"Family level {subset_level} = {subset_family}."
    ])
    # caption = "Insert caption here"

    # And finally, add them to the plot.
    fig.text(0.1, 0.93, title, fontsize=25, weight="bold", ha="left", va="baseline")
    fig.text(0.1, 0.9, subtitle, fontsize=14, ha="left", va="top")
    # fig.text(0.5, 0.025, caption, fontsize=10, ha="center", va="baseline")

    if save:
        # Save figure
        fig.savefig(os.path.join(output_dir, f'family_stats_ChEMBLNewAnnotations_{family_level}{figure_tag}.svg'))

    plt.close()

    if save:
        return os.path.join(output_dir, f'family_stats_ChEMBLNewAnnotations_{family_level}{figure_tag}.svg')

if __name__ == "__main__":
    annotation_round = 1
    chembl_version, papyrus_version, papyrus_flavor = ['31', '05.5', 'nostereo']

    output_dir = f'C:\\Users\gorostiolam\Documents\Gorostiola Gonzalez, ' \
                 f'Marina\PROJECTS\\6_Mutants_PCM\DATA\\2_Analysis\\1_mutant_statistics\\0_family_stats\\' \
                 f'round_{annotation_round}'

    # Read ChEMBL family levels
    chembl_families = group_families(obtain_chembl_family(chembl_version))

    # Read annotated bioactivity data with mutants (ChEMBL + Papyrus, at least one variant defined per target)
    annotated_data = merge_chembl_papyrus_mutants(chembl_version, papyrus_version, papyrus_flavor, 1_000_000, annotation_round)

    # Read ChEMBL-only annotated bioactivity data for variants
    chembl_annotated_data = chembl_annotation(chembl_version, annotation_round)

    # Add family annotations
    annotated_data_families = link_bioactivity_to_family(annotated_data, chembl_families)
    chembl_annotated_data_families = link_bioactivity_to_family(chembl_annotated_data, chembl_families)

    # Plot circular barplots
    for figure_panel in [True,False]:
        # Bioactivity data
        plot_circular_barplot_families(annotated_data_families, 'l1', output_dir, subset_level=None, subset_family=None,
                                       save=True, figure_panel=figure_panel)
        plot_circular_barplot_families(annotated_data_families, 'l4', output_dir, subset_level='l2',
                                       subset_family='Kinase', save=True, figure_panel=figure_panel)
        plot_circular_barplot_families(annotated_data_families, 'l4', output_dir, subset_level='l2',
                                       subset_family='GPCR', save=True, figure_panel=figure_panel)
        # Mutant annotations
        plot_circular_barplot_families_newannotations(chembl_annotated_data_families, 'l1', output_dir,
                                                      subset_level=None,subset_family=None,
                                                      save=True, figure_panel=figure_panel)
        plot_circular_barplot_families_newannotations(chembl_annotated_data_families, 'l4', output_dir,
                                                      subset_level='l2',subset_family='Kinase',
                                                      save=True, figure_panel=figure_panel)
        plot_circular_barplot_families_newannotations(chembl_annotated_data_families, 'l4', output_dir,
                                                      subset_level='l2',subset_family='GPCR',
                                                      save=True, figure_panel=figure_panel)

