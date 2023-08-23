import pandas as pd
from UniProtMapper import UniProtRetriever

def annotate_organism_taxonomy(data: pd.DataFrame):

    # Calculate statistics by organism
    stats_organism = data.groupby(['Organism'])['pchembl_value_Mean'].count()

    # Query from Uniprot by accession
    field_retriever = UniProtRetriever()

    fields = ["accession", "organism_name", "lineage", "virus_hosts"]
    result, failed = field_retriever.retrieveFields(data.accession.unique().tolist(),
                                                    fields=fields)
    # Keep unique organisms
    result = result[['Organism', 'Taxonomic lineage', 'Virus hosts']].drop_duplicates(subset='Organism')

    # Define domain classification based on taxonomy
    def tax_classification(x):
        if 'Archaea' in x:
            return 'Archaea'
        elif 'Bacteria' in x:
            return 'Bacteria'
        elif 'Viruses' in x:
            return 'Virus'
        elif 'Eukaryota' in x:
            return 'Eukaryota'
        else:
            return 'Other'

    result['classification'] = result['Taxonomic lineage'].apply(tax_classification)

    # Add taxonomical classification to each organism
    stats_organism_tax = stats_organism.reset_index().merge(result, how='left', on='Organism')

    # Manually fix classification labels that did not get a proper query from Uniprot
    classification_patch = {'Bacillus anthracis': 'Bacteria',
                            'Bacteroides thetaiotaomicron (strain ATCC 29148 / DSM 2079 / JCM 5827 / CCUG 10774 / NCTC '
                            '10582 / VPI-5482 / E50)': 'Bacteria',
                            'Francisella tularensis subsp. tularensis (strain SCHU S4 / Schu 4)': 'Bacteria',
                            'Hepacivirus C': 'Virus',
                            'Hepatitis B virus (HBV)': 'Virus',
                            'Hepatitis C virus subtype 1b': 'Virus',
                            'Human immunodeficiency virus 1': 'Virus',
                            'Human respiratory syncytial virus': 'Virus',
                            'Influenza A virus (A/Brisbane/59/2007(H1N1))': 'Virus',
                            'Leishmania infantum': 'Eukaryota',
                            'Leishmania major': 'Eukaryota',
                            'Magnetospirillum gryphiswaldense': 'Bacteria',
                            'Mycobacterium tuberculosis': 'Bacteria',
                            'Mycobacterium tuberculosis (strain CDC 1551 / Oshkosh)': 'Bacteria',
                            'Rhizobium leguminosarum bv. trifolii (strain WSM1325)': 'Bacteria',
                            'Arabidopsis thaliana (Mouse-ear cress)': 'Eukaryota',
                            'Anopheles gambiae (African malaria mosquito)': 'Eukaryota',
                            'Chlamydomonas reinhardtii (Chlamydomonas smithii)': 'Eukaryota',
                            'Crithidia fasciculata': 'Eukaryota',
                            'Magnaporthe oryzae (strain 70-15 / ATCC MYA-4617 / FGSC 8958) (Rice blast fungus) '
                            '(Pyricularia oryzae)': 'Eukaryota',
                            'Plasmodium falciparum (isolate 3D7)': 'Eukaryota',
                            'Plasmodium falciparum (isolate K1 / Thailand)': 'Eukaryota',
                            'Pneumocystis carinii': 'Eukaryota'
                            }
    stats_organism_tax['classification'] = stats_organism_tax['Organism'].map(classification_patch).fillna \
        (stats_organism_tax['classification'])

    # Annotate original data with taxonomical classification
    data_tax = data.merge(stats_organism_tax[['Organism', 'classification']], how='left', on='Organism')

    return data_tax