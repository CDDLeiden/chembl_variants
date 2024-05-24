# -*- coding: utf-8 -*-

"""Unit tests for Mutants-in-PCM(s modelling applicability domains."""

from mutants_in_pcm.data_path import set_data_path
from mutants_in_pcm.modelling import model_applicability_domains, model_external_molecular_applicability_domains

set_data_path(r'C:\Users\ojbeq\Documents\GitHub\mutants-in-pcm\data')

#model_applicability_domains()
model_external_molecular_applicability_domains(r'C:\Users\ojbeq\Dropbox\LACDR\PhD\Science\Projects\Mutants in PCM\Reference_dataset\Enamine_HLL_mold2.feather',
                                               njobs=1)
