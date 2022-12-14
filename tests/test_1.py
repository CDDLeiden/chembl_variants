# -*- coding: utf-8 -*-

"""Tests for ...."""

import unittest

'''
import numpy as np
import orjson

from prodec import Descriptor
from prodec.utils import std_amino_acids
from tests.constants import *


class Test1(unittest.TestCase):
    """Tests for 1."""
    
    def setUp(self):
        """Create custom descriptor for testing"""
        # Values below obtained at random
        self.replacements = {'A': [6.0, 3.0, 12.1], 'C': [120.6, -12.2, -0.0001],
                             'D': [-3.8, 2.1, 0.9], 'E': [-91.24, -82.82, 77.79 ],
                             'F': [0.12, 3.0, 4.0], 'G': [-77.87, -75.44, 30.11 ],
                             'H': [47.73, -77.07, -34.83], 'I': [14.22, 97.13, -87.40 ],
                             'K': [ 64.98, 48.73, -62.02], 'L': [-13.98, -31.64, 5.04 ],
                             'M': [-78.57, -90.04, 17.60], 'N': [ -83.47, -58.43, -21.13 ],
                             'P': [-52.18, 52.58, 74.57], 'Q': [-36.21, -62.56, -56.08 ],
                             'R': [ 86.56, 26.16, 69.17], 'S': [34.89, -87.35, 37.82 ],
                             'T': [-13.67, -93.15, -2.88], 'V': [68.36, 67.60, -66.56 ],
                             'W': [-48.21, -42.47, -84.41], 'Y': [-90.24, 44.75, 2.18 ]}
        self.definition = CSTM_DATA_NULL
        for aa in std_amino_acids:
            self.definition = self.definition.replace(f'{aa}_value', f'"{aa}" : {self.replacements[aa]}')
        self.desc = Descriptor(orjson.loads(self.definition))

    def test_descriptor_loaded_ID(self):
        self.assertEqual(self.desc.ID, 'CSTM_TESTS')

    def test_descriptor_loaded_Type(self):
        self.assertEqual(self.desc.Type, 'Linear')

    def test_descriptor_loaded_Name(self):
        self.assertEqual(self.desc.Name, 'TEST')

    def test_descriptor_summary(self):
        self.assertEqual(self.desc.summary, {'Authors': None, 'Year':None,
                                             'Journal':None, 'DOI':None,
                                             'PMID':None, 'Patent':None})

    def test_descriptor_loaded_Scales_Names_type(self):
        self.assertIsInstance(self.desc.Scales_names, list)

    def test_descriptor_loaded_Scales_Names(self):
        self.assertListEqual(self.desc.Scales_names, [ 'v1', 'v2', 'v3' ])

    def test_descriptor_definition(self):
        self.assertEqual(self.desc.definition, self.replacements)

    def test_descriptor_is_sequence_valid_default(self):
        self.assertTrue(self.desc.is_sequence_valid(DFLT_SEQ))

    def test_descriptor_is_sequence_valid_stupid(self):
        self.assertFalse(self.desc.is_sequence_valid(STUPID_SEQ))

    def test_descriptor_get_flattened_type(self):
        result = self.desc.get(DFLT_SEQ, True, 0)
        self.assertIsInstance(result, list)

    def test_descriptor_get_not_flattened_type(self):
        result = self.desc.get(DFLT_SEQ, False, 0)
        self.assertIsInstance(result, list)

    def test_descriptor_get_flattened_shape(self):
        result = np.array(self.desc.get(DFLT_SEQ, True, 0)).shape
        self.assertEqual(result[0], 3 * len(DFLT_SEQ))

    def test_descriptor_get_not_flattened_type(self):
        result = np.array(self.desc.get(DFLT_SEQ, False, 0)).shape
        self.assertTrue(result[0] == len(DFLT_SEQ) and result[1] == 3)
    
    def test_descriptor_get_flattened_value(self):
        result = self.desc.get(DFLT_SEQ, True, 0)
        self.assertAlmostEqual(np.mean(result), -9.968835, 6)
        self.assertAlmostEqual(np.sum(result), -598.1301, 4)
        self.assertAlmostEqual(np.std(result), 56.753686, 6)
    
    def test_descriptor_get_not_flattened_value(self):
        result = self.desc.get(DFLT_SEQ, True, 0)
        self.assertAlmostEqual(np.mean(result), -9.968835, 6)
        self.assertAlmostEqual(np.sum(result), -598.1301, 4)
        self.assertAlmostEqual(np.std(result), 56.753686, 6)
    
    def test_descriptor_get_empty_value(self):
        result = self.desc.get('', True, 0)
        self.assertListEqual(result, [])

'''