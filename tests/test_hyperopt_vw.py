#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `hyperopt_vw` package."""

import unittest

import pickle

import os
from click.testing import CliRunner

import hyperopt_vw.objective
from hyperopt_vw import search
from hyperopt_vw import cli


class TestObjective(unittest.TestCase):

    def test_pickle_objective(self):
        o = search.objective.Objective('', '', '', '', '', 1000)
        msg = pickle.dumps(o)
        o = pickle.loads(msg)
        o()

    def test_load_y_true(self):
        arr = search.load_y_true("test.vw")
        assert len(arr) == 5

    def test_validate(self):
        y_true = search.load_y_true("test.small.vw.00")
        y_score = search.load_y_score("score.txt")
        roc = search.calculate_loss(y_true, y_score, 'roc-auc')
        print(roc)

# class TestHyperopt_vw(unittest.TestCase):
#     """Tests for `hyperopt_vw` package."""
#
#     def setUp(self):
#         """Set up test fixtures, if any."""
#         self.project_dir = os.path.join(os.path.dirname(__file__), os.pardir)
#
#     def tearDown(self):
#         """Tear down test fixtures, if any."""
#
#     def test_000_something(self):
#         """Test something."""
#
#     def test_command_line_interface(self):
#         """Test the CLI."""
#         runner = CliRunner()
#         train_data = os.path.join(self.project_dir, 'tests', 'train.vw')
#         test_data = os.path.join(self.project_dir, 'tests', 'test.vw')
#         vw_space = '\'--algorithms=ftrl,sgd --l2=1e-8..1e-1~LO --l1=1e-8..1e-1~LO -l=0.01..10~L --power_t=0.01..1 ' \
#                    '--ftrl_alpha=5e-5..8e-1~L --ftrl_beta=0.01..1 --passes=1..10~I -q=:: -b=29 --link=logistic ' \
#                    '--loss_function=logistic --hash=all\''
#         args = [train_data, test_data, vw_space]
#
#         result = runner.invoke(cli.main, args=args)
#         print(result.output)
#         assert result.exit_code == 0
#         assert 'hyperopt_vw.cli.main' in result.output
#         help_result = runner.invoke(cli.main, ['--help'])
#         assert help_result.exit_code == 0
#         assert '--help  Show this message and exit.' in help_result.output


if __name__ == '__main__':
    unittest.main()
