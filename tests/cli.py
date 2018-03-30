# -*- coding: utf-8 -*-

"""Console script for hyperopt_vw."""
import logging
import sys

import click
from hyperopt import hp
from hyperopt.mongoexp import MongoTrials, Trials

try:
    import seaborn as sns
except ImportError:
    print("Warning: seaborn is not installed. "
          "Without seaborn, standard matplotlib plots will not look very charming. "
          "It's recommended to install it via pip install seaborn")

from hyperopt_vw import Objective, search


@click.command()
@click.option('--train_data', type=click.Path())
@click.option('--validation_data', type=click.Path(), default=None)
@click.option('--test_data', type=click.Path(), default=None)
@click.option('--trials_output', type=click.Path(), default=None)
@click.option('--vw_args', type=click.STRING, default='')
@click.option('--outer_loss_function',
              type=click.Choice(['logistic', 'roc-auc', 'pr-auc', 'hinge', 'squared']),
              default='logistic')
@click.option('--mongo', type=click.STRING, default=None)
@click.option('--timeout', type=click.INT, default=None)
def main(train_data, validation_data, test_data, trials_output, vw_args, outer_loss_function, mongo, timeout):
    space = {
        '--passes': hp.quniform('passes', 1, 10, 1),
    }

    trials = MongoTrials(mongo, exp_key='exp1_test_3') if mongo else Trials()

    objective = Objective(train_data=train_data, validation_data=validation_data, test_data=test_data,
                          vw_args=vw_args, outer_loss_function=outer_loss_function, timeout=timeout)
    search(space, objective, trials=trials, trials_output=trials_output, max_evals=10)

    return 0


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    sys.exit(main())  # pragma: no cover
