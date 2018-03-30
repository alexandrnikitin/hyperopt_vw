# -*- coding: utf-8 -*-

"""Main module."""
import json
import logging

import numpy as np
from hyperopt import fmin, tpe


def search(space, objective, *, search_algorithm=tpe.suggest, trials=None, trials_output=None, max_evals=None):
    logger = logging.getLogger(__name__)

    best = fmin(
        objective,
        space=space,
        algo=search_algorithm,
        max_evals=max_evals,
        trials=trials)

    logger.info("the best hyper parameters: %s" % str(best))
    if trials_output:
        json.dump(trials.results, open(trials_output, 'w'))
        logger.info('All the trials results are saved at %s' % trials_output)

    best_validation_loss = trials.results[np.argmin(trials.losses())]['loss']
    best_validation_cmd = trials.results[np.argmin(trials.losses())]['train_cmd']
    best_validation_test_loss = trials.results[np.argmin(trials.losses())]['test_loss']
    logger.info("\n\nThe best validation loss value: \n%s\n\n" % best_validation_loss)
    logger.info("\n\nThe train loss of that run: \n%s\n\n" % best_validation_test_loss)
    logger.info("\n\nThe full training command: \n%s\n\n" % best_validation_cmd)

    # TODO best train loss

    return 0
