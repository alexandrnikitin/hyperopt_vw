import logging
import os
import shlex
import subprocess
import uuid
from math import exp

import numpy as np
from hyperopt import STATUS_OK
from sklearn.metrics import roc_curve, auc, log_loss, average_precision_score, hinge_loss, mean_squared_error


class Objective(object):
    def __init__(self, *, train_data, validation_data=None, test_data=None, vw_args, outer_loss_function,
                 timeout=None) -> None:
        super().__init__()
        self.logger = logging.getLogger(__name__)

        self.train_data = train_data
        self.validation_data = validation_data
        self.test_data = test_data

        self.vw_args = vw_args
        self.outer_loss_function = outer_loss_function
        self.timeout = timeout

    def __call__(self, *args, **kwargs):
        run_id = str(uuid.uuid4())
        vw_hyper_args = self.get_hyper_args(**args[0])

        train_model = f'model.{run_id}'
        train_data_cache = f'{os.path.basename(self.train_data)}.cache.{run_id}'
        train_cmd = compose_train_cmd(self.train_data, train_data_cache, train_model, self.vw_args, vw_hyper_args)
        self.logger.info(f"train_cmd: {train_cmd}")
        subprocess.call(shlex.split(train_cmd), timeout=self.timeout)

        validation_predictions = f'{os.path.basename(self.validation_data)}.predictions.{run_id}'
        validation_cmd = compose_test_cmd(self.validation_data, train_model, validation_predictions)
        self.logger.info(f"validation_cmd: {validation_cmd}")
        subprocess.call(shlex.split(validation_cmd), timeout=self.timeout)
        validation_loss = self.calculate_loss(self.validation_data, validation_predictions)

        test_predictions = f'{os.path.basename(self.test_data)}.predictions.{run_id}'
        test_cmd = compose_test_cmd(self.test_data, train_model, test_predictions)
        self.logger.info(f"test_cmd: {test_cmd}")
        subprocess.call(shlex.split(test_cmd), timeout=self.timeout)
        test_loss = self.calculate_loss(self.test_data, test_predictions)

        # TODO handle fails
        return {
            'status': STATUS_OK,
            'loss': validation_loss,

            'validation_loss': validation_loss,
            'test_loss': test_loss,
            'train_cmd': train_cmd,
            'validation_cmd': validation_cmd,
            'test_cmd': test_cmd,
            'current_trial': run_id
        }

    def calculate_loss(self, data_file, predictions_file) -> float:
        y_true = np.loadtxt(data_file, dtype=np.float, delimiter=' ', usecols=[0])
        y_score = np.loadtxt(predictions_file, dtype=np.float, delimiter=' ', usecols=[0])

        loss = calculate_loss(y_true, y_score, self.outer_loss_function)
        self.logger.info('loss value: %.6f' % loss)
        return loss

    def __getstate__(self):
        d = self.__dict__.copy()
        if 'logger' in d:
            d['logger'] = d['logger'].name
        return d

    def __setstate__(self, d):
        if 'logger' in d:
            d['logger'] = logging.getLogger(d['logger'])
        self.__dict__.update(d)

    def get_hyper_args(self, **kwargs):
        return get_hyper_args(**kwargs)


def get_hyper_args(**kwargs):
    for arg in ['--passes']:
        if arg in kwargs:
            kwargs[arg] = int(kwargs[arg])

    args = ''
    for key in kwargs:
        if key.startswith('-'):
            if key.startswith('--classweight'):
                args += ' %s%s' % (key, kwargs[key])
            else:
                args += ' %s %s' % (key, kwargs[key])

    return args


def compose_train_cmd(data, cache_file, final_regressor, vw_args, vw_hyper_args):
    return f"vw --data {data} --cache_file {cache_file} --final_regressor {final_regressor} " \
           f"{vw_args} {vw_hyper_args}"


def compose_test_cmd(test_data, train_model, test_score):
    return f"vw --testonly --data {test_data} --initial_regressor {train_model} " \
           f"--predictions {test_score}"


def calculate_loss(y_true, y_score, outer_loss_function):
    if outer_loss_function == 'logistic':
        y_pred_holdout_proba = [1. / (1 + exp(-i)) for i in y_score]
        loss = log_loss(y_true, y_pred_holdout_proba)
    elif outer_loss_function == 'squared':
        loss = mean_squared_error(y_true, y_score)
    elif outer_loss_function == 'hinge':
        loss = hinge_loss(y_true, y_score)
    elif outer_loss_function == 'pr-auc':
        loss = -average_precision_score(y_true, y_score)
    elif outer_loss_function == 'roc-auc':
        fpr, tpr, _ = roc_curve(y_true, y_score)
        loss = 1 - auc(fpr, tpr)
    return loss
