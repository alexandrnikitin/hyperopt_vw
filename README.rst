===========
hyperopt vw
===========


.. image:: https://img.shields.io/pypi/v/hyperopt_vw.svg
        :target: https://pypi.python.org/pypi/hyperopt_vw

.. image:: https://img.shields.io/travis/alexandrnikitin/hyperopt_vw.svg
        :target: https://travis-ci.org/alexandrnikitin/hyperopt_vw

.. image:: https://readthedocs.org/projects/hyperopt-vw/badge/?version=latest
        :target: https://hyperopt-vw.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status




Hyperopt integration for Vowpal Wabbit


* Free software: MIT license
* Documentation: https://hyperopt-vw.readthedocs.io.


Features
--------

* TODO

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage



.. code-block:: bash
    --train_data=/root/alex/kaggle-talkingdata-adtracking-fraud-detection/data/processed/train.small.vw.00
    --test_data=/root/alex/kaggle-talkingdata-adtracking-fraud-detection/data/processed/test.small.vw.00
    --vw_space="--algorithms=ftrl,sgd --l2=1e-8..1e-1~LO --l1=1e-8..1e-1~LO -l=0.01..10~L --power_t=0.01..1 --ftrl_alpha=5e-5..8e-1~L --ftrl_beta=0.01..1 --passes=1..10~I -q=:: -b=29 --link=logistic --loss_function=logistic --hash=all --classweight=1:500"
    --mongo=mongo://10.3.14.9:27017/foo_db/jobs

