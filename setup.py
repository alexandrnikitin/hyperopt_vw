#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""
import os
from setuptools import setup, find_packages, findall

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['Click>=6.0', 'hyperopt', 'scikit-learn', 'numpy', 'matplotlib', 'seaborn']

setup_requirements = [ ]

test_requirements = [ ]


def find_scripts():
    return [s for s in findall('scripts/') if os.path.splitext(s)[1] != '.pyc']


setup(
    author="Alexandr Nikitin",
    author_email='nikitin.alexandr.a@gmail.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    description="Hyperopt integration for Vowpal Wabbit",
    # entry_points={
    #     'console_scripts': [
    #         'hyperopt_vw=hyperopt_vw.cli:main',
    #     ],
    # },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='hyperopt_vw',
    name='hyperopt_vw',
    packages=find_packages(include=['hyperopt_vw']),
    scripts=find_scripts(),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/alexandrnikitin/hyperopt_vw',
    version='0.1.0',
    zip_safe=False,
)
