__author__ = 'Artem Ryzhikov'

from setuptools import setup

setup(
    name="pytorch_ard",
    version='0.1.0',
    description="Make your PyTorch bayesian",
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/HolyBayes/pytorch_ard',
    author='Artem Ryzhikov',

    packages=['torch_ard'],

    classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3 ',
    ],
    keywords='pytorch, bayesian neural networks, ard, deep learning, neural networks, machine learning',
    install_requires=[
        'torch>=0.4.0'
    ],
)
