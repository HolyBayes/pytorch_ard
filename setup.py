__author__ = 'Artem Ryzhikov'

from setuptools import setup



setup(
    name="pytorch_ard",
    version='0.2.4',
    description="Make your PyTorch faster",
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
        'torch>=1.1.0',
        'torchvision>=0.2.1',
        'scikit-learn>=0.19.2',
        'pandas'
    ]
)
