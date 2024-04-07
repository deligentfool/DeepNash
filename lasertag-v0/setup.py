from setuptools import setup, find_packages

setup(
    name='lasertag',
    version='1.0.0',
    install_requires=['gym', 'pycolab'],
    packages=find_packages(include=['figs', 'lasertag']),
)

