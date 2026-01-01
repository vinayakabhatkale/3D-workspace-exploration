from setuptools import setup
from pathlib import Path

THISDIR = Path(__file__).parent

with open(THISDIR/'requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name="torch_points3d",
    author='vinayaka bhat',
    description='bhatkale45 training pipeline', 
#    url='https://github.com/cheind/pytorch-blender',
    license='MIT',
#   long_description=long_description,
#   long_description_content_type='text/markdown',
    version=open(THISDIR/'torch_points3d'/'__init__.py').readlines()[-1].split()[-1].strip('\''),
    packages=['torch_points3d'],    
    install_requires=required,
    zip_safe=False,
)

