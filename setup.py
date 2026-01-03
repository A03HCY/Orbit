import os
import re
from setuptools import setup, find_packages

def find_version() -> str:
    '''
    Read version from orbit/__init__.py.
    '''
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, 'orbit', '__init__.py'), 'r', encoding='utf-8') as f:
        version_file = f.read()
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError('Unable to find version string.')

setup(
    name='orbit-torch',
    version=find_version(),
    description='A PyTorch training engine with plugin system',
    author='Aiden Hopkins',
    author_email='acdphc@qq.com',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'torch>=1.10.0',
        'rich',
        'tensorboard',
        'matplotlib',
        'seaborn',
        'numpy',
        'scikit-learn',
        'einops',
        'tokenizers'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
