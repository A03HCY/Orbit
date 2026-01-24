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

def read_long_description() -> str:
    '''
    Read the README.md file for the long description.
    '''
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, 'README.md'), 'r', encoding='utf-8') as f:
        return f.read()

setup(
    name='orbit-torch',
    version=find_version(),
    description='A PyTorch training engine with plugin system and advanced model components',
    long_description=read_long_description(),
    long_description_content_type='text/markdown',
    author='Aiden Hopkins',
    author_email='acdphc@qq.com',
    url='https://github.com/A03HCY/Orbit',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'torch>=2.0.0',
        'rich',
        'tensorboard',
        'matplotlib',
        'seaborn',
        'numpy',
        'scikit-learn',
        'einops',
        'tokenizers',
        'transformers',
        'safetensors',
        'accelerate',
        'lpips'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
