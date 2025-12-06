from setuptools import setup, find_packages

setup(
    name="orbit-torch",
    version="0.0.2",
    description="A PyTorch training engine with plugin system",
    author="Aiden Hopkins",
    author_email="acdphc@qq.com",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "torch>=1.10.0",
        "rich",
        "tensorboard",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
