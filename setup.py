from setuptools import setup, find_packages
import os

# Read the contents of your README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="tsseg-eval",
    version="0.1.2",
    description="Evaluation measures for time series segmentation",
    long_description=long_description,
    long_description_content_type='text/markdown',
    author="Félix Chavelli",
    url="https://github.com/fchavelli/seg-eval", 
    packages=find_packages(include=["segeval", "segeval.*"]),
    install_requires=[
        "numpy",
        "scipy",
        "scikit-learn",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
