from setuptools import find_packages, setup

setup(
    name='ml_project',
    packages=find_packages(),
    version='0.1.0',
    description='Homework 1.',
    author='Vlaskina Darya',
    install_requires=[
        "python-dotenv>=0.5.1",
        "pandas==1.4.2",
        "numpy==1.22.3"
        "PyYAML==6.0"
        "scikit-learn~=1.0.2",
        "sklearn==0.0",
        "marshmallow==3.15.0",
        "marshmallow-dataclass==8.5.8",
        "joblib==1.1.0",
        "typing-inspect==0.7.1",
        "hydra-core==1.1.2",
        "omegaconf~=2.1.2",
        "setuptools~=62.1.0",
    ],
)
