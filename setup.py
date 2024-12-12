from setuptools import setup, find_packages

setup(
    name="dashboard-camara",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "streamlit",
        "pandas",
        "pyyaml",
    ],
)
