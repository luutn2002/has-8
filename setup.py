import os

import pkg_resources
from setuptools import setup, find_packages

setup(
    name="has_8",
    py_modules=["has_8"],
    version="0.0.1",
    description="Source code for HAS-8.",
    author="LUU Trong Nhan",
    author_email = "ltnhan0902@gmail.com",
    packages=find_packages(),
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ],
    include_package_data=True
)