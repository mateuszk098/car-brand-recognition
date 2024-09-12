from setuptools import find_packages, setup

setup(
    name="recognition-app",
    version="1.0.0",
    package_dir={"": "."},
    packages=find_packages(where="."),
    include_package_data=True,
    package_data={"classification.config": ["*.yaml"]},
)
