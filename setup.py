from setuptools import find_packages, setup

setup(
    name="car-recognition",
    version="0.1.0",
    package_dir={"": "."},
    packages=find_packages(where="."),
    include_package_data=True,
    package_data={"resnet.config": ["*.yaml"]},
)
