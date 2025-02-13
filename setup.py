from setuptools import find_packages, setup

setup(
    name="feature_importance",
    version="0.1",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "torch",
        "botorch",
        "ax-platform",
    ],
)
