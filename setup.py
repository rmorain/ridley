from setuptools import find_packages, setup

setup(
    name="ridley",
    version="1.0.0",
    url="https://github.com/rmorain/ridley",
    author="Robert Morain",
    author_email="robert.morain@gmail.com",
    description="Riddle Generation",
    packages=find_packages(),
    install_requires=["torch", "transformers"],
)
