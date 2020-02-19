from setuptools import setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='nexus',
    packages=['nexus'],
    include_package_data=True,
    install_requires=required,
)
