from setuptools import setup

setup(
    name='hungarianalg',
    url='https://github.com/jbrightuniverse/hungarianalg',
    author='James Yuming Yu',
    packages=['hungarianalg'],
    install_requires=['numpy'],
    version='0.0.1',
    license='MIT',
    description='Implementation of the Hungarian Algorithm for optimal matching in bipartite weighted graphs.',
    long_description=open('README.md').read()
)
