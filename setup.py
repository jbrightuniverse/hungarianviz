from setuptools import setup

setup(
    name='hungarianviz',
    url='https://github.com/jbrightuniverse/hungarianviz',
    author='James Yuming Yu',
    packages=['hungarianviz'],
    install_requires=['numpy', 'Pillow'],
    version='0.0.1',
    license='MIT',
    description='Implementation of the Hungarian Algorithm for optimal matching in bipartite weighted graphs.',
    long_description=open('README.md').read()
)
