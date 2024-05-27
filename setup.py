from setuptools import find_packages, setup

with open('README.md') as f:
    readme = f.read()

with open('LICENSE.txt') as f:
    license = f.read()

setup(
    name='elsim',
    version='0.1.0',
    description='Election simulation and analysis',
    long_description=readme,
    long_description_content_type="text/markdown",
    classifiers=[
        'Programming Language :: Python :: 3.6.2',
        'License :: OSI Approved :: MIT License',

        # Social choice theory
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Sociology',

        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6.2',  # f strings and underscore literals
    keywords='elections voting social choice theory',
    author_email='endolith@gmail.com',
    url='https://github.com/endolith/elsim',
    license=license,
    packages=find_packages(exclude=('tests', 'docs', 'examples')),
    install_requires=[
        'numpy',
        'scipy',
    ],
    tests_require=[
        'pytest',
        'hypothesis',
    ],
    extras_require={
        'fast':  ['numba<0.47'],  # reflected set will be removed in 0.47
    }
)
