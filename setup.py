import setuptools

with open('python_api/README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='machine_common_sense',
    version='0.0.1',
    scripts=[],
    author='Next Century, a wholly owned subsidiary of CACI',
    author_email='mcs-ta2@machinecommonsense.com',
    description='Machine Common Sense Python API to Unity 3D Simulation Environment',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/NextCenturyCorporation/MCS/',
    package_dir={'':'python_api'},
    packages=setuptools.find_packages('python_api'),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache License',
        'Operating System :: OS Independent',
    ]
)
