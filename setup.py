from setuptools import setup, find_packages

setup(
    name='guitartechs_dataset',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch',
        'librosa',
        'pretty_midi',
        'numpy',
        'pandas',
        'requests',
        'tqdm',
        'matplotlib'
    ],
    python_requires='>=3.7, <3.13',
)
