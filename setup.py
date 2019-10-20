from setuptools import setup, find_packages

setup(
    name='ada_in_style',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'tqdm',
        'pillow',
    ]
)
