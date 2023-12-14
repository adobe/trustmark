# Copyright 2023 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(name='trustmark',
    version='0.1.0',
    description='High fidelty image watermarking at arbitrary resolution',
    url='https://git.corp.adobe.com/adobe-research/trustmark',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='John Collomosse',
    author_email='collomos@adobe.com',
    license='MIT License',
    packages=['trustmark','trustmark.utils','trustmark.KBNet','trustmark.ldm'],
    package_data={'trustmark': ['**/*.yaml','**/*.ckpt']},
    include_package_data = True,
    install_requires=['omegaconf>=2.3',
                      'pathlib>=1.0.1',
                      'numpy>=1.26',
                      'torch',
                      'torchvision',
                      'pytorch_lightning>=1.8',
                      'six>=1.12',
                      'einops>=0.6.1',
                      'kornia>=0.7.0',
                      'bchlib==0.14.0',
                      ],

    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'License :: OSI Approved :: MIT License',  
        'Programming Language :: Python :: 3',],)
