# Copyright 2023 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(name='trustmark',
    version='0.5.4',
    python_requires='>=3.8.5',
    description='High fidelty image watermarking at arbitrary resolution',
    url='https://github.com/adobe/trustmark',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Tu Bui, Shruti Agarwal, John Collomosse',
    author_email='collomos@adobe.com',
    license='MIT License',
    packages=['trustmark','trustmark.KBNet'],
    package_data={'trustmark': ['**/*.yaml','**/*.ckpt','**/*.md']},
    include_package_data = True,
    install_requires=['omegaconf>=2.1',
                      'pathlib>=1.0.1',
                      'numpy>=1.20.0',
                      'torch>=2.1.2',
                      'torchvision>=0.16.2',
                      'lightning>=2.0',
                      'six>=1.9',
                      'einops>=0.4.0',
                      'kornia>=0.7.0'
                      ],

    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'License :: OSI Approved :: MIT License',  
        'Programming Language :: Python :: 3',],)
