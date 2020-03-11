from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import setuptools

_VERSION = '0.1'

REQUIRED_PACKAGES = [
]

DEPENDENCY_LINKS = [
]

setuptools.setup(
    name='Depthwise-Separable-Convolution_Pytorch',
    version=_VERSION,
    description="Unofficial PyTorch modification of Depthwise Separable Convolution",
    install_requires=REQUIRED_PACKAGES,
    dependency_links=DEPENDENCY_LINKS,
    url='https://github.com/seungjunlee96/Depthwise-Separable-Convolution_Pytorch',
    author='Seungjun Lee',
    license='MIT License',
    include_package_data=True,
    package_dir={},
    packages=setuptools.find_packages(exclude=['tests']),
)
