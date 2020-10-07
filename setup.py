
from setuptools import setup

import sys
#if sys.version_info[0] == 2:
#    sys.exit("Sorry, Python 2 is not supported yet")
setup(name='pynufft',
      version='2020.2.2',
      description='Python non-uniform fast Fourier transform (PyNUFFT)',
      author='PyNUFFT',
      author_email='pynufft@gmail.com',
      url = 'https://github.com/jyhmiinlin/pynufft', # use the URL to the github repo
      install_requires = ['numpy', 'scipy'],
      license='LGPLv3, AGPL',
      packages=['pynufft'],
      package_dir={'pynufft':'.'},
      package_data={'pynufft':['nufft','src','tests','linalg','nufft/*','src/*','src/*/*','tests/*','example/*', 'linalg/*']},
      include_package_data=True,
      zip_safe=False)
	
