#from setuptools import setup
#try:
#    from pypandoc import convert
#    read_md = lambda f: convert(f, 'rst')
#except ImportError:
#    print("warning: pypandoc module not found, could not convert Markdown to RST")
#    read_md = lambda f: open(f, 'r').read()



#import os
#long_description = 'A pythonic non-uniform FFT'
#if os.path.exists('README.txt'):
#    long_description = open('README.txt').read()

from setuptools import setup

import sys
if not sys.version_info[0] == 3:
    sys.exit("Sorry, Python 2 is not supported yet")
setup(name='pynufft',
      version='0.3.3.3',
      description='Python non-uniform fast Fourier transform (pynufft)',
      author='Jyh-Miin Lin',
      author_email='jyhmiinlin@gmail.com',
      url = 'https://github.com/jyhmiinlin/pynufft', # use the URL to the github repo
      install_requires = ['numpy', 'scipy', 'matplotlib'],
      license='MIT',
      packages=['pynufft'],
      package_dir={'pynufft':'.'},
      package_data={'pynufft':['src/*/*','tests/*','example/*']},
      include_package_data=True,
      zip_safe=False)
	
