from setuptools import setup
try:
    from pypandoc import convert
    read_md = lambda f: convert(f, 'rst')
except ImportError:
    print("warning: pypandoc module not found, could not convert Markdown to RST")
    read_md = lambda f: open(f, 'r').read()

#setup(
#    # name, version, ...
#    name=
#    long_description=read_md('README.md'),
#    install_requires=[]
#)


#import os
#long_description = 'A pythonic non-uniform FFT'
#if os.path.exists('README.txt'):
#    long_description = open('README.txt').read()

from setuptools import setup

setup(name='pynufft',
      version='0.3',
      description='A pythonic non-uniform FFT (pynufft)',
      author='Jyh-Miin Lin',
      long_description=read_md('README.md'),
      author_email='jyhmiinlin@gmail.com',
      install_requires = ['numpy', 'scipy', 'matplotlib'],
      license='MIT',
      packages=['pynufft'],
      package_dir={'pynufft':'pynufft'},
      package_data={'pynufft':['data/*.txt']},
      include_package_data=True,
      zip_safe=False)
	
