FFTW support (Compile Numpy with FFTW in Linux)
===============================================

FFTW is one of the fastest FFT library, which is licensed under GPL, which is imcompatible with the license used in Numpy. 

However, FFTW is unofficially supported. The trick is to compile Numpy from pip, with a configuration file telling Numpy to use FFTW. 

Create a Numpy configuration file in user's home directory::

    $ echo "[fftw]" >  ~/.numpy-site.cfg
    $ echo ""libraries = fftw3 >>  ~/.numpy-site.cfg

Then  install Numpy from pip::

   $ pip uninstall numpy
   $ pip install numpy

