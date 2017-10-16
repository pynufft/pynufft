Installation
============

**Dependencies**



- Tested using Python 3.4 on Linux-64.  

- Numpy-1.13.3 and Scipy-0.19.1.

- Matplotlib is required for displaying images.

- Continuum's Anaconda_ environment should provide all the above packages. 

.. _Anaconda: https://www.continuum.io/downloads


**Quick Installation**

Install pynufft by using pip_ command::

   $ pip install pynufft

.. _pip: https://en.wikipedia.org/wiki/Pip_(package_manager)
    
**Installation from Git Repository**

git_ is a version control program, which allows you to clone the latest code base from pynufft_ repository::
   
   $ git clone https://github.com/jyhmiinlin/pynufft
   $ cd pynufft
   $ python setup.py install --user 

.. _git: https://en.wikipedia.org/wiki/Git
.. _pynufft: https://github.com/jyhmiinlin/pynufft

**Test if the installation is successful**

In Python environment, import pynufft::

    >>> import pynufft.tests as tests
    >>> tests.test_installation()
    
If the required data and functions are available, you will see all the required files exist::

   Does pynufft.py exist?  True
   Does om1D.npz exist? True
   Does om2D.npz exist? True
   Does om3D.npz exist? True
   Does phantom_3D_128_128_128.npz exist? True
   Does phantom_256_256.npz exist? True
   Does example_1D.py exist? True
   Does example_2D.py exist? True
   reikna  has been installed.
   pyopencl  has been installed.
   pycuda  has been installed.
    
**Uninstall pynufft**

Simply use "pip uninstall" to remove pynufft::

    $ pip uninstall pynufft




