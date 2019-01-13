Installation
============

-------------------
System requirements
-------------------

**CPU**

Each PyNUFFT instance is designed to be executed on a single node. PyNUFFT is not designed for distributed computing on multiple nodes, but user may install PyNUFFT on multiple nodes and control them through network.  

Multple NUFFT_cpu instances on a single node provided that the total memory is sufficient to hold all of them. 

We recommend one or more modern x86_64 processors on a single node. Successful stories include Intel® Core™ i7-6700HQ Processor, 
Intel® Xeon® W-2125, Intel® Core™ i9-7900X. 



**Memory**

A general instruction is that the memory should be sufficient for computing a single NUFFT object, which is dependent on the type of problem. 
  
A single 2D problem of 256 × 256 matrix can be computed on a system with 8GB memory. 
 
For 3D NUFFT, it is not uncommon that a single NUFFT_cpu object can use more than 200GB memory.

**GPU**

Each PyNUFFT instance is initiated on a single GPU. An instance cannot be distributed across multiple GPUs. (PyNUFFT is not using cuFFT.)

However, multiple NUFFT_hsa instances may be initiated on a single GPU or multiple GPUs, but the performance may be impacted (limited by the memory, PCI-E bus or GPU cores).  

To use GPU, a recent NVIDIA's GPU (after Maxwell, Pascal) with the recent drivers should be working properly. 



The newest nvidia-driver versions of 415.18 is recommended. 
Earlier versions may be working but please be informed that Nvidia may discontinue the support for outdated drivers. 

A general rule is that the memory on the GPU has to be sufficient for computing the NUFFT problem.
Successful storeis include NVIDIA Geforce GTX 965m, NVIDIA Titan V100 (Amazon Web Services), 
NVIDIA Geforce GTX 1060 6GB, NVIDIA Titan X Pascal, NVIDIA Quadro P6000, NVIDIA Quadro GP100. 

**Operating System**

Ubunut 16.04 - 18.04 are recommended. 

Windows 10 has been tested but it requires Microsoft Studio 2015 community. Please refer to the following special topic about the installation under Windows 10. 

--------
Software
--------

**Python**
   
Users must be familiar with Python and its pip packaging system.  Python 2.7 and Python 3.6-3.7 are currently supported. 

To run the NUFFT_cpu, the basic CPython, Numpy and Scipy packages must be available on the system.
IronPython is compatible with CPython so ipython might be useful. 

PyNUFFT can be installed from pip system. Optionally, users can clone the github repository and build the package from the local folder. 

**Compiler**

NUFFT_cpu class does not require compiler. 

However, NUFFT_hsa requires JIT (just-in-time) compilation mechanism of Reikna/PyCUDA/PyOpenCL. The supporting compiler may be:

- gcc-7.3.0

- Microsoft (R) Visual Studio 2015 community edition. (Please refer to the following section: special topic: Installation under Windows 10). 

To accelerate the code on the graphic processing unit (GPU), Reikna, PyCUDA, PyOpencl must be avaialbe. Please refer the following special topic: Installation of OpenCL.

Users could know the concepts about GPU computing but practical CUDA programming skills are not strictly needed.  


--------------------
General Installation
--------------------

Continuum's Anaconda_ environment should provide all the above packages. 

.. _Anaconda: https://www.continuum.io/downloads

**Installation using pip**

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

--------------
Special topics
--------------

.. toctree::
   :maxdepth: 2
   
   Linux
   Windows
   OpenCL
