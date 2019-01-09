Installation
============

-------------------
System requirements
-------------------

**CPU**

Each PyNUFFT instance is designed to be executed on a single node. 

We recommend one or more modern x86_64 processors on a single node. Successful stories include Intel® Core™ i7-6700HQ Processor, 
Intel® Xeon® W-2125, Intel® Core™ i9-7900X. 
Multiple CPUs on a single node have been used for testing.

Distributed computing on multiple nodes is unsupported. 
 

**Memory**

A general instruction is that the memory should be sufficient for computing a single NUFFT object, which is dependent on the type of problem. 
  
A single 2D problem of 256 × 256 matrix can be computed on a system with 8GB memory. 
 
For 3D NUFFT, it is not uncommon that a single NUFFT_cpu object can use more than 200GB memory.

**GPU**

To use GPU, a recent NVIDIA's GPU (after Maxwell, Pascal) with the recent drivers should be working properly.

The newest nvidia-driver versions of 415.18 is recommended. 
Earlier versions may be working but please be informed that Nvidia may discontinue the support for outdated drivers. 

A general rule is that the memory on the GPU has to be sufficient for computing the NUFFT problem.
Successful storeis include NVIDIA Geforce GTX 965m, NVIDIA Titan V100 (Amazon Web Services), 
NVIDIA Geforce GTX 1060 6GB, NVIDIA Titan X Pascal, NVIDIA Quadro P6000, NVIDIA Quadro GP100. 

**Operating System**

Ubunut 16.04 - 18.04 have been tested. 

Windows 10 has been tested.  

--------
Software
--------

**Python**
To run the NUFFT_cpu, the basic CPython, Numpy and Scipy packages must be available on the system.
IronPython might be functioning but not guaranteed.
   
Users must be familiar with Python and its pip packaging system.    
Optionally, users can clone the github repository and build the package from the local folder. 

**Compiler**

NUFFT_cpu class does not require compiler. Only Python/Numpy/Scipy is needed. 

**GPU(CUDA and OpenCL) or multi-core CPU (OpenCL)**

However, NUFFT_hsa relies on the just-in-time (JIT) compilation so the CUDA/OpenCL must be able to compile the code. 

To accelerate the code on the graphic processing unit (GPU), Reikna, PyCUDA, PyOpencl must be avaialbe. 
Users need to know the basics of GPU computing.  

The CUDA of Nvidia has been tested. 

PyNUFFT has been working with up to OpenCL 1.2, which is well supproted by Nvidia and Intel.

Intel Studio 2019 supports Ubuntu 16.04, 18.04 and Windows. 
 
We have not tested OpenCL 2.0 but the PyNUFFT should be working with OpenCL properly.  

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



.. toctree::
   :maxdepth: 2
   
   installation/Linux
   installation/Windows
