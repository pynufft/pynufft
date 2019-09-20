Multiple NUFFT instances
========================

Multiple NUFFT_cpu instances, NUFFT_hsa instances, 
or different types of instances can co-exist at the same time.
Each instance has its own memory. 
However,  multiple instances  may be penalized by the insufficient memory and the degraded efficiency.

Multiple instances can be planned after all the instances have been created.
Alternatively, each instance can be created and planned, followed by another instance.   

Note that multiple PyCUDA backends are possible since 2019.1.1. 
This is made possible by introducing the push_cuda_context() decorating 
function. Version earlier than 2019.1.1 do not support multiple PyCUDA backends.
This is because every call to a CUDA context will require the current 
context to be popped up to the top of the stack of the contexts.

PyOpenCL does not have the context pop-up issue. 
Still, only the newest version is recommended.  

**Multiple NUFFT_cpu instances**

Multiple instances can be planned after all the instances are created::

   # Create the first NUFFT_cpu
   NufftObj1 = NUFFT_cpu()
   # Create the second NUFFT_cpu
   NufftObj2 = NUFFT_cpu()
   # Plan the first instance
   NufftObj1.plan(om1, Nd, Kd, Jd)
   NufftObj2.plan(om2, Nd, Kd, Jd)
   
   
or each instance can be created and planned sequentially::

   # Create the first NUFFT_cpu
   NufftObj1 = NUFFT_cpu()
   NufftObj1.plan(om1, Nd, Kd, Jd)
   
   # Create the second NUFFT_cpu
   NufftObj2 = NUFFT_cpu()
   NufftObj2.plan(om2, Nd, Kd, Jd)  
   
   y1 = NufftObj1.forward(x)
   y2 = NufftObj2.forward(x)    

**Multiple NUFFT_hsa instances**

Like NUFFT_cpu, each instance can be created and planned sequentially:

::

   # Create the first NUFFT_hsa
   NufftObj1 = NUFFT_hsa('cuda')
   NufftObj1.plan(om1, Nd, Kd, Jd)
   
   # Create the second NUFFT_hsa
   NufftObj2 = NUFFT_hsa('cuda')
   NufftObj2.plan(om2, Nd, Kd, Jd)
   
   y1 = NufftObj1.forward(x)
   y2 = NufftObj2.forward(x)

Mixing cuda and opencl is possible.

::

   # Create the first NUFFT_hsa
   NufftObj1 = NUFFT_hsa('ocl')
   NufftObj1.plan(om1, Nd, Kd, Jd)
   
   # Create the second NUFFT_hsa
   NufftObj2 = NUFFT_hsa('cuda')
   NufftObj2.plan(om2, Nd, Kd, Jd)
   
   y1 = NufftObj1.forward(x)
   y2 = NufftObj2.forward(x)

Multiprocessing (experimental)
==============================


Multiprocessing is the built-in package for parallelism 
of C-Python. 
PyNUFFT may (experimentally) work together with the multiprocessing 
module of Python.  

Mutliprocessing module relies on pickle to serialize the data, whereas 
the PyCUDA and PyOpenCL contexts are "unpicklable". 
Thus, I found that multiprocessing for PyNUFFT must fulfill the following 3 conditions: (1)
each NUFFT_hsa instance is to be created then executed in a separate process, 
(2) any CUDA/PyOpenCL related object cannot be sent or planned in advance, and
(3)Using taskset to assign a process to a specified CPU core. 

It is user's responsibility to take care of the hardware (total memory and IO). 



An example of working with multiprocessing (mixed CUDA and OpenCL backends) is as follows.
In this example, an "atomic_NUFFT" class is created as a high-level wrapper for the creation and execution of NUFFT_hsa.
This is only tested using Linux because parallel computing is highly platform dependent.

.. literalinclude::  ../../../example/parallel_NUFFT_hsa.py
 
