NUFFT_hsa
=========

**Explain the NUFFT_hsa class**

The NUFFT_hsa was designed for accelerating the NUFFT function 
on the multi-core CPU and GPU, using PyOpenCL and PyCUDA backends.
This was made possible by using Reikna meta-package. 

If multiple NUFFT_hsa objects are created with the PyCUDA backend, 
each call can only be executed after the  context is 'popped up'. This is 
achieved by the decorator function push_cuda_context(), and each 
call to a method of the NUFFT_hsa object will trigger the decorator before 
actually running the call. However, PyOpenCL has no such restriction 
and the call will automatically bypass the decorator for the NUFFT_hsa 
with the PyOpenCL backend. 

Mixing PyCUDA and PyOpenCL backends is possible. 

**The life-cycle of the PyNUFFT_hsa object**


NUFFT_hsa employs the plan-execution two-stage model.
This can be faster at the cost of the extra precomputation times and extra memory.

Instantiating an NUFFT_hsa instance also initiates the context and defines some instance attributes. 
The context is linked to the accelerator and the kernels are compiled on the selected context.
Instance attributes will be replaced later when plan() takes place.


Then the plan() method calls the helper.plan() function, 
which constructs the scaling factor and the interpolator.  
The interpolator is precomputed and stored as multiple 1D ELLpack (ELL) sparse matrices. 
Each ELL matrix preserves the sparse matrix as the data and column indices. 
Multi-dimensional interpolators are implemented as a concatenated multiple 1D ELL sparse matrices.
The actual data and column indeces are inferred from the meshindex.
At the end of the plan() method, the offload() method transfers the 
precomputed arrays to the accelerator. 

The run-time computations reuse the saved scaling factors and 
interpolators.  

  