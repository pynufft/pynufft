NUFFT_hsa
=========

**The NUFFT_hsa class**

 

Defining the equispaced to non-Cartesian transform as  operator :math:`A`, the
NUFFT_hsa class provides the following methods:

- forward() method computes the single-coil to single-coil, or multi-coil to
    multi-coil (batch mode) forward operation :math:`A`.

- adjoint() method computes the single-coil to single-coil, or multi-coil to
        multi-coil  (batch mode) adjoint operation  :math:`A^H`.

- selfadjoint() method computes the single-coil to single-coil, or multi-coil
        to multi-coil (batch mode) selfadjoint operation :math:`A^H A`.

- forward_one2many() method computes the single-coil to multi-coil forward
        operation :math:`A` in batch mode. The single-coil image is copied to
        multi-coil images before transform. If set_sense() is called first,
        multi-coil images will be implicitly multiplied by the coil
        sensitivities before transform. If set_sense() is never called,
        multi-coil images will not be changed by the coil sensitivities before
        transform.

- adjoint_many2one() method computes the multi-coil to single-coil adjoint
        operation  :math:`A^H` in batch mode.
        The final reduction will divide the summed image by the number of
        coils. If set_sense() is called first, multi-coil images will be
        implicitly multiplied by the conjugate of coil sensitivities before
        reduction. If set_sense() is never called, multi-coil images will not
        be changed by the coil sensitivities before reduction.

- selfadjoint_one2many2one () method computes the single-coil to single-coil
        selfadjoint operation :math:`A^H A` in batch mode.
        It connects forward_one2many() and adjoint_many2one() methods.
        If set_sense() is called first, coil sensitivities and the conjugate
        are used during forward_one2many() and adjoint_many2one().

- solve() method links many solvers in pynufft.linalg.solver_cpu,
          which is based on the solvers of scipy.sparse.linalg.cg,
          scipy.sparse.linalg.'lsqr', 'dc', 'bicg', 'bicgstab', 'cg',
          'gmres', 'lgmres'

**Attributes**


- NUFFT_hsa.ndims: the dimension

- NUFFT_hsa.ft_axes: the axes where the FFT takes place

- NUFFT_hsa.parallel_flag: 1 for parallel transform.
                           0 for single channel.
                           If 1, the additional axis is batch.

- NUFFT_hsa.batch: internal attribute saving the number of channels.
                   If parallel_flag is 0, the batch is 1.
                   Otherwise, the batch must be given explicitly during planning.

- NUFFT_hsa.Nd: Tuple, the dimensions of the image

- NUFFT_hsa.Kd: Tuple, the dimensions of oversampled k-space

**Acceleration on PyCUDA/PyOpenCL**

The NUFFT_hsa was designed to accelerate the NUFFT function 
on the multi-core CPU and GPU, using PyOpenCL and PyCUDA backends.
This was made possible by using Reikna meta-package. 

If multiple NUFFT_hsa objects are created with the PyCUDA backend, 
each call can be executed only after the  context has 'popped up'. This is 
achieved by the decorator function push_cuda_context():  
calling NUFFT_hsa methods will trigger the decorator and get the context popped up. 
However, PyOpenCL has no such restriction 
and the call will automatically bypass the decorator for the NUFFT_hsa 
with the PyOpenCL backend. 

Different objects can be constructed on different PyCUDA and PyOpenCL backends. 

**The life-cycle of the PyNUFFT_hsa object**


NUFFT_hsa employs the plan-execution two-stage model.
This can maximize the runtime speed, at the cost of the extra precomputation times and extra memory.

Instantiating an NUFFT_hsa instance also initiates the context and defines some instance attributes. 
The context is linked to the accelerator and the kernels are compiled on the chosen device.
Instance attributes will be replaced later when the method plan() is called.


Then the plan() method calls the helper.plan() function, 
which constructs the scaling factor and the interpolator.  
The interpolator is precomputed and stored as multiple 1D ELLpack (ELL) sparse matrices. 
Each ELL matrix preserves the sparse matrix as the data and column indices. 
Multi-dimensional interpolators are implemented as a concatenated multiple 1D ELL sparse matrices.
The actual data and column indices are inferred from the meshindex.
At the end of the plan() method, the offload() method transfers the 
precomputed arrays to the accelerator. 

The run-time computations reuse the saved scaling factors and 
interpolators.  

 
  