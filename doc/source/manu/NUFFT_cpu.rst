NUFFT_cpu
=========

**The NUFFT_cpu class**

The NUFFT_cpu class encapsulates the NUFFT function using the Numpy/Scipy environment. 
This allows portability so the NUFFT_cpu() can be easily ported to Windows and Linux.
Users can install their favourite Python distribution. 
So far, I have tested Anaconda, intel-python, intel-numpy and open-source python.

However, the performance of NUFFT_cpu is therefore impacted by the Numpy implementation.  


Defining the equispaced to non-Cartesian transform as  operator :math:`A`, the
NUFFT_cpu class provides the following methods:

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
          scipy.sparse.linalg. 'lsqr', 'dc', 'bicg', 'bicgstab', 'cg',
          'gmres', 'lgmres'


**Attributes**

- NUFFT_cpu.ndims: the dimension

- NUFFT_cpu.ft_axes: the axes where the FFT takes place

- NUFFT_cpu.parallel_flag: 1 for parallel transform.
                           0 for single channel.
                           If 1, the additional axis is batch.

- NUFFT_cpu.batch: internal attribute saving the number of channels.
                   If parallel_flag is 0, the batch is 1.
                   Otherwise, the batch must be given explictly during planning.

- NUFFT_cpu.Nd: Tuple, the dimensions of the image

- NUFFT_cpu.Kd: Tuple, the dimensions of the oversampled k-space


**The life-cycle of an NUFFT_cpu object**


NUFFT_cpu employs the plan-execution two-stage model.
This can maximize the runtime speed, at the cost of the extra precomputation times and extra memory.


Instantiating an NUFFT_cpu instance defines only some instance attributes. Instance attributes will be replaced later once the method plan() is called.
  
Then the plan() method will call the helper.plan() function, 
which constructs the scaling factor and the interpolator.  
The interpolator is precomputed and stored in the Compressed Sparse Row (CSR) format. 
See `scipy.sparse.csr_matrix <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html>`_ and 
`CSR in Wikipedia <https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)>`_.   
  
Once the object has been planned, the forward() and adjoint() methods reuse the saved scaling factors and interpolators. 

