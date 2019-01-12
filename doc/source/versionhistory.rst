Version history
===============

**v0.4.0.0**

- 0.4.0.0 is a beta version.

- Major updates for the NUFFT_hsa class, including memory reduction and split-radix. Multiple NUFFT_hsa() using cuda backend become possible, by pushing the context to the top of the stack when a method is called upon. 

- Tested in Windows 10 with PyCUDA 2018.1.1, nvidia-driver 417.35, CUDA 9.2, Visual Studio 2015 Community, and Anaconda Python 3.7 64-bit. PyOpenCL remains untested. 

- Add batch mode.  

  
 
**v0.3.3.12** 

- 0.3.3.12 is a bug-fixed version.

- Removal of the keyword async for compatibility reasons because Reikna has changed the keyword to async_.

**v0.3.3.8**
 
- Bugfix: mm = numpy.tile(mm, [numpy.prod(Jd).astype(int), 1])  to fix wrong type when numpy.prod(Jd) is not casted as int

- Bugfix: fix rcond=None error in Anaconda 3.6.5 and Numpy 1.13.1 (the recommended None in Numpy 1.14 is backward incompatible with 1.13)

- Bugfix:  indx1 = indx.copy() was replaced by indx1 = list(indx) for Python2 compatibility

**v0.3.3.7**

- Bugfix in 0.3.3.7 Toeplitz is removed from the NUFFT_cpu and NUFFT_gpu to avoid the MemoryError.

**v0.3.3.6**

- Bugfix: correct the error of import. Now import NUFFT_cpu, NUFFT_hsa at the top level.


**v0.3.3**

- Note: GPU support is superseded by the Heterogeneous System Architecture (HSA). 

- A variety of nonlinear solvers are provided, including conjugate gradient method (cg), L1 total-variation ordinary least square (L1TVOLS), and L1 total-variation least absolute deviation (L1TVLAD).

- The CPU version support other nonlinear solvers,  lsmr, lsqr, gmr, cg, bicgstab, bicg, cgs, gmres, lgmres , apart from cg, L1TVOLS and L1TVLAD.

- Support multi-dimensional transform and reconstruction (experimentally).

**v0.3.2.9**

- Experimental support of NVIDIA's graphic processing unit (GPU). 

- The experimental class gpuNUFFT requires pycuda, scikit-cuda, and python-cuda-cffi. scikit-cuda could be installed from standard command.

**v0.3.2.8**

- Tested under Linux and Windows Anaconda3
  
**v0.3**

- Updated setup.py

- Removal of pyfftw due to segfault under some Linux distributions

  
