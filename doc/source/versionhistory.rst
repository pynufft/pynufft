Version history
===============

**v0.3.3**

- Note: GPU support is superseded by the Heterogeneous System Architecture (HSA). 

- A variety of nonlinear solvers are provided, including conjugate gradient method (cg), L1 total-variation ordinary least square (L1TVOLS), and L1 total-variation least absolute deviation (L1TVLAD).

- The CPU version support other nonlinear solvers,  lsmr, lsqr, gmr, cg, bicgstab, bicg, cgs, gmres, lgmres , apart from cg, L1TVOLS and L1TVLAD.

- Support multi-dimensional transform and reconstruction (experimentally).

**v0.3.2.9**

- Experimental support of NVIDIA's graphic processing unit (GPU). 

 The experimental class gpuNUFFT requires pycuda, scikit-cuda, and python-cuda-cffi. 

 scikit-cuda could be installed from standard command.

**v0.3.2.8**

- Tested under Linux and Windows Anaconda3
 
  add scipy.misc.face().
  
**v0.3**

- Update setup.py

 remove pyfftw due to segfault under some Linux distributions

  
