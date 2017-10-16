# pynufft: Python non-uniform fast Fourier transform


FFT is the standard method that estimates the frequency components at equispaced locations.

NUFFT can calculate the frequency components at non-equispaced locations.

A minimal "getting start" tutorial is available at http://jyhmiinlin.github.io/pynufft/ .

### New in version 0.3.3

New in version 0.3.3

Note: GPU support is superseded by the Heterogeneous System Architecture (HSA).

A variety of nonlinear solvers are provided, including conjugate gradient method (cg), L1 total-variation ordinary least square (L1TVOLS), and L1 total-variation least absolute deviation (L1TVLAD).

The CPU version support other nonlinear solvers, lsmr, lsqr, gmr, cg, bicgstab, bicg, cgs, gmres, lgmres , apart from cg, L1TVOLS and L1TVLAD.

Support multi-dimensional transform and reconstruction (experimentally).

### Summary

Pynufft implements Fessler's min-max NUFFT, with the following features:

- Written in pure Python.
- Based on numerical libraries, such as Numpy, Scipy (matplotlib for displaying examples).
- Provides the python interface including forward transform, adjoint transform and other routines.
- Provides 1D/2D/3D examples for further developments.
- (Experimental) Supporting NUFFT on NVIDIA's graphic processing units (GPUs) and multi-core CPU platforms.

### Limitations

In Numpy, the default fft library is fftpack, so the speed of NUFFT transform may be optimized by FFTW.

Check out the tutorial (http://jyhmiinlin.github.io/pynufft/misc/fftw.html)

### Other nufft implementations in Python:

Python-nufft: Python bindings to Fortran nufft. (https://github.com/dfm/Python-nufft/), MIT license

pynfft: Python bindings around the NFFT C-library, which uses the speed of FFTW, (https://github.com/ghisvail/pyNFFT), GPL v3

nfft: Pure Python implementation of nfft (https://github.com/jakevdp/nfft).

nufftpy: Pure Python NUFFT of Python-nufft (https://github.com/jakevdp/nufftpy). 

mripy: A Python based MRI package (https://github.com/peng-cao/mripy). combines Numba and NUFFT.

### Acknowledgements

pynufft was funded by the Ministry of Science and Technology, Taiwan, Cambridge Commonwealth, European and International Trust, and Ministry of Education, Taiwan.  

If you find pynufft useful, please cite:

Jyh-Miin Lin and Hsiao-Wen Chung, Pynufft: python non-uniform fast Fourier transform for MRI, Building Bridges in Medical Sciences 2017, St John's College, CB2 1TP Cambridge, UK, 2017

And Fessler and Sutton's min-max NUFFT algorithm:
Fessler JA, Sutton BP. Nonuniform fast Fourier transforms using min-max interpolation. IEEE Trans Signal Process 2003;51(2):560-574.

Please open an issue if you have any question related to pynufft.

### Cite pynufft

@INPROCEEDINGS{lin2017bbmc_pynufft,
	author = {Jyh-Miin Lin and Hsiao-Wen Chung},
	booktitle = {Building Bridges in Medical Sciences 2017, St John's College, CB2 1TP Cambridge, UK},
	title = {{Pynufft: python non-uniform fast Fourier transform for MRI}},
	year = {2017}
}
