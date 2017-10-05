What's new
==========

**New in version 0.3.2.9**

- Experimental support of NVIDIA's graphic processing unit (GPU). 

The experimental class gpuNUFFT requires pycuda, scikit-cuda, and python-cuda-cffi. 

scikit-cuda could be installed from standard command::

    pip install scikit-cuda

python-cuda-cffi requires source and CUDA 8.0::

    git clone https://github.com/grlee77/python-cuda-cffi.git
    cd python-cuda-cffi
    python3 setup.py install 

gpuNUFFT class has been tested on Linux but hasn't been tested on Windows.   

The results of 1D and 2D are identical to results of CPU pynufft. However, the 3D pynufft is yet to be tested. 

**Install the latest pynufft-0.3.2.9**

Uninstall previous versions and install the latest v0.3.2.9::

    pip uninstall pynufft
    pip install pynufft
    
Import the gpu class::

    import pynufft.pynufft_gpu as pnft
    NufftObj=pnft.gpuNUFFT()
    NufftObj(om, Nd, Kd, Jd) 
    
Use the plan() as the same way as CPU pynufft. Forward and adjoint NUFFT work on Pycuda gpuarray.::

    import pycuda.gpuarray
    gpu_image=pycuda.gpuarray.to_gpu(image)

A tutorial document for gpuNUFFT will be provided soon.    
  
