# PyNUFFT: Python non-uniform fast Fourier transform


A minimal "getting start" tutorial is available at http://jyhmiinlin.github.io/pynufft/ .

### Summary

PyNUFFT attempts to implement the min-max NUFFT of Fessler and Sutton, with the following features:

- Based on Python numerical libraries, such as Numpy, Scipy (matplotlib for displaying examples).
- Provides 1D/2D/3D examples for further developments.
- Support of PyCUDA and PyOpenCL. 
[//]: # NVIDIA's graphic processing units (GPUs) and opencl devices (Intel OpenCL SDK, Intel OpenCL NEO or multi-core CPU). 
- LGPLv3

[//]: # ### Acknowledgements

[//]: # PyNUFFT was funded by the Cambridge Commonwealth, European and International Trust (Cambridge, UK) and Ministry of Education, Taiwan. 

[//]: # Thanks to the NVIDIA Corporation with the donation of a Titan X Pascal and a Quadro P6000 GPU used for testing the GPU code. 

[//]: # Thanks to the authors of Michigan Image 
[//]: # Reconstruction Toolbox (MIRT) for releasing the original min-max interpolator code. 

[//]: # The interpolator is implemented using the Fessler and Sutton's min-max NUFFT algorithm:

[//]: # Fessler JA, Sutton BP. Nonuniform fast Fourier transforms using min-max interpolation. IEEE Trans Signal Process 2003;51(2):560-574.

[//]: # If you want you could cite my paper. 

[//]: # Lin, Jyh-Miin. "Python Non-Uniform Fast Fourier Transform (PyNUFFT): An Accelerated Non-Cartesian MRI Package on a Heterogeneous Platform (CPU/GPU)." Journal of Imaging 4.3 (2018): 51.



```
@article{lin2018python,
  title={Python Non-Uniform Fast {F}ourier Transform ({PyNUFFT}): An Accelerated Non-{C}artesian {MRI} Package on a Heterogeneous Platform ({CPU/GPU})},
  author={Lin, Jyh-Miin},
  journal={Journal of Imaging},
  volume={4},
  number={3},
  pages={51},
  year={2018},
  publisher={Multidisciplinary Digital Publishing Institute}
}
```

[//]: # ### Postive response

[//]: # 1. "Scientists used different scripting languages (e.g., MATLAB, Python) and numerical libraries (e.g., PyNUFFT, BART, SigPy) to reproduce the original paper, and even some of the original authors welcomed the opportunity and reproduced their own work...."--Stikov, Nikola, Joshua D. Trzasko, and Matt A. Bernstein. "Reproducibility and the future of MRI research." Magnetic resonance in medicine (2019). 

[//]: # 2. The group of Otto-von-Guericke-Universität Magdeburg is using PyNUFFT to train the reconstruction network. (https://www.researchgate.net/publication/335473585_A_deep_learning_approach_for_reconstruction_of_undersampled_Cartesian_and_Radial_data)

