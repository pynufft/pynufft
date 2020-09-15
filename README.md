![](g5738.jpeg)
# PyNUFFT: Python non-uniform fast Fourier transform

A minimal "getting start" tutorial is available at http://jyhmiinlin.github.io/pynufft/ .
 
### Summary

PyNUFFT is developed for fun and it attempts to implement the min-max NUFFT of Fessler and Sutton, with the following features:

- Based on Python numerical libraries, such as Numpy, Scipy (matplotlib for displaying examples).
- Multi-dimensional NUFFT.
- Support of PyCUDA and PyOpenCL. 
- LGPLv3

If you find PyNUFFT useful, please cite:

Lin, Jyh-Miin. “Python Non-Uniform Fast Fourier Transform (PyNUFFT): An Accelerated Non-Cartesian MRI Package on a Heterogeneous Platform (CPU/GPU).” Journal of Imaging 4.3 (2018): 51. (Available at https://www.mdpi.com/2313-433X/4/3/51)

and/or

J.-M. Lin and H.-W. Chung, Pynufft: python non-uniform fast Fourier transform for MRI Building Bridges in Medical Sciences 2017, St John’s College, CB2 1TP Cambridge, UK

### Acknowledgements

Special thanks to the authors of MIRT, gpuNUFFT and BART, which have largely inspired the development of this package. 

The project also thanks contributors for providing testing results and patches. 

<!--

### Related projects

The PyNUFFT package has currently been used by signal processing experts, astronomers, and physicists for building their applications. 

1. Online PySAP-MRI reconstruction (https://github.com/CEA-COSMIC/pysap-mri, which enjoys the speed of PyNUFFT.) 
2. Accelerated tomography
3. Radiation distribution 
4. Machine learning based MRI reconstruction (https://www.researchgate.net/publication/335473585_A_deep_learning_approach_for_reconstruction_of_undersampled_Cartesian_and_Radial_data)
5. Spiral off-resonance correction
6. For motion estimation (NUFFT adjoint + SPyNET) (https://pubmed.ncbi.nlm.nih.gov/32408295/)
7. PyNUFFT was used in ISMRM reproducible study group and was mentioned in "Stikov, Nikola, Joshua D. Trzasko, and Matt A. Bernstein. "Reproducibility and the future of MRI research." Magnetic resonance in medicine 82.6 (2019): 1981-1983."

Open-source Python software is nice for delivering your products. So try PyNUFFT today!


-->
