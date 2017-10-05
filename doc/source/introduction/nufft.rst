Non-uniform fast Fourier transform (NUFFT)
==========================================

**FFT is limited to equispaced frequency locations**

Fast Fourier transform (FFT) has become one of the most important algorithms in signal processing. FFT delivers a fast and exact discrete Fourier transform (DFT) in a much shorter computation time than direct DFT does. 

However, FFT does not allow for efficient non-Cartesian DFT, and non-uniform fast Fourier transform (NUFFT) attempts to accelerate non-Cartesian DFT.

..
   DFT for off-grid points requires a complexity of :math:`O(MN)` 
   
   On the contrary, DFT allows for any frequency locations. 
    
   Thus, the frequency componets at off-grid locations must resort to the costy DFT. 
   
   Given a length :math:`N` time series, the complexity of DFT for :math:`M` frequency locations is :math:`O(MN)`.
   
   DFT could become very slow when :math:`M >> N`.   
   
   **Non-uniform fast Fourier transform (NUFFT)**
   
   NUFFT attempts to efficiently compute the amplitude at :math:`M` non-uniform frequency locations.  
   
   The basic idea of NUFFT is to combine FFT and interpolation.
   
   :math:`O(NlogN + MJ)` 


**Flow diagram of pynufft**

The flow diagram of pynufft can be found in :numref:`flow-diagram`, which includes the following three steps:

1. Scaling to extend the range of interpolation.

2. Oversampled FFT.

3. Interpolation (gridding). 


.. _flow-diagram:

.. figure:: ../figure/flow_diagram.png
   :width: 30%

   Flow diagram of pynufft

**Original NUFFT paper**
   
More information can be found in the original NUFFT paper:

*Fessler JA, Sutton BP. Nonuniform fast Fourier transforms using min-max interpolation. IEEE Trans Signal Process 2003;51(2):560-574.*

which details the min-max interpolator for NUFFT. 