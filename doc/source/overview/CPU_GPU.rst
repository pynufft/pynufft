CPU and GPU (HSA)
=================

The PyNUFFT was originally running on Numpy/Scipy. Unfortunately the default Numpy/Scipy is most efficient on a single CPU core. 

Later it was ported to PyCUDA and PyOpenCL, which allows us to leverage the speed of multi-core CPU and GPU.   

Mixing NUFFT_cpu and NUFFT_hsa, or multiple NUFFT_cpu or multiple NUFFT_hsa is possible but no warranty. 

The class methods are listed in :numref:`dimension_table`

.. _dimension_table:
.. list-table:: Methods implemented in NUFFT_cpu and NUFFT_hsa
   :widths: 25 12 12 30
   :header-rows: 1

   * - method name
     - NUFFT_cpu
     - NUFFT_hsa
     - References
   * - __init__()
     - ✓
     - ✓
     - Constructor
   * - plan()
     - ✓
     - ✓
     - Planning the instance
   * - forward()
     - ✓ 
     - ✓
     - (Normal and batch mode) Forward NUFFT :math:`A`
   * - adjoint()
     - ✓
     - ✓
     - (Normal and batch mode) Adjoint NUFFT :math:`A^H`
   * - forward_one2many()
     - ✓                   
     - ✓
     - (batch mode) Single image -> multi-coil data forward NUFFT :math:`A`
   * - adjoint_many2one()
     - ✓                   
     - ✓
     - (batch mode) Multi-coil data -> single image adjoint NUFFT :math:`A^H`
   * - selfadjoint_one2many2one()
     - ✓                   
     - ✓
     - (batch mode)  Single image -> multi-coil data -> single image selfadjoint  :math:`A^H A`
   * - set_sense()
     - ✓                   
     - ✓
     - (batch mode) Set the coil sensitivities
   * - reset_sense()
     - ✓                   
     - ✓
     - (batch mode) reset the coil sensitivities to ones
   * - offload()
     - ×          
     - ✓
     - Offload the NUFFT_hsa() to device. 
   * - x2xx()
     - ✓          
     - ✓
     - (Normal and batch mode) Apply the scaling factor 
   * - xx2k()
     - ✓          
     - ✓
     - (Normal and batch mode) Oversampled FFT    
   * - k2y()
     - ✓          
     - ✓
     - (Normal and batch mode) Interpolation
   * - k2vec()
     - ✓          
     - ×   
     - (Normal and batch mode) Reshape the k-space to the vector       
   * - vec2y()
     - ✓          
     - ×   
     - (Normal and batch mode) Multiply the vector to generate the data          
   * - vec2k()
     - ✓          
     - ×   
     - (Normal and batch mode) Reshape the vector to k-space      
   * - y2vec()
     - ✓          
     - ×   
     -  (Normal and batch mode) Multiply the data to get the vector       
   * - y2k()
     - ✓          
     - ✓
     - (Normal and batch mode) Adjoint of k2y()
   * - k2xx()
     - ✓          
     - ✓
     - (Normal and batch mode) Inverse FFT (excessive parts are cropped)
   * - xx2x()
     - ✓          
     - ✓
     - (Normal and batch mode) Apply the scaling factor      
   * - _precompute
     - ✓          
     - ✓
     - (Normal and batch mode) Apply the scaling factor                   
