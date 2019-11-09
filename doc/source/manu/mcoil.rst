Multi-coil NUFFT
================

Coil sensitivity profiles can be incorporated into the
NUFFT.  The set_sense() instance method accepts the coil sensitivity profile.
Once the coil sensitivity profile is configured, the forward() method 
applies the coil sensitivity profile, while the  adjoint() method 
applies the conjugate of the coil sensitivity profile. 

The reset_sense() resets the coil sensitivities to one and it returns to batch NUFFT.
If the set_sense() is not called, the default coil sensitivities are all 
ones and it is a batch NUFFT.

Both the NUFFT_cpu and NUFFT_hsa instances support the multi-coil mode.

.. literalinclude::  ../../../example/batch_multicoil_NUFFT.py
   :start-after: # Begin of test_multicoil_NUFFT()
   :end-before: # end of test_multicoil_NUFFT()