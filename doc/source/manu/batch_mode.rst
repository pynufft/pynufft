Batched NUFFT
=============

The batch mode NUFFT can compute multiple NUFFTs at the same time. 
This relies on the 'batch' keyword during the plan() stage. 
If the batch is a postive integer, multiple NUFFTs are enclosed 
in the instance.  
If the batch is not provided (batch='None'), it regresses to the non-batch mode.   
Note the the input data and the output data of different channels are usually different. 

In this batch mode, the size of the input array is Nd + (batch, )
and the size of the output array is (M, batch).
In the non-batch mode, the size of the input array is Nd, and the size of the output array is (M, ).

Warning: the maximum size of a single NUFFT instance is still restricted by the Max memory allocation. 
OpenCL allows a single buffer which is 1/4 of the total memory.   

The following example shows the batch mode

.. literalinclude::  ../../../example/batch_multicoil_NUFFT.py
   :start-after: # Begin of test_batch_NUFFT()
   :end-before: # end of test_batch_NUFFT()