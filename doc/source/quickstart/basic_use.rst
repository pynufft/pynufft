Basic use of PyNUFFT
====================

This section navigates you through the basic use of PyNUFFT. 

---------------------------
Initiating a PyNUFFT object
---------------------------

We can initiate a PyNUFFT by importing the NUFFT_cpu object::

   # import NUFFT_cpu class
   from pynufft import NUFFT_cpu
   
   # Initiate the NufftObj object
   NufftObj = NUFFT_cpu()


The NufftObj object has been created but it is still empty at this point.





Now we have to plan the NufftObj by calling the plan() method. 
The plan() method takes the input variables and plans for the object. 
Now we can plan for the NufftObj object given the non-Cartesian coordinates (om).
 

In the following code we have 100 random samples spreading across the 2D plane.  

 ::

   # generating 2D random coordinates
   import numpy
   om = numpy.random.randn(100, 2)

 
-------------------------
Plan for the NUFFT object
-------------------------

Now we call: ::

   NufftObj.plan(om, Nd, Kd, Jd, ft_axes=(0,1), batch=None)
   

See :py:class: `pynufft.NUFFT_cpu` and  :py:class: `pynufft.NUFFT_hsa`



-------------
Forward NUFFT
-------------
   
The forward NUFFT transforms the image into the non-Cartesian samples. ::

   y = NufftObj.forward(image)
   
If batch = None, the image.shape is equal to Nd. The returned y has a shape which is equal to (M, )

If batch exists, the image.shape is equal to Nd + (batch, ). The returned y has a shape which is equal to (M, batch) 
   
See :py:func:`pynufft.NUFFT_cpu.forward` :py:func: `pynufft.NUFFT_hsa.forward`

-------------
Adjoint NUFFT
-------------

The adjoint NUFFT transforms the non-Cartesian samples into the image ::

   x2 = NufftObj.adjoint(y)
   
If batch = None, y has a shape which is equal to (M, ). The returned image.shape is equal to Nd. 

If batch exists, y has a shape which is equal to (M, batch). The returned image.shape is equal to Nd + (batch, ). 
 
   





