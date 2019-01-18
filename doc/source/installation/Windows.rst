Special topic: Installation under Windows 10
============================================



PyNUFFT has been tested under the Windows 10 home edition. 

The successful installation experience may be useful, but the actual process can be different due to various software and hardware environments.

I used Nvidia-driver 417.35-notebook-win10-64bit-international-whql-rp, Anaconda3-2018.12-Windows-x86_64, PyCUDA 2018.1.1 from official pip, Microsoft Visual Studio 2015 Community, CUDA 9.2.148_win10 and cuda_9.2.148.1_windows (patch).  

The following general guidance may be working in your case but there is no warranty.  

**Pre-requisites**

- GPU, a clean Windows 10, Windows Visual Studio 2015 Community Version and CUDA 9.2.148 

This is the most complex step and please refer to the official documentation of Nvidia:

`https://docs.nvidia.com/cuda/archive/9.2/cuda-installation-guide-microsoft-windows/index.html`

- First, install the nvidia-driver 417.35-notebook-win10-64bit-international-whql-rp. 

- Second, install the Microsoft Visual Studio 2015 Community.

- Third, install CUDA-9.2. Do not install older driver, Geforce experience.

Now open command prompt `cmd`, type ::

   nvcc -V

You may see ::

   nvcc: NVIDIA (R) Cuda compiler driver
   Copyright (c) 2005-2018 NVIDIA Corporation
   Built on Tue_Jun_12_23:09_12_Central_Daylight_time_2018
   Cuda compilation tool, release 9.2, V9.2.148    
   
which indicates that nvcc can be found in the system. 


- Add the environmental variable path.

In case the system cannot find `cl.exe` when you type `cl`: ::

   C:\Users\User>cl
   `cl` is not recognized as an internal or external command, 
   operable program or batch file.
   
This error is due to Visual Studio is not added to the system path, so the system cannot find cl. 
 
Try to follow the webpage at `https://docs.microsoft.com/en-us/previous-versions/office/developer/sharepoint-2010/ee537574(v%3Doffice.14)`.
Add "C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\bin" and "C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\Common7\\IDE".
 
Once adding Visual Studio to the system, open windows cmd, it should find cl.exe ::
   
   C:\Users\User>cl
   Microsoft (R) C/C++ Optimizing Compiler Version 19.0024215.1 for x86
   Copyright (C) Microsoft Corpoortion. All rights reserved.
   
   usage: cl [ option... ] filename... [ /link linkoption... ]

without the earlier error message. If the error persists, the path must be modified again. 

**Installation of Anaconda3**

-  Now install Anaconda3. I downloaded Anaconda3-2018.12-Windows-x86_64. Once done you can follow the general installation procedure as mentioned in the earlier sections. 

**Installation of Pytools, PyCUDA, and PyNUFFT**

- Open Anaconda3 command prompt, type::

   set PYTHONIOENCODING=UTF-8
   pip install pytools
   pip install pycuda
   pip install reikna
   pip install pynufft
   
- Test PyNUFFT::

   python
   from pynufft import tests
   tests.test_init()
   
**Installation of Pyopencl**   




