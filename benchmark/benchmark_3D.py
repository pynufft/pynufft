special_license='''
The license of the 3D Shepp-Logan phantom:
Copyright (c) 2006, Matthias Schabel 
All rights reserved.

Redistribution and use in source and binary forms, with or without 
modification, are permitted provided that the following conditions are 
met:

* Redistributions of source code must retain the above copyright 
notice, this list of conditions and the following disclaimer. 
* Redistributions in binary form must reproduce the above copyright 
notice, this list of conditions and the following disclaimer in 
the documentation and/or other materials provided with the distribution 
* Neither the name of the University of Utah Department of Radiology nor the names 
of its contributors may be used to endorse or promote products derived 
from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE 
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
POSSIBILITY OF SUCH DAMAGE.
'''
import numpy 
import matplotlib.pyplot as pyplot
from matplotlib import cm
gray = cm.gray

import pkg_resources
DATA_PATH = pkg_resources.resource_filename('pynufft', './src/data/')   
image = numpy.load(DATA_PATH +'phantom_3D_128_128_128.npz')['arr_0']#[0::2, 0::2, 0::2]
image = numpy.array(image, order='C')
print(special_license)

# pyplot.imshow(numpy.abs(image[:,:,64]), label='original signal',cmap=gray)
# pyplot.show()

def benchmark(nufftobj, gx, maxiter):
    import time
    t0= time.time()
    for pp in range(0, maxiter):
        gy = nufftobj.forward(gx)
    t1 = time.time()
    for pp in range(0, maxiter):
        gx2 = nufftobj.adjoint(gy)
    t2 = time.time()
    return (t1 - t0)/maxiter, (t2 - t1)/maxiter, gy, gx2
        
 
Nd = (128,128,128) # time grid, tuple
Kd = (256,256,256) # frequency grid, tuple
Jd = (6,6,6) # interpolator 
#     om=       numpy.load(DATA_PATH+'om3D.npz')['arr_0']
om = numpy.random.randn(1000000,3)*2
# om = numpy.load('/home/sram/UCL/DATA/G/3D_Angio/greg_3D.npz')['arr_0'][::1, :]
# for pp in range(0, 3):
#     om[:, pp] = numpy.sort(om[:, pp])
print(om.shape)
from pynufft import NUFFT_cpu, NUFFT_hsa, NUFFT_memsave
# from pynufft import NUFFT_memsave
NufftObj_cpu = NUFFT_cpu()
# NufftObj_hsa = NUFFT_hsa()
NufftObj_memsave = NUFFT_memsave()

import time
t0=time.time()
NufftObj_cpu.plan(om, Nd, Kd, Jd)
t1 = time.time()
# NufftObj_hsa.plan(om, Nd, Kd, Jd)
NufftObj_memsave.plan(om, Nd, Kd, Jd)
t2 = time.time()
# proc = 0 # cpu
proc = 1 # gpu
# NufftObj_hsa.offload(API = 'ocl',   platform_number = proc, device_number = 0)

NufftObj_memsave.offload(API = 'ocl',   platform_number = proc, device_number = 0)
# NufftObj_memsave.offload(API = 'cuda',   platform_number = 0, device_number = 0)
t3 = time.time()
print('planning time of CPU = ', t1 - t0)
print('planning time of GPU = ', t2 - t1)
print('loading time of GPU = ', t3 - t2)
# gx_hsa = NufftObj_hsa.thr.to_device(image.astype(numpy.complex64))
gx_memsave = NufftObj_memsave.thr.to_device(image.astype(numpy.complex64))

maxiter = 1
tcpu_forward, tcpu_adjoint, ycpu, xcpu = benchmark(NufftObj_cpu, image, maxiter)
print(tcpu_forward, tcpu_adjoint)
maxiter = 20
# thsa_forward, thsa_adjoint, yhsa, xhsa = benchmark(NufftObj_hsa, gx_hsa, maxiter)
# print(thsa_forward, thsa_adjoint, numpy.linalg.norm(yhsa.get() - ycpu)/  numpy.linalg.norm( ycpu))
tmem_forward, tmem_adjoint, ymem, xmem = benchmark(NufftObj_memsave, gx_memsave, maxiter)
# print(tmem_forward, tmem_adjoint)
print(tmem_forward, tmem_adjoint, numpy.linalg.norm(ymem.get() - ycpu)/ numpy.linalg.norm( ycpu))
