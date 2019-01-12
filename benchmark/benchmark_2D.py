
import numpy 
import matplotlib.pyplot as pyplot
import scipy.misc
import scipy.io
from matplotlib import cm
gray = cm.gray

import pkg_resources
DATA_PATH = pkg_resources.resource_filename('pynufft', './src/data/')   
image = scipy.misc.ascent()



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
        
 
Nd = (256,256) # time grid, tuple
image = scipy.misc.imresize(image, Nd)*(1.0 + 0.0j)
Kd = (512,512) # frequency grid, tuple
Jd = (6,6) # interpolator 
#     om=       numpy.load(DATA_PATH+'om3D.npz')['arr_0']
# om = numpy.random.randn(10000,3)*2
# om = numpy.load('/home/sram/Cambridge_2012/DATA_MATLAB/Ciuciu/Trajectories_and_data_sparkling_radial/radial/')['arr_0']
om = scipy.io.loadmat('/home/sram/Cambridge_2012/DATA_MATLAB/Ciuciu/Trajectories_and_data_sparkling_radial/sparkling/samples_sparkling_x8_64x3072.mat')['samples_sparkling']
# om = scipy.io.loadmat('/home/sram/Cambridge_2012/DATA_MATLAB/Ciuciu/Trajectories_and_data_sparkling_radial/radial/samples_radial_x8_64x3072.mat')['samples_radial']
om = om/numpy.max(om.real.ravel()) * numpy.pi

print(om.shape)
from pynufft import NUFFT_cpu, NUFFT_hsa, NUFFT_hsa_legacy
# from pynufft import NUFFT_memsave
NufftObj_cpu = NUFFT_cpu()
NufftObj_hsa = NUFFT_hsa()
NufftObj_memsave = NUFFT_hsa()

import time
t0=time.time()
NufftObj_cpu.plan(om, Nd, Kd, Jd)
t1 = time.time()
NufftObj_hsa.plan(om, Nd, Kd, Jd)
NufftObj_memsave.plan(om, Nd, Kd, Jd)
t2 = time.time()
# proc = 0 # cpu
proc = 1 # gpu
# NufftObj_hsa.offload(API = 'ocl',   platform_number = proc, device_number = 0)

# NufftObj_memsave.offload(API = 'ocl',   platform_number = proc, device_number = 0)
# NufftObj_memsave.offload(API = 'cuda',   platform_number = 0, device_number = 0)
t3 = time.time()
print('planning time of CPU = ', t1 - t0)
print('planning time of GPU = ', t2 - t1)
print('loading time of GPU = ', t3 - t2)
gx_hsa = NufftObj_hsa.thr.to_device(image.astype(numpy.complex64))
gx_memsave = NufftObj_memsave.thr.to_device(image.astype(numpy.complex64))

maxiter = 10 
tcpu_forward, tcpu_adjoint, ycpu, xcpu = benchmark(NufftObj_cpu, image, maxiter)
print(tcpu_forward, tcpu_adjoint)
maxiter = 100
thsa_forward, thsa_adjoint, yhsa, xhsa = benchmark(NufftObj_hsa, gx_hsa, maxiter)
print(thsa_forward, thsa_adjoint, numpy.linalg.norm(yhsa.get() - ycpu)/  numpy.linalg.norm( ycpu))
tmem_forward, tmem_adjoint, ymem, xmem = benchmark(NufftObj_memsave, gx_memsave, maxiter)
print(tmem_forward, tmem_adjoint)
print(tmem_forward, tmem_adjoint, numpy.linalg.norm(ymem.get() - ycpu)/ numpy.linalg.norm( ycpu))
