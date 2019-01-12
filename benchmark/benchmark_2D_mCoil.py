"""
1. Write a NUFFT_hsa benchmark using for-loop
2. Write a NUFFT_memsave benchmark using for-loop
3. Write a NUFFT_mCoil benchmark without for-loop


"""
import numpy 
import matplotlib.pyplot as pyplot
import scipy.misc
import scipy.io
from matplotlib import cm
import matplotlib
gray = cm.gray

import pkg_resources
DATA_PATH = pkg_resources.resource_filename('pynufft', './src/data/')   




# pyplot.imshow(numpy.abs(image[:,:,64]), label='original signal',cmap=gray)
# pyplot.show()

def benchmark(nufftobj, gx, maxiter, sense=1):
    import time
    t0= time.time()
    for pp in range(0, maxiter*sense):
        gy = nufftobj.forward(gx)
    t1 = time.time()
    for pp in range(0, maxiter*sense):
        gx2 = nufftobj.adjoint(gy)
    t2 = time.time()
    t_iter0 = time.time()
    for pp in range(0, maxiter*sense):
        pass
    t_iter1 = time.time()
    t_delta = t_iter1 - t_iter0 
    return (t1 - t0 )/maxiter, (t2 - t1)/maxiter, gy, gx2
        
def test_mCoil(sense_number):
    image = scipy.misc.ascent()
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
    print('om.shape, ', om.shape)
#     sense_number = 16
#     sense = numpy.ones(Nd + (sense_number,), dtype=numpy.complex64)
    m = om.shape[0]
    print(om.shape)
    from pynufft import NUFFT_cpu, NUFFT_hsa, NUFFT_hsa_legacy
        # from pynufft import NUFFT_memsave
    NufftObj_cpu = NUFFT_cpu()
    api = 'ocl'
    proc = 1
    NufftObj_hsa = NUFFT_hsa_legacy(api, proc, 0)
    NufftObj_radix1 = NUFFT_hsa(api, proc, 0)
    NufftObj_radix2 = NUFFT_hsa(api, proc, 0)
        
    import time
    t0=time.time()
    NufftObj_cpu.plan(om, Nd, Kd, Jd)
    t1 = time.time()
    NufftObj_hsa.plan(om, Nd, Kd, Jd)
    t12 = time.time()
    NufftObj_radix1.plan(om, Nd, Kd, Jd, batch = sense_number, radix = 1)
    t2 = time.time()
    NufftObj_radix2.plan(om, Nd, Kd, Jd, batch = sense_number, radix = 2)
    tc = time.time()
#     proc = 0 # GPU
#     proc = 1 # gpu
#     NufftObj_hsa.offload(API = 'ocl',   platform_number = proc, device_number = 0)
#     t22 = time.time()
#     NufftObj_radix1.offload(API = 'ocl',   platform_number = proc, device_number = 0)
    # NufftObj_radix1.offload(API = 'cuda',   platform_number = 0, device_number = 0)
#     t3 = time.time()
#     NufftObj_radix2.offload(API = 'ocl',   platform_number = proc, device_number = 0)
#     tp = time.time()
    if proc is 0:
        print('CPU')
    else:
        print('GPU')
    print('Number of samples = ', om.shape[0])
    print('planning time of CPU = ', t1 - t0)
    print('planning time of HSA = ', t12 - t1)
    print('planning time of MEM = ', t2 - t12)
    print('planning time of mCoil = ', tc - t2)
    
    
#     print('loading time of HSA = ', t22 - tc)
#     print('loading time of MEM = ', t3 - t22)
#     print('loading time of mCoil = ', tp - t3)
        
    gx_hsa = NufftObj_hsa.thr.to_device(image.astype(numpy.complex64))
    gx_memsave0 = NufftObj_radix1.thr.to_device(image.astype(numpy.complex64))
    gx_memsave = NufftObj_radix1.s2x(gx_memsave0)
    gx_mCoil0 = NufftObj_radix2.thr.to_device(image.astype(numpy.complex64))    
    gx_mCoil = NufftObj_radix2.s2x(gx_mCoil0)
    maxiter = 2
    tcpu_forward, tcpu_adjoint, ycpu, xcpu = benchmark(NufftObj_cpu, image, maxiter, sense_number)
    print('CPU', int(m), tcpu_forward, tcpu_adjoint)
    maxiter = 5
    thsa_forward, thsa_adjoint, yhsa, xhsa = benchmark(NufftObj_hsa, gx_hsa, maxiter, sense_number)
    print('CSR', int(m), thsa_forward, thsa_adjoint, )#numpy.linalg.norm(yhsa.get() - ycpu)/  numpy.linalg.norm( ycpu))
    tmem_forward, tmem_adjoint, ymem, xmem = benchmark(NufftObj_radix1, gx_memsave, maxiter)#, sense_number)
    print('radix-1' , int(m), tmem_forward, tmem_adjoint)
    
    tmCoil_forward, tmCoil_adjoint, ymCoil, xmCoil = benchmark(NufftObj_radix2, gx_mCoil, maxiter)
    print('radix-2' , int(m), tmCoil_forward, tmCoil_adjoint)    
    
    for ss in range(0, sense_number):
        erry = numpy.linalg.norm(ymCoil.get()[:,ss] - ycpu)/ numpy.linalg.norm( ycpu)
        errx = numpy.linalg.norm(xmCoil.get()[...,ss] - xcpu)/ numpy.linalg.norm( xcpu)
        if erry > 1e-6 or errx > 1e-6:
            print("degraded accuracy:", sense_number, erry, errx)
        else:
            print("Pass test for coil: ", ss)
    NufftObj_radix1.release()
    NufftObj_radix2.release()
    NufftObj_hsa.release()
    del NufftObj_radix1, NufftObj_hsa, NufftObj_radix2, NufftObj_cpu
    return tcpu_forward, tcpu_adjoint,  thsa_forward, thsa_adjoint, tmem_forward, tmem_adjoint,  tmCoil_forward, tmCoil_adjoint

import numpy
CPU_forward = ()
HSA_forward = ()
MEM_forward = ()
mCoil_forward = ()
CPU_adjoint = ()
HSA_adjoint = ()
MEM_adjoint = ()
mCoil_adjoint = ()
SENSE_NUM = ()

for sense_number in (128, 64, 32, 16, 8, 4, 2, 1):
    print('SENSE = ', sense_number)
    t = test_mCoil(sense_number)
    CPU_forward += (t[0], )
    CPU_adjoint += (t[1], )
    HSA_forward  += (t[2], )
    HSA_adjoint  += (t[3], )
    MEM_forward  += (t[4], )
    MEM_adjoint  += (t[5], )
    mCoil_forward += (t[6], )
    mCoil_adjoint  += (t[7], )
    SENSE_NUM += (sense_number, )

CPU_forward = numpy.array(CPU_forward)
HSA_forward = numpy.array(HSA_forward)
MEM_forward = numpy.array(MEM_forward)
mCoil_forward = numpy.array(mCoil_forward)
CPU_adjoint = numpy.array(CPU_adjoint)
HSA_adjoint = numpy.array(HSA_adjoint)
MEM_adjoint = numpy.array(MEM_adjoint)
mCoil_adjoint = numpy.array(mCoil_adjoint)
SENSE_NUM = numpy.array(SENSE_NUM)


matplotlib.pyplot.subplot(1,3, 1)

matplotlib.pyplot.plot(SENSE_NUM, CPU_forward/HSA_forward, '*-', label='loop, CSR')
matplotlib.pyplot.plot(SENSE_NUM, CPU_forward/MEM_forward, 'D--', label='batched, radix-1')
matplotlib.pyplot.plot(SENSE_NUM, CPU_forward/mCoil_forward, 'x:', label='batched, radix-2')
matplotlib.pyplot.legend()
matplotlib.pyplot.ylabel('Acceleration')
matplotlib.pyplot.xlabel('Number of coils')
matplotlib.pyplot.title('Forward')

matplotlib.pyplot.subplot(1,3, 2)

matplotlib.pyplot.plot(SENSE_NUM, CPU_adjoint/HSA_adjoint, '*-', label='loop, CSR')
matplotlib.pyplot.plot(SENSE_NUM, CPU_adjoint/MEM_adjoint, 'D--', label='batched, radix-1')
matplotlib.pyplot.plot(SENSE_NUM, CPU_adjoint/mCoil_adjoint, 'x:', label='batched, radix-2')
matplotlib.pyplot.legend()
matplotlib.pyplot.ylabel('Acceleration')
matplotlib.pyplot.xlabel('Number of coils')
matplotlib.pyplot.title('Adjoint')

matplotlib.pyplot.subplot(1,3, 3)

matplotlib.pyplot.plot(SENSE_NUM, (CPU_adjoint+CPU_forward)/(HSA_adjoint + HSA_forward), '*-', label='loop, CSR')
matplotlib.pyplot.plot(SENSE_NUM, (CPU_adjoint+CPU_forward)/(MEM_adjoint + MEM_forward), 'D--', label='batched, radix-1')
matplotlib.pyplot.plot(SENSE_NUM, (CPU_adjoint+CPU_forward)/(mCoil_adjoint + mCoil_forward), 'x:', label='batched, radix-2')
matplotlib.pyplot.legend()
matplotlib.pyplot.ylabel('Acceleration')
matplotlib.pyplot.xlabel('Number of coils')
matplotlib.pyplot.title('Selfadjoint')
matplotlib.pyplot.show()


#     matplotlib.pyplot.imshow(xmCoil.get()[:,:].real)
#     matplotlib.pyplot.show()
#     matplotlib.pyplot.imshow(xcpu.real)
#     matplotlib.pyplot.show()
# print(tmem_forward, tmem_adjoint, numpy.linalg.norm(ymem.get() - ycpu)/ numpy.linalg.norm( ycpu))
