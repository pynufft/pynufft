import scipy.io
import matplotlib.pyplot
import numpy
import scipy.ndimage.filters
def fake_Cartesian(Nd):
    dd = len(Nd)
    Ndprod = numpy.prod(Nd)
    om = numpy.zeros((Ndprod, dd), dtype = numpy.float)
    vd = {}
    for dimid in range(0, dd):
        N = Nd[dimid]
        vd[dimid] = (numpy.arange(0, N)*2.0/N - 1.0/1.0)*numpy.pi
    if dd == 1:
        om[:,0] = vd[0]
    elif dd == 2:
        a, b = numpy.meshgrid(vd[0], vd[1], indexing='ij')
        om[:,0] = a.ravel(order='C')
        om[:,1] = b.ravel(order='C')
    elif dd == 3:
        a, b, c = numpy.meshgrid(vd[0], vd[1], vd[2])
        om[:,0] = a.ravel(order='C')
        om[:,1] = b.ravel(order='C')
        om[:,2] = c.ravel(order='C')
    elif dd == 4:
        a, b, c, d = numpy.meshgrid(vd[0], vd[1], vd[2], vd[3])
        om[:,0] = a.ravel(order='C')
        om[:,1] = b.ravel(order='C')
        om[:,2] = c.ravel(order='C')
        om[:,3] = d.ravel(order='C')
#     print(a,b)
    return om
    
    
def Nd_smooth(input_H, sigma, coil_axis):
    """
    Note: coil_axis is not smoothed
    Assume isotropic smoothing
    """
    ncoils = input_H.shape[coil_axis]
    image_G = input_H.copy()
    for jj in range(0, ncoils):
        image_G[..., jj ].real = scipy.ndimage.filters.gaussian_filter(image_G[...,jj].real, sigma = sigma)
        image_G[..., jj ].imag = scipy.ndimage.filters.gaussian_filter(image_G[...,jj].imag, sigma = sigma)

    return image_G
    
def Nd_sense(image_stack, maxiter = 20, sigma = 10):
#     image_stack += 1e-2
#     image_stack = Nd_smooth(image_stack, 10, -1)
    ncoils = image_stack.shape[-1]
   
    axis =len( image_stack.shape) 
    tmp_shape = image_stack.shape[0:axis -1 ] + (1, )
    H0 = numpy.ones_like(image_stack) #+ 1.0j*numpy.ones_like(image_stack)
#     H0 = image_stack
#     H0 = numpy.random.randn(H0.shape[0], H0.shape[1], H0.shape[2], H0.shape[3])
    
    H = H0#/ncoils
    
    for itr in range(0, maxiter):
        print(itr)
        H2 = numpy.conj(image_stack) * H  
        H2 = numpy.mean(H2,  axis = axis - 1)
        H2 = numpy.reshape(H2, tmp_shape )
        H = H2*(image_stack) 
        
        new_norm = H*numpy.conj(H)
        new_norm = numpy.sum(new_norm,  axis = axis - 1) #**0.7      
        new_norm = numpy.reshape(new_norm, tmp_shape )
        H = H/new_norm
        
        new_norm = numpy.linalg.norm(H)
        H = H/new_norm
#         H = Nd_smooth(H, 0.1, -1)



#     rms = numpy.mean(image_stack*numpy.conj(image_stack) ,  axis = axis - 1)
#     rms = numpy.reshape(rms**0.5, tmp_shape )
#     H = H*rms
    
        mu0 = numpy.conj( image_stack) * H
        mu0 = numpy.sum(mu0,  axis = axis - 1)
        
        mu = numpy.reshape(mu0, tmp_shape )
        mu = mu*image_stack
        mu = mu*numpy.conj(H)
        rayleigh = numpy.linalg.norm(mu/H*numpy.conj(H))
        mu = numpy.mean(mu,  axis = axis - 1)
        
        
#         print('Rayleigh Quotient', rayleigh)
        
    
    H = H/numpy.mean(numpy.abs(H.ravel()))
    H = Nd_smooth(H, sigma, -1)
#     print(numpy.shape(H), type(H))
    return H, mu0, mu

matplotlib.pyplot.gray()
Nc = 8
filename = '/home/sram/Cambridge_2012/WORD_PPTS/multicoil_NUFFT/simulation/pMRI/coils_12.mat'
K0 = scipy.io.loadmat(filename)['K0'][:,:,0:Nc]
Kn = scipy.io.loadmat(filename)['Kn'][:,:,0:Nc]
M0 = scipy.io.loadmat(filename)['M0']
Nd = M0.shape
multi_image = numpy.fft.ifftn(numpy.fft.fftshift(K0, axes = (0,1)), axes = (0,1))
multi_image_noisy = numpy.fft.ifftn(numpy.fft.fftshift(Kn, axes = (0,1)), axes = (0,1))
H, mu0, mu = Nd_sense(multi_image_noisy, maxiter = 2, sigma = 20)

om2= fake_Cartesian(Nd)

from pynufft import NUFFT_cpu, NUFFT_excalibur

NufftObj = NUFFT_cpu()
NufftObj_coil = NUFFT_excalibur()

NufftObj.plan(om2, Nd, (512,512), (6,6))

import multicoil_solver
fake_coil =  multicoil_solver.create_fake_coils(Nd[0], Nc)

H = numpy.ones(Nd + (Nc,), dtype = numpy.complex)
for pp in range(0, Nc):
    H[...,pp] = fake_coil[pp]

# H = H*numpy.reshape(M0, Nd+(1,))*(1.0+0.0j)/256

NufftObj_coil.plan1(om2, Nd, (512,512), (6,6), ft_axes = (0,1), image_stack = H)

y = NufftObj.forward(M0)



y2 = numpy.fft.fftshift(numpy.fft.ifft2(numpy.fft.fftshift(y.reshape(Nd,order='C'))))
y3 = NufftObj_coil.forward((M0)*(1.0 + 0.0j))
# print(y3.shape)
y3 = y3.reshape((Nc, 256, 256), order='C')
 
y4 =  numpy.einsum('ijk -> jki', y3) 
# y4 =  numpy.fft.fftshift(numpy.fft.ifftn(numpy.fft.fftshift(y3, axes = (1,2)), axes = (1,2)), axes = (1,2))
y4 =  numpy.fft.fftshift(numpy.fft.ifftn(numpy.fft.fftshift(y4, axes = (0,1)), axes = (0,1)), axes = (0,1))
# y2 = NufftObj.adjoint(y) / 512



# matplotlib.pyplot.imshow(M0[:,:].real)
# matplotlib.pyplot.show()

matplotlib.pyplot.figure('noisy image')

for pp in range(0, Nc):
    matplotlib.pyplot.subplot(2,6,pp + 1)
    matplotlib.pyplot.imshow(abs(multi_image_noisy[:,:,pp]))

# matplotlib.pyplot.show()


matplotlib.pyplot.figure('noise-free image')

for pp in range(0, Nc):
    matplotlib.pyplot.subplot(2,6,pp + 1)
    matplotlib.pyplot.imshow(abs(multi_image[:,:,pp]))

matplotlib.pyplot.figure('coil sensitivities')

for pp in range(0, Nc):
    matplotlib.pyplot.subplot(2,6,pp + 1)
    matplotlib.pyplot.imshow((H[:,:,pp].real))




matplotlib.pyplot.figure('restored coil images')

for pp in range(0, Nc):
    matplotlib.pyplot.subplot(2,6,pp + 1)
    matplotlib.pyplot.imshow((H[:,:,pp]*M0).real)

matplotlib.pyplot.figure('y2')

# for pp in range(0, 1):
matplotlib.pyplot.subplot(1,2,1)
matplotlib.pyplot.imshow( abs(y2 - M0), vmin = 0, vmax = 255)
matplotlib.pyplot.subplot(1,2,2)
matplotlib.pyplot.imshow((y2[:,:]).real, vmin = 0, vmax = 255)


matplotlib.pyplot.figure('y4')
for pp in range(0, Nc):
#     M = 256*256
#     x2 =  NufftObj.solve(y3[M*pp: M*(pp + 1)], 'lsmr', maxiter= 10)
    matplotlib.pyplot.subplot(2,6,pp + 1)
    matplotlib.pyplot.imshow(abs(y4[:,:,pp]))


matplotlib.pyplot.show()

# matplotlib.pyplot.show()

# import  pynufft.src._helper.tensor_coil_sensitivity


# print(c)

