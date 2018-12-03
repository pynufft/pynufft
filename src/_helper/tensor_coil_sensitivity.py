# import phantom
import numpy
import matplotlib.pyplot
matplotlib.pyplot.gray()
import scipy.misc

# import shrinkage_operator
# S = shrinkage_operator.Shrinkage_L1()
import scipy.ndimage.filters

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
    
def Nd_sense(image_stack, maxiter = 20):

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
        H = Nd_smooth(H, 1, -1)



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
        
        
        print('Rayleigh Quotient', rayleigh)
        
#     H = Nd_smooth(H, 3, -1)
    H = H/numpy.mean(numpy.abs(H.ravel()))
    print(numpy.shape(H), type(H))
    return H, mu0, mu

if __name__ == "__main__":

    N= 256
    n_coil = 16
    A = scipy.misc.ascent()
    A = scipy.misc.imresize(A,(N,N)) + 0.0j
#     A= phantom.phantom(N)*(1.0+0.0j)
    
    A[45:77, 45:77] *= (1.0j )/1.414
    
    from multicoil_solver import *
    # matplotlib.pyplot.imshow(A.real)
    # matplotlib.pyplot.show()
    
    coil_sensitivity = create_fake_coils(N, n_coil) # tuple
    multi_image = apply_coil_sensitivities(A, coil_sensitivity, noise_level=5e-20) # tuple
    
    del A # make sure that the original image is not misused.
    
    image_stack = numpy.empty((N,N,n_coil), dtype = numpy.complex)
    for pp in range(0, n_coil):
        image_stack[...,pp] = multi_image[pp]
    
    # H, mu0, mu = tensor_coil_sensitivity(image_stack)
    H, mu0, mu = Nd_sense(image_stack)
    
    show_x = numpy.floor(numpy.sqrt(n_coil))
    show_y = n_coil / show_x
    
    matplotlib.pyplot.figure(0)
    for pp in range(0, n_coil):
        matplotlib.pyplot.subplot(show_x, show_y, pp + 1)
        matplotlib.pyplot.imshow(multi_image[pp].real)
    matplotlib.pyplot.title('coil images')
    # matplotlib.pyplot.show()
    
    matplotlib.pyplot.figure(1)
    for pp in range(0, n_coil):
        matplotlib.pyplot.subplot(show_x, show_y, pp + 1)
        matplotlib.pyplot.imshow(coil_sensitivity[pp].real)
    matplotlib.pyplot.title('coil_sensitivities')
    
    
    matplotlib.pyplot.figure(4)
    for pp in range(0, n_coil):
        matplotlib.pyplot.subplot(show_x, show_y, pp + 1)
        matplotlib.pyplot.imshow(numpy.real(H[...,pp]*numpy.conj(H[...,pp])))
    matplotlib.pyplot.title('estimated sensitivities')
    # matplotlib.pyplot.show()
    
    matplotlib.pyplot.show()
    
    matplotlib.pyplot.figure(0)
    for pp in range(0, n_coil):
        matplotlib.pyplot.subplot(show_x, show_y, pp + 1)
        matplotlib.pyplot.imshow(multi_image[pp].imag)
    matplotlib.pyplot.title('coil images')
    # matplotlib.pyplot.show()
    
    matplotlib.pyplot.figure(1)
    for pp in range(0, n_coil):
        matplotlib.pyplot.subplot(show_x, show_y, pp + 1)
        matplotlib.pyplot.imshow(coil_sensitivity[pp].imag)
    matplotlib.pyplot.title('coil_sensitivities')
    
    
    matplotlib.pyplot.figure(4)
    for pp in range(0, n_coil):
        matplotlib.pyplot.subplot(show_x, show_y, pp + 1)
        matplotlib.pyplot.imshow(H[...,pp].imag)
    matplotlib.pyplot.title('estimated coil')
    # matplotlib.pyplot.show()
    
    matplotlib.pyplot.figure(99)
    matplotlib.pyplot.imshow(numpy.real(mu0), )
    matplotlib.pyplot.title('mu')
    
    matplotlib.pyplot.figure(98)
    matplotlib.pyplot.imshow(numpy.real(numpy.mean(image_stack,2)), )
    matplotlib.pyplot.title('average')
    # import scipy.ndimage.filters
    # 
    # matplotlib.pyplot.figure(120)
    # # matplotlib.pyplot.imshow(scipy.ndimage.filters.gaussian_filter(numpy.real(numpy.mean(image_stack,2)/mu),    sigma = 4),  vmin=-1., vmax=1.)
    # matplotlib.pyplot.imshow(numpy.real(image_stack[...,2]/mu),  vmin=-1., vmax=1.)
    
    matplotlib.pyplot.figure(100)
    matplotlib.pyplot.imshow(numpy.real(numpy.mean(image_stack*numpy.conj(image_stack),2)**0.5))
    
    matplotlib.pyplot.show()
