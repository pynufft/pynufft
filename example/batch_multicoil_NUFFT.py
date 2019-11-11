# Generating trajectories for Cartesian k-space
import numpy
import matplotlib.pyplot
matplotlib.pyplot.gray()
def create_fake_coils(N, n_coil):
    dtype = numpy.complex64
    xx,yy = numpy.meshgrid(numpy.arange(0,N),numpy.arange(0,N))
    
#     ZZ= numpy.exp(-((xx-128)/32)**2-((yy-128)/32)**2)     
    coil_sensitivity = []
    
    image_sense = ()
    
    r = N
    phase_factor =  0
    sense_array = numpy.empty( (N, N, n_coil), dtype = dtype)
    for nn in range(0,n_coil):
        
        tmp_angle = nn*2*numpy.pi/n_coil
        shift_r = int(N)
        shift_x= (numpy.cos(tmp_angle)*shift_r).astype(dtype)
        shift_y= (numpy.sin(tmp_angle)*shift_r).astype(dtype)
#         ZZ= numpy.exp(-((xx-N/2-shift_x)/r)**2-((yy-N/2-shift_y)/r)**2).astype(numpy.complex64)
        ZZ= numpy.exp(1.0j * numpy.random.randn()*0)* numpy.exp(-phase_factor *1.0j*((xx-N/2-shift_x)/N + (yy-N/2-shift_y)/N))* numpy.exp(-((xx-N/2-shift_x)/r)**2-((yy-N/2-shift_y)/r)**2).astype(numpy.complex64)  
#         coil_sensitivity +=(numpy.roll(numpy.roll(ZZ,shift_x,axis=0),shift_y,axis=1),)
        coil_sensitivity +=[ZZ,]
        sense_array[:,:, nn] = ZZ
#     if n_coil > 1:
#         coil_sensitivity[0] = coil_sensitivity[0] + coil_sensitivity[1]
    
    
    
    return sense_array
def fake_Cartesian(Nd):
    dim = len(Nd) # dimension
    M = numpy.prod(Nd)
    om = numpy.zeros((M, dim), dtype = numpy.float)
    grid = numpy.indices(Nd)
    for dimid in range(0, dim):
        om[:, dimid] = (grid[dimid].ravel() *2/ Nd[dimid] - 1.0)*numpy.pi
    return om    


# Begin of test_batch_NUFFT()
def test_batch_NUFFT():
    
    import scipy.misc
    from pynufft import NUFFT_cpu, NUFFT_hsa
    
    Nd = (256,256)
    Kd = (512,512)
    Jd = (6,6)
    
    image = scipy.misc.ascent()
    image = scipy.misc.imresize(image, Nd).astype(numpy.complex64)
    om = fake_Cartesian(Nd)
    
    batch = 3
    print('Number of samples (M) = ', om.shape[0])
    print('Dimension = ', om.shape[1])
    print('Nd = ', Nd)
    print('Kd = ', Kd)
    print('Jd = ', Jd)
     
#     NufftObj = NUFFT_cpu()
    NufftObj = NUFFT_hsa()
    NufftObj.plan(om, Nd, Kd, Jd, ft_axes = (0,1),  batch=batch)
    
    # Now transform 1 image to multiple channels using forward_one2many() method
#     y = NufftObj.forward_one2many(image) # for NUFFT_cpu()
    multi_image = numpy.broadcast_to(numpy.reshape(image, Nd + (1,)), Nd + (batch,))
    y0 = NufftObj.forward(multi_image)
    y = y0.get()

#     y = NufftObj.forward_one2many(image).get() # for NUFFT_hsa()
    
    # Now reshape the data for IFFT
#     y2 = y.reshape(Nd + (batch, ), order='C') 
    
    # Perform IFFT to check the correctness of the k-space
#     x2 = numpy.fft.ifftshift(
#                 numpy.fft.ifftn(
#                     numpy.fft.ifftshift(y2, axes = (0,1)
#                     ), axes = (0,1)
#                 ), axes = (0,1)
#             )
    
    x2 = NufftObj.adjoint(y0).get()
    
    
    # display the result
    for pp in range(0, batch):
        matplotlib.pyplot.subplot(batch,    3, 1 + pp*3)
        matplotlib.pyplot.imshow(image.real, )#vmin = 0, vmax = 255)
        matplotlib.pyplot.title('Original image')
        matplotlib.pyplot.subplot(batch,3,2 + pp*3)
        matplotlib.pyplot.imshow(x2[:,:,pp].real, )#vmin = 0, vmax = 255)
        matplotlib.pyplot.title('Restored image  of coil '+str(pp + 1))
#         matplotlib.pyplot.subplot(batch,3,3 + pp*3)
#         matplotlib.pyplot.imshow(abs(image - x2[:,:,pp]), vmin = 0, vmax = 255)
#         matplotlib.pyplot.title('Difference map of coil '+str(pp + 1))
    matplotlib.pyplot.show()
    # end of test_batch_NUFFT()


# Begin of test_multicoil_NUFFT()
def test_senselike_NUFFT():
    
    import scipy.misc
    from pynufft import NUFFT_cpu, NUFFT_hsa
    
    Nd = (256,256)
    Kd = (512,512)
    Jd = (6,6)
    
    image = scipy.misc.ascent()
    image = scipy.misc.imresize(image, Nd).astype(numpy.complex64)
    om = fake_Cartesian(Nd)
    
    batch = 3
    print('Number of samples (M) = ', om.shape[0])
    print('Dimension = ', om.shape[1])
    print('Nd = ', Nd)
    print('Kd = ', Kd)
    print('Jd = ', Jd)
     
#     NufftObj = NUFFT_cpu()
    NufftObj = NUFFT_hsa()
    NufftObj.plan(om, Nd, Kd, Jd, ft_axes = (0,1),batch=batch)
    
    # Now create fake coil sensitivity profiles
    sense_array = create_fake_coils(256, batch)
    
    # set_sense() method apply the sensitivities to the object
    NufftObj.set_sense(sense_array)
    
#     NufftObj.reset_sense() # reset the coil sensitivity profile
    
    # Now transform 1 image to multiple channels using forward_one2many() method
#     y = NufftObj.forward_one2many(image) # for NUFFT_cpu()

#     multi_image = numpy.broadcast_to(numpy.reshape(image, Nd + (1,)), Nd + (batch,))
#     y = NufftObj.forward(multi_image).get()

    y0 = NufftObj.forward_one2many(image)# for NUFFT_hsa()
    y = y0.get()
    # Now reshape the data for IFFT
#     y2 = y.reshape(Nd +(batch, ) , order='C') 
#     
#     # Perform IFFT to check the correctness of the k-space
#     x2 = numpy.fft.ifftshift(
#                 numpy.fft.ifftn(
#                     numpy.fft.ifftshift(y2, axes = (0,1)
#                     ), axes = (0,1)
#                 ), axes = (0,1)
#             )
    
    x2 = NufftObj.adjoint(y0).get()
    
    # display the result
    for pp in range(0, batch):
        matplotlib.pyplot.subplot(batch,    3, 1 + pp*3)
        matplotlib.pyplot.imshow(image.real, vmin = 0, vmax = 255)
        matplotlib.pyplot.title('Original image')
        
        matplotlib.pyplot.subplot(batch,3,2 + pp*3)
        matplotlib.pyplot.imshow(sense_array[:,:,pp].real)
        matplotlib.pyplot.title('Simulated coil sensitivity '+str(pp + 1))
        
        matplotlib.pyplot.subplot(batch,3,3 + pp*3)
        matplotlib.pyplot.imshow(x2[:,:,pp].real)
        matplotlib.pyplot.title('Restored image of coil '+str(pp + 2))

    matplotlib.pyplot.show()
    # end of test_multicoil_NUFFT()
    
if __name__ == '__main__':
    test_batch_NUFFT()  # no sense, multiply sense outside the forward() method
    test_senselike_NUFFT()    # multiplication of sense happens inside the forward_one2many() method