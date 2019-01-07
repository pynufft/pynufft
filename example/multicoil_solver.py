import numpy
import scipy
import matplotlib.pyplot
# import pyximport

# pyximport.install(setup_args={'include_dirs': numpy.get_include()})

dtype = numpy.complex64
import scipy.linalg
def tailor_fftn(X):
#     X = numpy.fft.fftshift(numpy.fft.fftn(numpy.fft.fftshift((X))))
    X = numpy.fft.fftshift(numpy.fft.fftn(numpy.fft.fftshift((X))))
    return X
def tailor_ifftn(X):
#     X = numpy.fft.fftshift(numpy.fft.ifftn(numpy.fft.ifftshift(X)))
    X = numpy.fft.fftshift(numpy.fft.ifftn(numpy.fft.ifftshift(X)))
    return X
import scipy.sparse.linalg
def myeig(g):
    '''
    access lapack's cggev which should be faster 
     than scipy's eig() 
    '''
#     w,wr,vl ,v, info= scipy.linalg.lapack.sgeev(g ,compute_vl= 0,compute_vr = 1, overwrite_a=1)
    w,v, info= scipy.linalg.lapack.ssyev(g , overwrite_a=1) # Lapack's ssyev routine for 
                                                            # symmetric eigenproblem (20% faster 
                                                            #than complex cgeev and 100% faster than 
                                                            #zgeev) 
#     w,v = scipy.sparse.linalg.eigsh(g ,1, which='LM') # Lapack's ssyev routine for 
#                                                             # symmetric eigenproblem (20% faster 
#                                                             #than complex cgeev and 100% faster than 
#                                                             #zgeev)
    return w,v
 
def create_fake_coils(N, n_coil):
    
    xx,yy = numpy.meshgrid(numpy.arange(0,N),numpy.arange(0,N))
    
    
#     
#     ZZ= numpy.exp(-((xx-128)/32)**2-((yy-128)/32)**2)     
    coil_sensitivity = []
    
    image_sense = ()
    
    r = N/2
    phase_factor =  2
    for nn in range(0,n_coil):
        
        tmp_angle = nn*2*numpy.pi/n_coil
        shift_r = int(N/3)
        shift_x= (numpy.cos(tmp_angle)*shift_r).astype(dtype)
        shift_y= (numpy.sin(tmp_angle)*shift_r).astype(dtype)
#         ZZ= numpy.exp(-((xx-N/2-shift_x)/r)**2-((yy-N/2-shift_y)/r)**2).astype(numpy.complex64)
        ZZ= numpy.exp(1.0j * numpy.random.randn()*0)* numpy.exp(-phase_factor *1.0j*((xx-N/2-shift_x)/N + (yy-N/2-shift_y)/N))* numpy.exp(-((xx-N/2-shift_x)/r)**2-((yy-N/2-shift_y)/r)**2).astype(numpy.complex64)  
#         coil_sensitivity +=(numpy.roll(numpy.roll(ZZ,shift_x,axis=0),shift_y,axis=1),)
        coil_sensitivity +=[ZZ,]
#     if n_coil > 1:
#         coil_sensitivity[0] = coil_sensitivity[0] + coil_sensitivity[1]
    return coil_sensitivity

def apply_coil_sensitivities(input_image, coil_sensitivity, noise_level=0.02):
    
#     shape = input_image.shape
#     print(shape, shape[0], shape[1])
    output_image = () # coil_sensitivity
    n_coil = len(coil_sensitivity)
    shape = input_image[0].shape
    print(shape[0])
    threshold =numpy.max( input_image[...])*noise_level
    for pp in range(n_coil):
        output_image += (input_image*coil_sensitivity[pp]+  threshold*(numpy.random.randn(shape[0],shape[0]) + 1j*numpy.random.randn(shape[0],shape[0])),)
#         matplotlib.pyplot.imshow(output_image[pp].real, cmap=matplotlib.cm.gray)
#         matplotlib.pyplot.show()    
    
    
    return output_image
def fft_multi_image(input_image):
    n_coil = len(input_image)
    multi_kspace=()
    for pp in range(n_coil):
        multi_kspace += (tailor_fftn(input_image[pp]), )
    return multi_kspace

def crop_kspace(single_kspace, shift_x, shift_y, dim_x, dim_y): # numpy.ndarray
    
    
    return calibration_colume

def build_A(multi_kspace, reps_acs, mysize): # tuple
    # building calibration matrix A

     
    half_size = int(mysize/2) # half length of one side of the square
    Nx = multi_kspace[0].shape[0] # size of Nx
    Ny = multi_kspace[0].shape[1] # size of Ny
    reps_dimx = int(reps_acs**0.5)
    reps_dimy = int(reps_acs/reps_dimx)
    nx_min = int( Nx/2 - half_size - reps_dimx)
    nx_max = int(Nx/2 + half_size - reps_dimx)
    ny_min =int(Ny/2 - half_size - reps_dimy)
    ny_max = int(Ny/2 + half_size - reps_dimy)
    
    n_coil = len(multi_kspace)
    
    # now allocation A matrix row = mysize**2, colume= reps_acs * n_coil
    counter = 0
    A = numpy.empty( (mysize**2, reps_acs * n_coil), dtype = dtype)
    
    for nn in range(0,n_coil):
        for jjx in range(0, reps_dimx):
            for jjy in range(0, reps_dimy):
                

                A[:,counter]=multi_kspace[nn][nx_min + jjx: nx_max+jjx, ny_min+jjy:ny_max+jjy].reshape((int(mysize**2),),order='F')
                counter += 1

    return A

def build_V_KN(V,  reps_acs,   mysize, K,  shape_of_image):
    
    V_parallel = V[:,0:60]
    
    n_coils = int(V.shape[0]/(reps_acs))
    
    half_size = int(mysize/2) # half length of one side of the square

    reps_dimx = int(reps_acs**0.5)
    
    reps_dimy = int(reps_acs/reps_dimx)
    
    tmp_filter = numpy.zeros( shape_of_image, dtype=dtype)
    
    x0 = int(shape_of_image[0]/2)
    y0 = int(shape_of_image[0]/2) 
    dx = int(reps_dimx/2)
    dy = int(reps_dimy/2)

    prod_image_shape = numpy.prod(shape_of_image)
#     V_KN = ()
    V_KN = numpy.zeros((K,n_coils, prod_image_shape),dtype=dtype)
    
    for kk in range(K):

        for nn in range(n_coils):
            tmp_vec = V[ nn*reps_acs: (nn+1)*reps_acs , kk]
            
            x_min = x0 - dx
            x_max = x0 + dx
            y_min = y0 - dy
            y_max = y0 + dy
            
            '''
            Note: flipped
            '''
            tmp_filter[x_max:x_min:-1, y_max:y_min:-1] = tmp_vec.reshape( (reps_dimx, reps_dimy),order='F').T
#             print(tmp_filter.shape)
#             tmp_vkn += (tailor_ifftn(tmp_filter).reshape(prod_image_shape,order='F'), )
            V_KN[kk,nn,:]=tailor_ifftn(tmp_filter).reshape(prod_image_shape,order='F')
#         V_KN += (tmp_vkn,   ) # [K][n_coil][shape_of_image]
#             if nn == 6:
#             print(kk, nn)
#             matplotlib.pyplot.imshow(tailor_ifftn(tmp_filter).imag)
#             matplotlib.pyplot.show()
#     Gq = numpy.empty((K, N), dtype=dtype)   
            
    return tuple(V_KN) # [K][n_coil][ numpy.prod(shape_of_image) ]
def V_KN_to_Gq(V_KN2, Gq,    K, n_coil   ,jj):
     
#     for kk in range(0,K):
#         for nn in range(0,n_coil):
#             Gq[kk, nn]   =   V_KN[kk][nn][jj]
#     g= Gq.conj().T.dot(Gq)
 
    g = V_KN2[...,jj].T.dot(V_KN2[...,jj].conj())
     
    return g
def build_maps(V_KN,    shape_of_image):
     
    result_coil_sensitivity = ()
    K = len(V_KN)
     
    n_coil = len(V_KN[0])
    print(K,n_coil)
    prod_image_shape = numpy.prod(shape_of_image) # fortran order
     
    Gq = numpy.empty((K, n_coil),dtype=dtype)
    g=numpy.empty((n_coil, n_coil),dtype=dtype)
     
    C = numpy.ones( (n_coil, prod_image_shape ) ,dtype=dtype )
    V_KN2 = numpy.array(V_KN)
    print(numpy.shape(V_KN2))
    for jj in range(prod_image_shape):
  
#         for kk in range(0,K):
#             for nn in range(0,n_coil):
# #                 V_KN[kk][nn][jj]
#                 Gq[kk, nn]   =   V_KN[kk][nn][jj]
#         g[:,:]= Gq.conj().T.dot(Gq)
        g=V_KN_to_Gq(V_KN2, Gq,    K, n_coil   ,jj)
#         w, v = numpy.linalg.eig(g)
        w, v = myeig(g.real)
        ind = numpy.argmax(numpy.abs(w)) # find the maximum eigenvalue
         
        C[:,jj] = v[:,ind]
#     import matplotlib.pyplot as pyplot
#     pyplot.imshow(g.real)
#     pyplot.show()
  
    C= C/numpy.exp( 1.0j* numpy.angle(C))
#     C = numpy.ones_like(C)
#     print(C.shape, K, shape_of_image)
    C = C.reshape((n_coil,)+  shape_of_image,order='F')
    C= tuple(C)
    return C #result_coil_sensitivity    
    
def build_maps1(V_KN,    shape_of_image):
    
    result_coil_sensitivity = ()
    K = len(V_KN)
    
    n_coil = len(V_KN[0])
    print(K,n_coil)
    prod_image_shape = numpy.prod(shape_of_image) # fortran order
    
#     Gq = numpy.empty((K, n_coil),dtype=dtype)
#     g=numpy.empty((n_coil, n_coil),dtype=dtype)
    
    C = numpy.ones( (n_coil, prod_image_shape  ), numpy.complex64)
    V_KN2 = numpy.array(V_KN)
#     print(numpy.dot(V_KN2[0,0,:],V_KN2[1,0,:]))
#     V_KN3 = V_KN2.conj().copy()
    vmat0 = numpy.transpose(V_KN2,(2,1,0))
    vmat1 = numpy.transpose(V_KN2,(2,0,1)).copy()
    vmat2 = numpy.matmul(vmat0.conj(), vmat1)
    V_KN4 = numpy.transpose(vmat2, (1,2,0))
#     print('KN4 = ',V_KN4.shape)
#     print('iter = ', 0)
    for pp in range(0, 300):
        """
        Power iteration
        """
        v1 = numpy.einsum('ijk,jk->ik',V_KN4, C)

        v1_norm_2 = numpy.tile(numpy.sum( v1*v1.conj(), 0), (n_coil,1))
#         v1_norm_2 = numpy.sum( v1*v1.conj()[...])
#         print(v1_norm_2.shape)
#         C = v1
        C = v1/v1_norm_2
#     print('iter = ', pp)        
#     for jj in range(prod_image_shape):
#  
# #         for kk in range(0,K):
# #             for nn in range(0,n_coil):
# # #                 V_KN[kk][nn][jj]
# #                 Gq[kk, nn]   =   V_KN[kk][nn][jj]
# #         g[:,:]= Gq.conj().T.dot(Gq)
#         g=V_KN_to_Gq(V_KN2, Gq,    K, n_coil   ,jj)
# #         w, v = numpy.linalg.eig(g)
#         w, v = myeig(g.real)
#         ind = numpy.argmax(numpy.abs(w)) # find the maximum eigenvalue
#         
#         C[:,jj] = v[:,ind]
#     import matplotlib.pyplot as pyplot
#     pyplot.imshow(g.real)
#     pyplot.show()
 
#     C= C/numpy.exp( 1.0j* numpy.angle(C))
#     print(C.shape, K, shape_of_image)
    v1 = numpy.einsum('ijk,jk->ik',V_KN4, C)
#         print('v1shape',v1.shape)
    v1_norm_2 = numpy.tile(numpy.sum( C*C.conj(), 0), (n_coil,1))
    mu = numpy.abs(v1 * C.conj() / v1_norm_2)
    mu = mu / numpy.max(mu)
#     ind = (mu > 0.1)
#     C2 = numpy.ones_like(C)
#     C2[ind] = C[ind]
    C = C*mu
    C = C.reshape((n_coil,)+  shape_of_image,order='F')
    C= tuple(C)
    return C #result_coil_sensitivity
def image2coilprofile(multi_image):
    multi_image = remove_low_frequency_phase(multi_image)
    multi_kspace = fft_multi_image(multi_image) # tuple
    
    N = multi_image[0].shape[0]
    
    reps_acs = 16 # number of adjacent squares for SVD
    mysize = 16 # length of one side of the square
    print('Using ESPIRIT coil decomposition method')
    A= build_A(multi_kspace,    reps_acs,   mysize) # build a 2-D calibration matrix
#     print('215')
    U,S,Vh = numpy.linalg.svd(A)
#     print('216')
    V = Vh.conj().T
#     import matplotlib
#     matplotlib.pyplot.plot(S)
#     matplotlib.pyplot.show()
#     print('217')
    ind = S > 0.1*S[0]
    print(ind[-1])
    K = len(S[ind])
    shape_of_image=(N, N)
#     print('222')
    V_KN =  build_V_KN(V,  reps_acs,   mysize, K, shape_of_image)
#     print('224')
# 
#     print("V_KN", numpy.shape(V_KN) )
    result_coil_sensitivity = build_maps1(V_KN,  shape_of_image)
#     print('228')
    
    
    
    return result_coil_sensitivity



def remove_low_frequency_phase( stack_image_input):
    print('Removing low-frequency phase')
    import scipy.ndimage.filters
    Nc = len(stack_image_input)
    stack_image_output = ()
    shape = stack_image_input[0].shape
    gauss_rad = 1
    for nc in range(0, Nc):
        gfilter = scipy.ndimage.filters.gaussian_filter(stack_image_input[nc].real, gauss_rad) + scipy.ndimage.filters.gaussian_filter(stack_image_input[nc].imag, gauss_rad) * 1.0j
#     import matplotlib.pyplot
    
        low_pass_phase = (gfilter+1e-7) / numpy.abs(gfilter+1e-7)
        stack_image_output += ((gfilter+1e-7)/(low_pass_phase+1e-7), )
#     matplotlib.pyplot.imshow(gfilter.real)
#     matplotlib.pyplot.show()
    
    
    
    return stack_image_output
def image2coilprofile3(multi_image):
    print('Using exponent of sos')
    multi_image = remove_low_frequency_phase(multi_image)
    Nc = len(multi_image)
    import scipy.ndimage.filters

    result_coil_sensitivity = ()
    sos_image = numpy.zeros_like(multi_image[0])
    low_reso_image = ()

    gauss_rad = 0.1
    for nc in range(0, Nc):
        gfilter = scipy.ndimage.filters.gaussian_filter(multi_image[nc].real, gauss_rad) + scipy.ndimage.filters.gaussian_filter(multi_image[nc].imag, gauss_rad) * 1.0j
        low_reso_image += (gfilter, )
#     import matplotlib.pyplot
        sos_image +=  gfilter**2
    sos_image = sos_image**0.5
#     thr = 0.01* numpy.max(sos_image[...])
    for nc in range(0, Nc):
        result_coil_sensitivity += (   numpy.exp(low_reso_image[nc] )/numpy.exp(sos_image )  ,)
    
    
    return result_coil_sensitivity

def image2coilprofile2(multi_image):
    """
    image divided by the root sum of squares
    """
    print("coil profile by image/sos")
#     multi_image = remove_low_frequency_phase(multi_image)
    Nc = len(multi_image)
    import scipy.ndimage.filters

    result_coil_sensitivity = ()
    sos_image = numpy.zeros_like(multi_image[0])
    low_reso_image = ()

    gauss_rad = 1
    for nc in range(0, Nc):
        gfilter_real = scipy.ndimage.filters.gaussian_filter(multi_image[nc].real, gauss_rad) 
        gfilter_imag =  scipy.ndimage.filters.gaussian_filter(multi_image[nc].imag, gauss_rad) 
        low_reso_image += (gfilter_real + gfilter_imag*1.0j, )
#     import matplotlib.pyplot
        sos_image_real =  numpy.hypot( sos_image.real, gfilter_real)
        sos_image_imag=  numpy.hypot( sos_image.imag, gfilter_imag)
        sos_image = numpy.hypot( sos_image_real, sos_image_imag)
#     sos_image = sos_image**0.5
    thr = 0.1* numpy.max(sos_image[...])
    gauss_rad = 120
    for nc in range(0, Nc):
        result_coil_sensitivity += ( scipy.ndimage.filters.gaussian_filter( (low_reso_image[nc]+thr).real/(sos_image+thr), gauss_rad)+
                                     1.0j*scipy.ndimage.filters.gaussian_filter( (low_reso_image[nc]+thr).imag/(sos_image+thr), gauss_rad),)
    
    """
    A final Espirit process
    """
#     result_coil_sensitivity = image2coilprofile(result_coil_sensitivity) 
    return result_coil_sensitivity

def test_solver_cpu():
    '''
    Test the coil decomposition method
    '''
    
    import phantom
    N = 256
    a = phantom.phantom(N).astype(numpy.complex64)
#     a += numpy.random.randn(256,256) + 1j*numpy.random.randn(256,256)
#     a = scipy.misc.ascent()
    
    n_coil = 8
    coil_sensitivity = create_fake_coils(N, n_coil) # tuple
    multi_image = apply_coil_sensitivities(a, coil_sensitivity) # tuple
#     multi_image = remove_low_frequency_phase(multi_image)
    result_coil_sensitivity =image2coilprofile(multi_image)
    
    
    print(type(result_coil_sensitivity))

#     A = svd_coil_sense(multi_image)
    for pp in range(0,n_coil):
        matplotlib.pyplot.subplot(6,4,pp+1)
        matplotlib.pyplot.imshow(result_coil_sensitivity[pp][...].real, cmap=matplotlib.cm.gray)
        matplotlib.pyplot.subplot(6,4,pp+9)
        matplotlib.pyplot.imshow(multi_image[pp][...].real, cmap=matplotlib.cm.gray)
        matplotlib.pyplot.subplot(6,4,pp+17)
        matplotlib.pyplot.imshow(coil_sensitivity[pp][...].real, cmap=matplotlib.cm.gray)       
    matplotlib.pyplot.show()
def test_tensor():
    '''
    Test the coil decomposition method
    '''
    
    import phantom
    N = 256
    a = phantom.phantom(N).astype(numpy.complex64)
#     a += numpy.random.randn(256,256) + 1j*numpy.random.randn(256,256)
#     a = scipy.misc.ascent()
    
    n_coil = 8
    coil_sensitivity = create_fake_coils(N, n_coil) # tuple
    multi_image = apply_coil_sensitivities(a, coil_sensitivity) # tuple
    import Nd_tensor
    coil_array = numpy.empty((N, N, n_coil), dtype = numpy.complex)
    for pp in range(0, n_coil):
        coil_array[:,:,pp] = multi_image[pp]
#     multi_image = remove_low_frequency_phase(multi_image)
#     result_coil_sensitivity =image2coilprofile2(multi_image)
    sos =  numpy.sum(coil_array**2, 2)**0.5
    for pp in range(0, n_coil):
        coil_array[:,:,pp] /= sos 

    H = Nd_tensor.htensor()
    
    H.factorise(coil_array, (5,5, n_coil))
    
    core = H.dot(coil_array)
    core[numpy.abs(core)< 0.5 ] = 0
    
    recovered = H.adjoint(core)
#     sos =  numpy.sum(recovered**2, 2)
    result_coil_sensitivity = ()
    for pp in range(0, n_coil):
        result_coil_sensitivity += (recovered [:,:,pp] ,)    
    
    
    print(type(result_coil_sensitivity))

#     A = svd_coil_sense(multi_image)
    for pp in range(0,n_coil):
        matplotlib.pyplot.subplot(6,4,pp+1)
        matplotlib.pyplot.imshow(result_coil_sensitivity[pp][...].real, cmap=matplotlib.cm.gray)
        matplotlib.pyplot.subplot(6,4,pp+9)
        matplotlib.pyplot.imshow(multi_image[pp][...].real, cmap=matplotlib.cm.gray)
        matplotlib.pyplot.subplot(6,4,pp+17)
        matplotlib.pyplot.imshow(coil_sensitivity[pp][...].real, cmap=matplotlib.cm.gray)       
    matplotlib.pyplot.show()
# def test_solver_gpu():
#     '''
#     Test the soil decomposition method
#     '''
#     
#     import phantom
#     N = 128
#     a = phantom.phantom(N).astype(numpy.complex64)
#     
#     n_coil = 8
#     coil_sensitivity = create_fake_coils(N, n_coil) # tuple
#     multi_image = apply_coil_sensitivities(a, coil_sensitivity) # tuple
#     
#     multi_kspace = fft_multi_image(multi_image) # tuple
#     reps_acs = 16 # number of adjacent squares for SVD
#     mysize = 20 # length of one side of the square
#     A= build_A(multi_kspace,    reps_acs,   mysize) # build a 2-D calibration matrix
#     
#     U,S,Vh = numpy.linalg.svd(A)
#     V = Vh.conj().T
#     
#     K = 30
#     shape_of_image=(N, N)
#     V_KN =  build_V_KN(V,  reps_acs,   mysize, K, shape_of_image)
#     print("V_KN.shape", V_KN.shape)
# 
#     
#     result_coil_sensitivity = build_maps(V_KN,  shape_of_image)
#     print(type(result_coil_sensitivity))
# 
# #     A = svd_coil_sense(multi_image)
#     for pp in range(0,n_coil):
#         matplotlib.pyplot.subplot(6,4,pp+1)
#         matplotlib.pyplot.imshow(result_coil_sensitivity[pp][...].real, cmap=matplotlib.cm.gray)
#         matplotlib.pyplot.subplot(6,4,pp+9)
#         matplotlib.pyplot.imshow(multi_image[pp][...].real, cmap=matplotlib.cm.gray)
#         matplotlib.pyplot.subplot(6,4,pp+17)
#         matplotlib.pyplot.imshow(coil_sensitivity[pp][...].real, cmap=matplotlib.cm.gray)       
#     matplotlib.pyplot.show()

    
if __name__ == "__main__":
#     test()
    import cProfile
    cProfile.run('test_solver_cpu()')
#     test_tensor()
#     remove_low_frequency_phase( )
    
    
