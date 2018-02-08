"""
Helper functions
=======================================

bugfix: mm = numpy.tile(mm, [numpy.prod(Jd).astype(int), 1])  to fix wrong type when numpy.prod(Jd) is not casted as int

"""


import numpy
dtype = numpy.complex64
import scipy

def create_laplacian_kernel(nufft):
    """
    Create the multi-dimensional laplacian kernel in k-space
    
    :param nufft: the NUFFT object
    :returns: uker: the multi-dimensional laplacian kernel in k-space (no fft shift used)
    :rtype: numpy ndarray
    """
#===============================================================================
# #        # Laplacian oeprator, convolution kernel in spatial domain
#         # related to constraint
#===============================================================================
    uker = numpy.zeros(nufft.st['Kd'][:],dtype=numpy.complex64,order='C')
    n_dims= numpy.size(nufft.st['Nd'])

#     if n_dims == 1:
#         uker[0] = -2.0
#         uker[1] = 1.0
#         uker[-1] = 1.0
#     elif n_dims == 2:
#         uker[0,0] = -4.0
#         uker[1,0] = 1.0
#         uker[-1,0] = 1.0
#         uker[0,1] = 1.0
#         uker[0,-1] = 1.0
#     elif n_dims == 3:  
#         uker[0,0,0] = -6.0
#         uker[1,0,0] = 1.0
#         uker[-1,0,0] = 1.0
#         uker[0,1,0] = 1.0
#         uker[0,-1,0] = 1.0
#         uker[0,0,1] = 1.0
#         uker[0,0,-1] = 1.0      
#     else: # n_dims > 3
################################
#    Multi-dimensional laplacian kernel (generalize the above 1D - 3D to multi-dimensional arrays)
################################
    indx = [slice(0, 1) for ss in range(0, n_dims)] # create the n_dims dimensional slice which are all zeros
    uker[indx] = - 2.0*n_dims # Equivalent to  uker[0,0,0] = -6.0
    for pp in range(0,n_dims):
        indx1 = indx.copy() # indexing the 1
        indx1[pp] = 1
        uker[indx1] = 1
        indx1 = indx.copy() # indexing the -1
        indx1[pp] = -1
        uker[indx1] = 1
################################
#    FFT of the multi-dimensional laplacian kernel
################################        
    uker =numpy.fft.fftn(uker) #, self.nufftobj.st['Kd'], range(0,numpy.ndim(uker)))
    return uker  
def indxmap_diff(Nd):
    """
    Preindixing for rapid image gradient ()
    
    Diff(x) = x.flat[d_indx[0]] - x.flat
    
    Diff_t(x) =  x.flat[dt_indx[0]] - x.flat
                            
    :param Nd: the dimension of the image
    :type Nd: tuple with integers
    :returns d_indx: iamge gradient
    :returns  dt_indx:  the transpose of the image gradient 
    :rtype: d_indx: lists with numpy ndarray
    :rtype: dt_indx: lists with numpy ndarray
    """    

    ndims = len(Nd)
    Ndprod = numpy.prod(Nd)
    mylist = numpy.arange(0, Ndprod).astype(numpy.int32)
    mylist = numpy.reshape(mylist, Nd)
    d_indx = []
    dt_indx = []
    for pp in range(0, ndims):
        d_indx = d_indx + [ numpy.reshape(   numpy.roll(  mylist, +1 , pp  ), (Ndprod,)  ,order='C').astype(numpy.int32) ,]
        dt_indx = dt_indx + [ numpy.reshape(   numpy.roll(  mylist, -1 , pp  ) , (Ndprod,) ,order='C').astype(numpy.int32) ,]

    return d_indx,  dt_indx  

def plan(om, Nd, Kd, Jd):
#         self.debug = 0  # debug

    if type(Nd) != tuple:
        raise TypeError('Nd must be tuple, e.g. (256, 256)')

    if type(Kd) != tuple:
        raise TypeError('Kd must be tuple, e.g. (512, 512)')

    if type(Jd) != tuple:
        raise TypeError('Jd must be tuple, e.g. (6, 6)')

    if (len(Nd) != len(Kd)) | (len(Nd) != len(Jd))  | len(Kd) != len(Jd):
        raise KeyError('Nd, Kd, Jd must be in the same length, e.g. Nd=(256,256),Kd=(512,512),Jd=(6,6)')

    dd = numpy.size(Nd)

###############################################################
# check input errors
###############################################################
    st = {}
    ud = {}
    kd = {}

###############################################################
# First, get alpha and beta: the weighting and freq
# of formula (28) in Fessler's paper
# in order to create slow-varying image space scaling
###############################################################
    for dimid in range(0, dd):
        (tmp_alpha, tmp_beta) = nufft_alpha_kb_fit(
            Nd[dimid], Jd[dimid], Kd[dimid])
        st.setdefault('alpha', []).append(tmp_alpha)
        st.setdefault('beta', []).append(tmp_beta)
    st['tol'] = 0
    st['Jd'] = Jd
    st['Nd'] = Nd
    st['Kd'] = Kd
    M = om.shape[0]
    st['M'] = numpy.int32(M)
    st['om'] = om
    st['sn'] = numpy.array(1.0 + 0.0j)
    dimid_cnt = 1
###############################################################
# create scaling factors st['sn'] given alpha/beta
# higher dimension implementation
###############################################################
    for dimid in range(0, dd):
        tmp = nufft_scale(
            Nd[dimid],
            Kd[dimid],
            st['alpha'][dimid],
            st['beta'][dimid])
        dimid_cnt = Nd[dimid] * dimid_cnt
###############################################################
# higher dimension implementation: multiply over all dimension
###############################################################
        st['sn'] = numpy.dot(st['sn'], tmp.T)
        st['sn'] = numpy.reshape(st['sn'], (dimid_cnt, 1), order='C')
        # JML do not apply scaling

    # order = 'F' is for fortran order
    st['sn'] = st['sn'].reshape(Nd, order='C')  # [(Nd)]
    ###############################################################
    # else:
    #     st['sn'] = numpy.array(st['sn'],order='F')
    ###############################################################

    st['sn'] = numpy.real(st['sn'])  # only real scaling is relevant

    # [J? M] interpolation coefficient vectors.
    # Iterate over all dimensions and
    # multiply the coefficients of all dimensions
    for dimid in range(0, dd):  # loop over dimensions
        N = Nd[dimid]
        J = Jd[dimid]
        K = Kd[dimid]
        alpha = st['alpha'][dimid]
        beta = st['beta'][dimid]
        ###############################################################
        # formula 29 , 26 of Fessler's paper
        ###############################################################

        # pseudo-inverse of CSSC using large N approx [J? J?]
        T = nufft_T(N, J, K, alpha, beta)
        ###############################################################
        # formula 30  of Fessler's paper
        ###############################################################

        (r, arg) = nufft_r(om[:, dimid], N, J,
                           K, alpha, beta)  # large N approx [J? M]

        ###############################################################
        # formula 25  of Fessler's paper
        ###############################################################
        c = numpy.dot(T, r)

        ###############################################################
        # grid intervals in radius
        ###############################################################
        gam = 2.0 * numpy.pi / (K * 1.0)

        phase_scale = 1.0j * gam * (N - 1.0) / 2.0
        phase = numpy.exp(phase_scale * arg)  # [J? M] linear phase
        ud[dimid] = phase * c
        # indices into oversampled FFT components
        # FORMULA 7
        koff = nufft_offset(om[:, dimid], J, K)
        # FORMULA 9, find the indexes on Kd grids, of each M point
        kd[dimid] = numpy.mod(
            outer_sum(
                numpy.arange(
                    1,
                    J + 1) * 1.0,
                koff),
            K)

        """
            JML: For GPU computing, indexing must be C-order (row-major)
            Multi-dimensional cuda or opencl arrays are row-major (order="C"), which  starts from the higher dimension.
            Note: This is different from the MATLAB indexing(for fortran order, colum major, low-dimension first 
        """

        if dimid < dd - 1:  # trick: pre-convert these indices into offsets!
            #            ('trick: pre-convert these indices into offsets!')
            kd[dimid] = kd[dimid] * numpy.prod(Kd[dimid+1:dd]) - 1 
        """
        Note: F-order matrices must be reshaped into an 1D array before sparse matrix-vector multiplication.
        The original F-order (in Fessler and Sutton 2003) is not suitable for GPU array (C-order).
        Currently, in-place reshaping in F-order only works in numpy.
        
        """
#             if dimid > 0:  # trick: pre-convert these indices into offsets!
#                 #            ('trick: pre-convert these indices into offsets!')
#                 kd[dimid] = kd[dimid] * numpy.prod(Kd[0:dimid]) - 1           

    kk = kd[0]  # [J1 M] # pointers to indices
    uu = ud[0]  # [J1 M]
    Jprod = Jd[0]
    Kprod = Kd[0]
    for dimid in range(1, dd):
        Jprod = numpy.prod(Jd[:dimid + 1])
        Kprod = numpy.prod(Kd[:dimid + 1])
        kk = block_outer_sum(kk, kd[dimid]) + 1  # outer sum of indices
        kk = kk.reshape((Jprod, M), order='C')
        # outer product of coefficients
        uu = block_outer_prod(uu, ud[dimid])
        uu = uu.reshape((Jprod, M), order='C')
        # now kk and uu are [*Jd M]
        # now kk and uu are [*Jd M]
    # *numpy.tile(phase,[numpy.prod(Jd),1]) #    product(Jd)xM
    uu = uu.conj()
    mm = numpy.arange(0, M)  # indices from 0 to M-1
    mm = numpy.tile(mm, [numpy.prod(Jd).astype(int), 1])  # product(Jd)xM 
    # Now build sparse matrix from uu, mm, kk

    # convert array to list
    csrdata = numpy.reshape(uu.T, (Jprod * M, ), order='C')

    # row indices, from 1 to M convert array to list
    rowindx = numpy.reshape(mm.T, (Jprod * M, ), order='C')

    # colume indices, from 1 to prod(Kd), convert array to list
    colindx = numpy.reshape(kk.T, (Jprod * M, ), order='C')

    # The shape of sparse matrix
    csrshape = (M, numpy.prod(Kd))

    # Build sparse matrix (interpolator)
    st['p0'] = scipy.sparse.csr_matrix((csrdata, (rowindx, colindx)),
                                       shape=csrshape)
    # Note: the sparse matrix requires the following linear phase,
    #       which moves the image to the center of the image
    st['p0'].prune() # Scipy sparse: removing empty space after all non-zero elements.
    
    return st
def preindex_copy(Nd, Kd):
    """
    Building the array index for copying two arrays of sizes Nd and Kd
    
    Only the front part of the input/output arrays are copied. 
    
    The oversize  parts of the input array are truncated (if Nd > Kd). 
    
    And the smaller size are zero-padded (if Nd < Kd)
    
    :param Nd: tuple, the dimensions of array1
    :param Kd: tuple, the dimensions of array2
    :type Nd: tuple with integer elements
    :type Kd: tuple with integer elements
    :returns: inlist: the index of the input array
    :returns: outlist: the index of the output array
    :returns: nelem: the length of the inlist and outlist (equal length)
    :rtype: inlist: list with integer elements
    :rtype: outlist: list with integer elements
    :rtype: nelem: int
    """
    ndim = len(Nd)
    kdim = len(Kd)
    if ndim != kdim:
        print("mismatched dimensions!")
        print("Nd and Kd must have the same dimensions")
        raise
    else:
        nelem = 1
        min_dim = ()
        for pp in range(ndim - 1, -1,-1):
            YY = numpy.minimum(Nd[pp], Kd[pp])
            nelem *= YY
            min_dim = (YY,) + min_dim 
        mylist = numpy.arange(0, nelem).astype(numpy.int32)
#             a=mylist
        BB=()
        for pp in range(ndim - 1, 0, -1):
             a = numpy.floor(mylist/min_dim[pp])
             b = mylist%min_dim[pp]
             mylist = a
             BB=(b,) + BB
            
        
        inlist = mylist
        outlist = mylist
        for pp in range(0, ndim-1):
            inlist = inlist*Nd[pp+1] + BB[pp]
            outlist = outlist*Kd[pp+1] + BB[pp]

    return inlist.astype(numpy.int32), outlist.astype(numpy.int32), nelem.astype(numpy.int32)

def dirichlet(x):
    return numpy.sinc(x)

def outer_sum(xx, yy):
    """
    Superseded by numpy.add.outer() function
    """
    
    return numpy.add.outer(xx,yy)
#     nx = numpy.size(xx)
#     ny = numpy.size(yy)
# 
#     arg1 = numpy.tile(xx, (ny, 1)).T
#     arg2 = numpy.tile(yy, (nx, 1))
#     return arg1 + arg2


def nufft_offset(om, J, K):
    '''
    For every om points(outside regular grids), find the nearest
    central grid (from Kd dimension)
    '''
    gam = 2.0 * numpy.pi / (K * 1.0)
    k0 = numpy.floor(1.0 * om / gam - 1.0 * J / 2.0)  # new way
    return k0


def nufft_alpha_kb_fit(N, J, K):
    """
    Find parameters alpha and beta for scaling factor st['sn']
    The alpha is hardwired as [1,0,0...] when J = 1 (uniform scaling factor)
    
    :param int N: the size of image
    :param int J:the size of interpolator
    :param int K: the size of oversampled k-space
    
    
    
    """
    beta = 1
    Nmid = (N - 1.0) / 2.0
    if N > 40:
        L = 13
    else:
        L = numpy.ceil(N / 3).astype(numpy.int16)

    nlist = numpy.arange(0, N) * 1.0 - Nmid
    (kb_a, kb_m) = kaiser_bessel('string', J, 'best', 0, K / N)
    if J > 1:
        sn_kaiser = 1 / kaiser_bessel_ft(nlist / K, J, kb_a, kb_m, 1.0)
    elif J == 1:  # The case when samples are on regular grids
        sn_kaiser = numpy.ones((1, N), dtype=dtype)
    gam = 2 * numpy.pi / K
    X_ant = beta * gam * nlist.reshape((N, 1), order='F')
    X_post = numpy.arange(0, L + 1)
    X_post = X_post.reshape((1, L + 1), order='F')
    X = numpy.dot(X_ant, X_post)  # [N,L]
    X = numpy.cos(X)
    sn_kaiser = sn_kaiser.reshape((N, 1), order='F').conj()
    X = numpy.array(X, dtype=dtype)
    sn_kaiser = numpy.array(sn_kaiser, dtype=dtype)
    coef = numpy.linalg.lstsq(numpy.nan_to_num(X), numpy.nan_to_num(sn_kaiser))[0]
    alphas = coef
    if J > 1:
        alphas[0] = alphas[0]
        alphas[1:] = alphas[1:] / 2.0
    elif J == 1:  # cases on grids
        alphas[0] = 1.0
        alphas[1:] = 0.0
    alphas = numpy.real(alphas)
    return (alphas, beta)


def kaiser_bessel(x, J, alpha, kb_m, K_N):
    if K_N != 2:
        kb_m = 0
        alpha = 2.34 * J
    else:
        kb_m = 0

        # Parameters in Fessler's code
        # because it was experimentally determined to be the best!
        # input: number of interpolation points
        # output: Kaiser_bessel parameter

        jlist_bestzn = {2: 2.5,
                        3: 2.27,
                        4: 2.31,
                        5: 2.34,
                        6: 2.32,
                        7: 2.32,
                        8: 2.35,
                        9: 2.34,
                        10: 2.34,
                        11: 2.35,
                        12: 2.34,
                        13: 2.35,
                        14: 2.35,
                        15: 2.35,
                        16: 2.33}

        if J in jlist_bestzn:
            alpha = J * jlist_bestzn[J]
        else:
            tmp_key = (jlist_bestzn.keys())
            min_ind = numpy.argmin(abs(tmp_key - J * numpy.ones(len(tmp_key))))
            p_J = tmp_key[min_ind]
            alpha = J * jlist_bestzn[p_J]
    kb_a = alpha
    return (kb_a, kb_m)


def kaiser_bessel_ft(u, J, alpha, kb_m, d):
    '''
    Interpolation weight for given J/alpha/kb-m
    '''

    u = u * (1.0 + 0.0j)
    import scipy.special
    z = numpy.sqrt((2 * numpy.pi * (J / 2) * u) ** 2.0 - alpha ** 2.0)
    nu = d / 2 + kb_m
    y = ((2 * numpy.pi) ** (d / 2)) * ((J / 2) ** d) * (alpha ** kb_m) / \
        scipy.special.iv(kb_m, alpha) * scipy.special.jv(nu, z) / (z ** nu)
    y = numpy.real(y)
    return y


def nufft_scale1(N, K, alpha, beta, Nmid):
    '''
    Calculate image space scaling factor
    '''
#     import types
#     if alpha is types.ComplexType:
    alpha = numpy.real(alpha)
#         print('complex alpha may not work, but I just let it as')

    L = len(alpha) - 1
    if L > 0:
        sn = numpy.zeros((N, 1))
        n = numpy.arange(0, N).reshape((N, 1), order='F')
        i_gam_n_n0 = 1j * (2 * numpy.pi / K) * (n - Nmid) * beta
        for l1 in range(-L, L + 1):
            alf = alpha[abs(l1)]
            if l1 < 0:
                alf = numpy.conj(alf)
            sn = sn + alf * numpy.exp(i_gam_n_n0 * l1)
    else:
        sn = numpy.dot(alpha, numpy.ones((N, 1)))
    return sn


def nufft_scale(Nd, Kd, alpha, beta):
    dd = numpy.size(Nd)
    Nmid = (Nd - 1) / 2.0
    if dd == 1:
        sn = nufft_scale1(Nd, Kd, alpha, beta, Nmid)
    else:
        sn = 1
        for dimid in numpy.arange(0, dd):
            tmp = nufft_scale1(Nd[dimid], Kd[dimid], alpha[dimid],
                               beta[dimid], Nmid[dimid])
            sn = numpy.dot(list(sn), tmp.H)
    return sn


def mat_inv(A):
#     I = numpy.eye(A.shape[0], A.shape[1])
    B = scipy.linalg.pinv2(A)
    return B


def nufft_T(N, J, K, alpha, beta):
    '''
     The Equation (29) and (26) in Fessler and Sutton 2003.
     Create the overlapping matrix CSSC (diagonal dominent matrix)
     of J points and find out the pseudo-inverse of CSSC '''

#     import scipy.linalg
    L = numpy.size(alpha) - 1
#     print('L = ', L, 'J = ',J, 'a b', alpha,beta )
    cssc = numpy.zeros((J, J))
    [j1, j2] = numpy.mgrid[1:J + 1, 1:J + 1]
    overlapping_mat = j2 - j1
    for l1 in range(-L, L + 1):
        for l2 in range(-L, L + 1):
            alf1 = alpha[abs(l1)]
#             if l1 < 0: alf1 = numpy.conj(alf1)
            alf2 = alpha[abs(l2)]
#             if l2 < 0: alf2 = numpy.conj(alf2)
            tmp = overlapping_mat + beta * (l1 - l2)

            tmp = dirichlet(1.0 * tmp / (1.0 * K / N))
            cssc = cssc + alf1 * alf2 * tmp
       
    return mat_inv(cssc)


def nufft_r(om, N, J, K, alpha, beta):
    '''
    equation (30) of Fessler's paper

    '''
    def iterate_sum(rr, alf, r1):
        rr = rr + alf * r1
        return rr
    def iterate_l1(L, alpha, arg, beta, K, N, rr):
        oversample_ratio = (1.0 * K / N)
        import time
        t0=time.time()
        for l1 in range(-L, L + 1):
            alf = alpha[abs(l1)] * 1.0
    #         if l1 < 0:
    #             alf = numpy.conj(alf)
        #             r1 = numpy.sinc(1.0*(arg+1.0*l1*beta)/(1.0*K/N))
            input_array = (arg + 1.0 * l1 * beta) / oversample_ratio
            r1 = dirichlet(input_array)
            rr = iterate_sum(rr, alf, r1)
        return rr
    
    M = numpy.size(om)  # 1D size
    gam = 2.0 * numpy.pi / (K * 1.0)
    nufft_offset0 = nufft_offset(om, J, K)  # om/gam -  nufft_offset , [M,1]
    dk = 1.0 * om / gam - nufft_offset0  # om/gam -  nufft_offset , [M,1]
    arg = outer_sum(-numpy.arange(1, J + 1) * 1.0, dk)
    L = numpy.size(alpha) - 1
#     print('alpha',alpha)
    rr = numpy.zeros((J, M), dtype=numpy.float32)
    rr = iterate_l1(L, alpha, arg, beta, K, N, rr)
    return (rr, arg)


def block_outer_prod(x1, x2):
    '''
    Multiply x1 (J1 x M) and x2 (J2xM) and extend the dimension to 3D (J1xJ2xM)
    '''
    (J1, M) = x1.shape
    (J2, M) = x2.shape
#    print(J1,J2,M)
    xx1 = x1.reshape((J1, 1, M), order='F')  # [J1 1 M] from [J1 M]
    xx1 = numpy.tile(xx1, (1, J2, 1))  # [J1 J2 M], emulating ndgrid
    xx2 = x2.reshape((1, J2, M), order='F')  # [1 J2 M] from [J2 M]
    xx2 = numpy.tile(xx2, (J1, 1, 1))  # [J1 J2 M], emulating ndgrid

    y = xx1 * xx2

    return y  # [J1 J2 M]


def block_outer_sum(x1, x2):
    '''
    Update the new index after adding a new axis
    '''
    (J1, M) = x1.shape
    (J2, M) = x2.shape
    xx1 = x1.reshape((J1, 1, M), order='F')  # [J1 1 M] from [J1 M]
    xx1 = numpy.tile(xx1, (1, J2, 1))  # [J1 J2 M], emulating ndgrid
    xx2 = x2.reshape((1, J2, M), order='F')  # [1 J2 M] from [J2 M]
    xx2 = numpy.tile(xx2, (J1, 1, 1))  # [J1 J2 M], emulating ndgrid
    y = xx1 + xx2
    return y  # [J1 J2 M]


def crop_slice_ind(Nd):
    '''
    (Deprecated in v.0.3.4) 
    Return the "slice" of Nd size to index multi-dimensional array.  "Slice" functions as the index of the array.
    Superseded by preindex_copy() which avoid run-time indexing.    
    '''
    return [slice(0, Nd[ss]) for ss in range(0, len(Nd))]
def diagnose():
    """
    Diagnosis function
    Find available device when NUFFT.offload() failed
    """
    from reikna import cluda
    import reikna.transformations
    from reikna.cluda import functions, dtypes    
    try:
        api = cluda.cuda_api()
        print('cuda interface is available')
        available_cuda_device = cluda.find_devices(api)
        print(api, available_cuda_device)
        cuda_flag = 1
        print("try to load cuda interface:")
        for api_n in available_cuda_device.keys():
            print("API='cuda',  ", "platform_number=", api_n, ", device_number=", available_cuda_device[api_n][0])     
    except:
        cuda_flag = 0
        print('cuda interface is not available') 
    try:
        api = cluda.ocl_api()
        print('ocl interface is available')
        available_ocl_device = cluda.find_devices(api)
        print(api, available_ocl_device)
        ocl_flag = 1
        print("try to load ocl interface with:")
        for api_n in available_ocl_device.keys():
            print("API='ocl',  ", "platform_number=", api_n, ", device_number=", available_ocl_device[api_n][0])        
    except:
        print('ocl interface is not available')
        ocl_flag = 0
        
