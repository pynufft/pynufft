"""
Helper functions
=======================================

bugfix: mm = numpy.tile(mm, [numpy.prod(Jd).astype(int), 1])  to fix wrong type when numpy.prod(Jd) is not casted as int
bugfix: fix rcond=None error in Anaconda 3.6.5 and Numpy 1.13.1 (the recommended None in Numpy 1.14 is backward incompatible with 1.13)
bugfix:  indx1 = indx.copy() was replaced by indx1 = list(indx) for Python2 compatibility
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

    ################################
    #    Multi-dimensional laplacian kernel (generalize the above 1D - 3D to multi-dimensional arrays)
    ################################
    
    indx = [slice(0, 1) for ss in range(0, n_dims)] # create the n_dims dimensional slice which are all zeros
    uker[indx] = - 2.0*n_dims # Equivalent to  uker[0,0,0] = -6.0
    for pp in range(0,n_dims):
#         indx1 = indx.copy() # indexing the 1 Only for Python3
        indx1 = list(indx)# indexing; adding list() for Python2/3 compatibility
        indx1[pp] = 1
        uker[indx1] = 1
#         indx1 = indx.copy() # indexing the -1  Only for Python3
        indx1 = list(indx)# indexing the 1 Python2/3 compatible
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

def QR_process(om, N, J, K, sn):

    M = numpy.size(om)  # 1D size
    gam = 2.0 * numpy.pi / (K * 1.0)
    nufft_offset0 = nufft_offset(om, J, K)  # om/gam -  nufft_offset , [M,1]
    dk = 1.0 * om / gam - nufft_offset0  # om/gam -  nufft_offset , [M,1]
    
    
    arg = outer_sum(-numpy.arange(1, J + 1) * 1.0, dk) #[J, M]
    C_0 =numpy.outer(numpy.arange(0, N) - N/2,  numpy.arange(1, J + 1))

    C = numpy.exp(1.0j * gam*C_0)
    

    sn2 = numpy.reshape(sn, (N, 1))
    C = C*sn2
    bn =numpy.exp(1.0j*gam* numpy.outer(numpy.arange(0, N) - N/2, dk))

    return C, arg, bn

def solve_c(C, bn):
    CH = C.T.conj()
    C = CH.dot(C)
    bn = CH.dot(bn)

    C = numpy.linalg.pinv(C)
    c = C.dot(bn)
    return c    

def QR2(om, N, J, K, sn, ft_flag):
    C, arg, bn = QR_process(om, N, J, K, sn)
    c = solve_c(C, bn)
    u2 = OMEGA_u(c, N, K, om, arg, ft_flag).T.conj()
    
    return u2



def get_sn(J, K, N):
    
    Nmid = (N - 1.0) / 2.0
    nlist = numpy.arange(0, N) * 1.0 - Nmid    
    (kb_a, kb_m) = kaiser_bessel('string', J, 'best', 0, K / N)
    if J > 1:
        sn = 1.0/ kaiser_bessel_ft(nlist / K, J, kb_a, kb_m, 1.0)
    #                 sn = 1.0 / kaiser_bessel_ft(nlist / K, J, kb_a, kb_m, 1.0)
    elif J == 1:  # The case when samples are on regular grids
        sn = numpy.ones((1, N), dtype=dtype)
    return sn       

def OMEGA_u(c, N, K, omd, arg, ft_flag):
    
    gam = 2.0 * numpy.pi / (K * 1.0)
    
    phase_scale =  1.0j * gam * (N*1.0 - 1.0) / 2.0
    phase = numpy.exp(phase_scale * arg)  # [J? M] linear phase
    
    if ft_flag is True:
        u = phase * c
        phase0= numpy.exp( - 1.0j*omd*N/2.0)
        u = phase0 * u
    
    else:
        u = c
    return u
    
def OMEGA_k(J,K, omd, Kd, dimid, dd, ft_flag):
    """
    Compute the index of k-space k_indx
    """
        # indices into oversampled FFT components
    # FORMULA 7
    M = omd.shape[0]
    koff = nufft_offset(omd, J, K)
    # FORMULA 9, find the indexes on Kd grids, of each M point
    if ft_flag is True: # tensor
        k_indx = numpy.mod(
            outer_sum(
                numpy.arange(
                    1,
                    J + 1) * 1.0,
                koff),
            K)        
    else:
        k_indx =  numpy.reshape(omd, (1, M)).astype(numpy.int)
    #         k_indx = numpy.mod(
    #             outer_sum(
    #                 numpy.arange(
    #                     1,
    #                     J + 1) * 1.0,
    #                 koff),
    #             K)
    
    """
        JML: For GPU computing, indexing must be C-order (row-major)
        Multi-dimensional cuda or opencl arrays are row-major (order="C"), which  starts from the higher dimension.
        Note: This is different from the MATLAB indexing(for fortran order, colum major, low-dimension first 
    """
    
    if dimid < dd - 1:  # trick: pre-convert these indices into offsets!
        #            ('trick: pre-convert these indices into offsets!')
        k_indx = k_indx * numpy.prod(Kd[dimid+1:dd]) - 1 
    """
    Note: F-order matrices must be reshaped into an 1D array before sparse matrix-vector multiplication.
    The original F-order (in Fessler and Sutton 2003) is not suitable for GPU array (C-order).
    Currently, in-place reshaping in F-order only works in numpy.
    
    """
    #             if dimid > 0:  # trick: pre-convert these indices into offsets!
    #                 #            ('trick: pre-convert these indices into offsets!')
    #                 kd[dimid] = kd[dimid] * numpy.prod(Kd[0:dimid]) - 1           
    return k_indx

class pELL:
    def __init__(self, M,  Jd, curr_sumJd, meshindex, kindx, udata):
        
        self.nRow = M
        self.prodJd = numpy.prod(Jd)
        self.dim = len(Jd)
        self.sumJd = numpy.sum(Jd)
        self.Jd =  numpy.array(Jd).astype(numpy.uint32)
        self.curr_sumJd = curr_sumJd
        self.meshindex = numpy.array(meshindex, order='C')
        self.kindx = numpy.array(kindx, order='C')
        self.udata = numpy.array(udata, order='C')
        
#         print('self.kindx', self.kindx.shape)
#         print('self.udata',self.udata.shape)
#         print('self.meshindex', self.meshindex.shape)
#         print('meshindex = ', meshindex)
#         print('curr_sumJd', self.curr_sumJd)
#         print('prodJd', self.prodJd)
#         print('nRow', self.nRow)
#         if 0 == 1:
#             y = 0+0j
#             myRow = 0 
#     #         tmp_sumJd = 0
#     #         col = 0
#             
#             
#             csrdata=numpy.empty((self.prodJd, ),dtype = numpy.complex64)
#             csrindeces=numpy.empty((self.prodJd, ),dtype = numpy.uint32)
#             
#             for j in range(0, self.prodJd):
#                 tmp_sumJd = 0
#                 J = Jd[0]
#                 index = myRow * self.sumJd +   tmp_sumJd +  self.meshindex.ravel()[j*self.dim +  0]  
#     #             print(0 , j, index)   
#                 col = self.kindx.ravel()[ index] 
#                 spdata =self.udata.ravel()[index]
#                 tmp_sumJd += J
#                 
#     #             if self.dim > 1:
#                 for dimid in range(1, self.dim):
#                     J = Jd[dimid]
#                     index =   myRow * self.sumJd + tmp_sumJd + self.meshindex.ravel()[j* self.dim + dimid] 
#     #                     print(dimid, j, index)   
#                     col += self.kindx.ravel()[ index] + 1
#                     spdata *= self.udata.ravel()[index]
#                     tmp_sumJd  += J;
#                 csrdata[j] = spdata      
#                 csrindeces[j] = col
#         print('csrdata2 = ', csrdata)
#         print('csrindeces2 = ', csrindeces)
#         print('kindx[0, :]', kindx[0,:])
#             tmp_sumJd = tmp_sumJd + J;             
        
#         
#         
#         for pp in range(0, self.prodJd):
            
        
    
def create_csr(uu, kk, Kd, Jd, M):
#     Jprod = numpy.prod(Jd)
#     mm = numpy.arange(0, M).reshape( (1, M), order='C')  # indices from 0 to M-1
#     mm_rep = numpy.ones(( Jprod, 1))
#     mm = mm * mm_rep
    #     Jprod = numpy.prod(Jd)
    csrdata =uu.ravel(order='C')#numpy.reshape(uu.T, (Jprod * M, ), order='C')
    
    # row indices, from 1 to M convert array to list
#     rowindx = mm.ravel(order='C') #numpy.reshape(mm.T, (Jprod * M, ), order='C')
    
    Jdprod = numpy.prod(Jd)
    rowptr = numpy.arange(0, (M+1)*Jdprod, Jdprod)
    # colume indices, from 1 to prod(Kd), convert array to list
    colindx =kk.ravel(order='C')#numpy.reshape(kk.T, (Jprod * M, ), order='C')

    # The shape of sparse matrix
    csrshape = (M, numpy.prod(Kd))

    # Build sparse matrix (interpolator)
#     csr = scipy.sparse.csr_matrix((csrdata, (rowindx, colindx)),
#                                        shape=csrshape)
    csr = scipy.sparse.csr_matrix((csrdata, colindx, rowptr),
                                       shape=csrshape)
#     csr.has_sorted_indices = False
#     csr.sort_indices() # sort the indices in-place
    return csr
class ELL:
    """
    ELL is slow on a single core CPU
    """
    def __init__(self, elldata, ellcol):
        
#         self.shape = shape
        self.colWidth = ellcol.shape[1]
        self.nRow = ellcol.shape[0]
        self.data = elldata.reshape((self.nRow, self.colWidth),order='C')
        self.col = ellcol.astype(numpy.int32).reshape((self.nRow, self.colWidth),order='C')
    def spmv(self, x):
        y = numpy.einsum('ij,ij->i', self.data, x[self.col])
#         y = self.data * x[self.col]
#         y = numpy.sum(y, 1)
        return y 
         
    def spmvH(self, y):
        x = numpy.zeros(self.shape[1],  dtype = numpy.complex64)
        x[self.col.ravel()] += numpy.einsum('ij,i->ij', self.data.conj(), y).ravel()
        return x
def create_ell(uu, kk):
#     Jprod = numpy.prod(Jd)

    # The shape of sparse matrix
#     ellshape = (M, numpy.prod(Kd))

    # Build sparse matrix (interpolator)
#     csr = scipy.sparse.csr_matrix((csrdata, (rowindx, colindx)),
#                                        shape=csrshape)
    ell = ELL(uu, kk)
#     csr.has_sorted_indices = False
#     csr.sort_indices() # sort the indices in-place
    return ell
def create_partialELL(ud, kd, Jd, M):
    """
    Input:
    ud (the struct of all 1D interpolators), kd (the struct of all 1D indeces of 1D interpolators), 
    Jd: tuple of interpolation sizes
    dd: the number of dimensions
    M: the number of samples
    
    output:
    M: number of non-uniform locations 
    Jd: tuple,  (Jd[0], Jd[1], Jd[2],    ...,    Jd[dd -1]
    curr_sumJd: summation of curr_sumJd[dimid] = numpy.sum(Jd[0:dimid - 1])
    meshindex: For prodJd hypercubic interpolators, find the indices of tensor, shape = (prodJd, dd)
    kindx: column indeces, shape = (M, sumJd)
    udata: interpolators, shape = (M, sumJd)
    
    """
    dd = len(Jd)
    curr_sumJd = numpy.zeros( ( dd, ), dtype = numpy.uint32)
    kindx = numpy.zeros( ( M, numpy.sum(Jd)), dtype = numpy.uint32)
    udata = numpy.zeros( ( M, numpy.sum(Jd)), dtype = dtype)
    
    meshindex = numpy.zeros(  (numpy.prod(Jd),  dd), dtype = numpy.uint32)
    
    tmp_curr_sumJd = 0
    for dimid in range(0, dd):
        J = Jd[dimid]
        curr_sumJd[dimid] = tmp_curr_sumJd
        tmp_curr_sumJd +=int( J) # for next loop 
        kindx[:, int(curr_sumJd[dimid] ): int(curr_sumJd[dimid]  + J)] = numpy.array(kd[dimid], order='C')
        udata[:, int(curr_sumJd[dimid] ): int(curr_sumJd[dimid]  + J)] = numpy.array(ud[dimid], order='C')

    series_prodJd = numpy.arange(0, numpy.prod(Jd))
    
    for dimid in range(dd-1, -1, -1):  # iterate over all dimensions
        
        J = Jd[dimid]
        xx = series_prodJd % J
        yy = numpy.floor(series_prodJd/ J)
        series_prodJd =  yy
        meshindex[:, dimid] = xx.astype(numpy.uint32)
#         else: 
#             meshindex[:, dimid] = yy.astype(numpy.uint32)
    partialELL = pELL(M, Jd, curr_sumJd, meshindex, kindx, udata)
    return partialELL

def partial_combination(ud, kd, Jd):
    """
    Input:
    ud (the struct of all 1D interpolators), kd (the struct of all 1D indeces of 1D interpolators), 
    Jd: tuple of interpolation sizes
    dd: the number of dimensions
    M: the number of samples
    
    output:
    M: number of non-uniform locations 
    Jd: tuple,  (Jd[0], Jd[1], Jd[2],    ...,    Jd[dd -1]
    curr_sumJd: summation of curr_sumJd[dimid] = numpy.sum(Jd[0:dimid - 1])
    meshindex: For prodJd hypercubic interpolators, find the indices of tensor, shape = (prodJd, dd)
    kindx: column indeces, shape = (M, sumJd)
    udata: interpolators, shape = (M, sumJd)
    
    """

    dd = len(Jd)
    
    kk = kd[0]  # [M, J1] # pointers to indices
    M = kd[0].shape[0]
    uu = ud[0]  # [M, J1]
    Jprod = Jd[0]
    for dimid in range(1, dd):
        Jprod *= Jd[dimid]#numpy.prod(Jd[:dimid + 1])

        kk = block_outer_sum(kk, kd[dimid]) + 1  # outer sum of indices
        kk = kk.reshape((M, Jprod), order='C')
        uu = numpy.einsum('ij,ik->ijk', uu, ud[dimid])
        uu = uu.reshape((M, Jprod), order='C')
    kd2 = (kk, )
    ud2 = (uu, )
    Jd2 = (Jprod, )
    return ud2, kd2, Jd2

def full_kron(ud, kd, Jd, Kd, M):
#     (udata, kindx)=khatri_rao(ud, kd, Jd)
    udata = khatri_rao_u(ud)
    kindx = khatri_rao_k(kd)
    CSR  = create_csr(udata, kindx, Kd, Jd, M) # must have 
    # Dimension reduction: Nd -> 1 
    # Tuple (Nd) -> array (shape = M*prodJd)
    
#     Note: the shape of uu and kk is (M, prodJd)
    ELL = create_ell(   udata,  kindx)#, Kd, Jd, M)    
    return CSR, ELL
def khatri_rao_k(kd):
    dd = len(kd)
    
    kk = kd[0]  # [M, J1] # pointers to indices
    M = kd[0].shape[0]
#     uu = ud[0]  # [M, J1]
    Jprod = kd[0].shape[1]
    for dimid in range(1, dd):
        Jprod *= kd[dimid].shape[1] #numpy.prod(Jd[:dimid + 1])

        kk = block_outer_sum(kk, kd[dimid]) + 1  # outer sum of indices
        kk = kk.reshape((M, Jprod), order='C')
#         uu = numpy.einsum('mi,mj->mij', uu, ud[dimid])
#         uu = uu.reshape((M, Jprod), order='C')
     
    return kk
def khatri_rao_u( ud):
    dd = len(ud)
    
#     kk = kd[0]  # [M, J1] # pointers to indices
    M = ud[0].shape[0]
    uu = ud[0]  # [M, J1]
    Jprod = ud[0].shape[1]
    for dimid in range(1, dd):
        Jprod *=ud[dimid].shape[1]#numpy.prod(Jd[:dimid + 1])

#         kk = block_outer_sum(kk, kd[dimid]) + 1  # outer sum of indices
#         kk = kk.reshape((M, Jprod), order='C')
        uu = numpy.einsum('mi,mj->mij', uu, ud[dimid])
        uu = uu.reshape((M, Jprod), order='C')
     
    return uu
# def khatri_rao(ud, kd, Jd):
#     dd = len(Jd)
#     
#     kk = kd[0]  # [M, J1] # pointers to indices
#     M = kd[0].shape[0]
#     uu = ud[0]  # [M, J1]
#     Jprod = Jd[0]
#     for dimid in range(1, dd):
#         Jprod *= Jd[dimid]#numpy.prod(Jd[:dimid + 1])
# 
#         kk = block_outer_sum(kk, kd[dimid]) + 1  # outer sum of indices
#         kk = kk.reshape((M, Jprod), order='C')
#         uu = numpy.einsum('mi,mj->mij', uu, ud[dimid])
#         uu = uu.reshape((M, Jprod), order='C')
#      
#     return uu, kk
def rdx_kron(ud, kd, Jd, M):
    """
    Radix-n Kronecker product of multi-dimensional array
    """
    dd = len(Jd)
    kk = kd[0]  # [J1 M] # pointers to indices
    uu = ud[0]  # [J1 M]
    Jprod = Jd[0]
    for dimid in range(1, dd):
        Jprod *= Jd[dimid]#numpy.prod(Jd[:dimid + 1])

        kk = block_outer_sum(kk, kd[dimid]) + 1  # outer sum of indices
        kk = kk.reshape((M, Jprod), order='C')
        uu = numpy.einsum('ij,ik->ijk', uu, ud[dimid])
        uu = uu.reshape((M, Jprod), order='C')
     
    return uu, kk#, Jprod
def Kronector_snd(snd, dd):
    """
    Compute the Kronecker product of scaling factor
    """
    shape_broadcasting = ()
    for dimid in range(0, dd):
        shape_broadcasting += (1, )
#     sn = numpy.array(1.0 + 0.0j)
    sn= numpy.reshape(1.0, shape_broadcasting)
    for dimid in range(0, dd):
        sn_shape = list(shape_broadcasting)
        sn_shape[dimid] = snd[dimid].shape[0]
        tmp = numpy.reshape(snd[dimid], tuple(sn_shape))
#         print('tmp.shape = ', tmp.shape)
        ###############################################################
        # higher dimension implementation: multiply over all dimension
        ###############################################################        
        sn = sn  * tmp # multiply using broadcasting    
    return sn

def min_max(N, J, K, alpha, beta, om, ft_flag):
    T = nufft_T(    N,  J,  K,  alpha,  beta)
    ###############################################################
    # formula 30  of Fessler's paper
    ###############################################################
    (r, arg) = nufft_r( om,   N,  J,
                   K,   alpha,  beta)  # large N approx [J? M]
    ###############################################################
    # Min-max interpolator
    ###############################################################
    c = T.dot(r)
    u2 = OMEGA_u(c, N, K, om, arg, ft_flag).T.conj()
    return u2

def plan(om, Nd, Kd, Jd, ft_axes = None):
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

    if ft_axes is None:
        ft_axes = tuple(xx for xx in range(0, dd))

#     print('ft_axes = ', ft_axes)
    ft_flag = () # tensor
    
    for pp in range(0, dd):
        if pp in ft_axes:
            ft_flag += (True, )
        else:
            ft_flag += (False, )
#     print('ft_flag = ', ft_flag)
###############################################################
# check input errors
###############################################################
    st = {}
    

###############################################################
# First, get alpha and beta: the weighting and freq
# of formula (28) in Fessler's paper
# in order to create slow-varying image space scaling
###############################################################
#     for dimid in range(0, dd):
#         (tmp_alpha, tmp_beta) = nufft_alpha_kb_fit(
#             Nd[dimid], Jd[dimid], Kd[dimid])
#         st.setdefault('alpha', []).append(tmp_alpha)
#         st.setdefault('beta', []).append(tmp_beta)
    st['tol'] = 0
    st['Jd'] = Jd
    st['Nd'] = Nd
    st['Kd'] = Kd
    M = om.shape[0]
    st['M'] = numpy.int32(M)
    st['om'] = om
    
###############################################################
# create scaling factors st['sn'] given alpha/beta
# higher dimension implementation
###############################################################

    """
    Now compute the 1D scaling factors
    snd: list
    """
    
    for dimid in range(0, dd):

        (tmp_alpha, tmp_beta) = nufft_alpha_kb_fit(
            Nd[dimid], Jd[dimid], Kd[dimid])
        st.setdefault('alpha', []).append(tmp_alpha)
        st.setdefault('beta', []).append(tmp_beta)
        
    snd = []
    for dimid in range(0, dd):        
        snd += [nufft_scale(
            Nd[dimid],
            Kd[dimid],
            st['alpha'][dimid],
            st['beta'][dimid]), ]
    """
     higher-order Kronecker product of all dimensions
    """      
    st['sn'] = Kronector_snd(snd, dd).real # only real scaling is relevant
    
    # [J? M] interpolation coefficient vectors.
    # Iterate over all dimensions and
    # multiply the coefficients of all dimensions

    ud = []
    for dimid in range(0, dd):  # iterate through all dimensions
        N = Nd[dimid]
        J = Jd[dimid]
        K = Kd[dimid]
        alpha = st['alpha'][dimid]
        beta = st['beta'][dimid]
       ###############################################################
        # formula 29 , 26 of Fessler's paper
        ###############################################################

        # pseudo-inverse of CSSC using large N approx [J? J?]
        if ft_flag[dimid] is True:

#         ###############################################################
#         # formula 30  of Fessler's paper
#         ###############################################################

#         ###############################################################
#         # fast approximation to min-max interpolator 
#         ###############################################################

#             c, arg = min_max(N, J, K, alpha, beta, om[:, dimid])
#         ###############################################################
#        # QR: a more accurate solution but slower than above fast approximation
#        ###############################################################
            
#             c, arg = QR_process(om[:,dimid], N, J, K, snd[dimid])

            #### phase shift
#             ud += [QR2(om[:,dimid], N, J, K, snd[dimid], ft_flag[dimid]),]
            ud += [min_max(N, J, K, alpha, beta, om[:, dimid], ft_flag[dimid]),]

        else:
            ud += [numpy.ones((1, M), dtype = dtype).T, ]


    """
    Now compute the column indeces for 1D interpolators
    Each length-Jd interpolator includes Jd points, which are linked to Jd k-space locations
    kd is a tuple storing the 1D interpolators. 
    A following Kronecker product will be needed.
    """
    kd = []
    for dimid in range(0, dd):  # iterate over all dimensions
        kd += [OMEGA_k(J,K, om[:,dimid], Kd, dimid, dd, ft_flag[dimid]).T, ]


    
    CSR, ELL = full_kron(ud, kd, Jd, Kd, M)
    st['p'] = CSR
#     st['ell'] = ELL

    ud2, kd2, Jd2 = partial_combination(ud, kd, Jd)
    
    st['pELL'] = create_partialELL(ud2, kd2, Jd2, M) 
    
    # no dimension-reduction Nd -> Nd
    # Tuple (Nd) -> array (shape = M*sumJd)

    return st #new

# def plan1(om, Nd, Kd, Jd, ft_axes = None):
#     """
#     Compute the coil sensitivity aware interpolator
#     """
# #         self.debug = 0  # debug
# 
#     if type(Nd) != tuple:
#         raise TypeError('Nd must be tuple, e.g. (256, 256)')
# 
#     if type(Kd) != tuple:
#         raise TypeError('Kd must be tuple, e.g. (512, 512)')
# 
#     if type(Jd) != tuple:
#         raise TypeError('Jd must be tuple, e.g. (6, 6)')
# 
#     if (len(Nd) != len(Kd)) | (len(Nd) != len(Jd))  | len(Kd) != len(Jd):
#         raise KeyError('Nd, Kd, Jd must be in the same length, e.g. Nd=(256,256),Kd=(512,512),Jd=(6,6)')
# 
#     dd = numpy.size(Nd)
# 
#     if ft_axes is None:
#         ft_axes = tuple(xx for xx in range(0, dd))
# 
# #     print('ft_axes = ', ft_axes)
#     ft_flag = () # tensor
#     
#     for pp in range(0, dd):
#         if pp in ft_axes:
#             ft_flag += (True, )
#         else:
#             ft_flag += (False, )
# 
#     st = {}
#     
# 
#     st['tol'] = 0
#     st['Jd'] = Jd
#     st['Nd'] = Nd
#     st['Kd'] = Kd
#     M = om.shape[0]
#     st['M'] = numpy.int32(M)
#     st['om'] = om
#     
#     
#     for dimid in range(0, dd):
# 
#         (tmp_alpha, tmp_beta) = nufft_alpha_kb_fit(
#             Nd[dimid], Jd[dimid], Kd[dimid])
#         st.setdefault('alpha', []).append(tmp_alpha)
#         st.setdefault('beta', []).append(tmp_beta)
#         
#     snd = ()
#     list_C = []
#     list_arg = []
#     list_bn = []
#     for dimid in range(0, dd):        
#         snd += (nufft_scale(
#             Nd[dimid],
#             Kd[dimid],
#             st['alpha'][dimid],
#             st['beta'][dimid]), )
#         C, arg, bn = QR_process(om[:, dimid], Nd[dimid], Jd[dimid], Kd[dimid], snd[dimid]) 
#         """
#         Save the C matrix, arg and bn. They will be reused.
#         """
#         list_C += [C, ]
#         list_arg += [arg, ]
#         list_bn += [bn, ]
#     st['sn'] = Kronector_snd(snd, dd).real # only real scaling is relevant
# 
#     # [J? M] interpolation coefficient vectors.
#     # Iterate over all dimensions and
#     # multiply the coefficients of all dimensions
# 
#     ud = []
#     for dimid in range(0, dd):  # iterate through all dimensions
#         N = Nd[dimid]
#         J = Jd[dimid]
#         K = Kd[dimid]
#         alpha = st['alpha'][dimid]
#         beta = st['beta'][dimid]
# 
#         """
#         Compute the coil sensitivity aware interpolator
#         """
# 
#         if ft_flag[dimid] is True:
# 
#             c = solve_c( list_C[dimid],  list_bn[dimid]) 
#             # C: NxJ, 
#             # bn: NxM (xr)
#             # c: J x M    (x    r)
# 
#             u2 = OMEGA_u( c, N, K, om[:, dimid], list_arg[dimid], ft_flag[dimid]).T.conj()
#             # u2: J x M    (x    r)
# 
#             ud += [u2,]
# 
#         else:
#             ud += (numpy.ones((1, M), dtype = dtype).T, )
# 
#     """
#     Now compute the column indeces for 1D interpolators
#     Each length-Jd interpolator includes Jd points, which are linked to Jd k-space locations
#     kd is a tuple storing the 1D interpolators. 
#     A following Kronecker product will be needed.
#     """
#     
#     kd = []
#     for dimid in range(0, dd):  # iterate over all dimensions
#         kd += (OMEGA_k(J,K, om[:,dimid], Kd, dimid, dd, ft_flag[dimid]).T, )
# 
#     CSR, ELL = full_kron(ud, kd, Jd, Kd, M)
#     st['p'] = CSR
# 
#     return st #new

def plan0(om, Nd, Kd, Jd):
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
        kk = block_outer_sum0(kk, kd[dimid]) + 1  # outer sum of indices
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
    st['p'] = scipy.sparse.csr_matrix((csrdata, (rowindx, colindx)),
                                       shape=csrshape)
    # Note: the sparse matrix requires the following linear phase,
    #       which moves the image to the center of the image
     
    om = st['om']
    M = st['M']
     
    n_shift = tuple(0*x for x in st['Nd'])
     
    final_shifts = tuple(
        numpy.array(n_shift) +
        numpy.array(
            st['Nd']) /
        2)
     
    phase = numpy.exp(
        1.0j *
        numpy.sum(
            om *
            numpy.tile(
                final_shifts,
                (M,
                 1)),
            1))
    # add-up all the linear phasees in all axes,
 
    st['p'] = scipy.sparse.diags(phase, 0).dot(st['p'])
 
     
#     st['p0'].prune() # Scipy sparse: removing empty space after all non-zero elements.
     
    return st # plan0()

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
    coef = numpy.linalg.lstsq(numpy.nan_to_num(X), numpy.nan_to_num(sn_kaiser), rcond = -1)[0]
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


def block_outer_sum0(x1, x2):
    '''
    Multiply x1 (J1 x M) and x2 (J2xM) and extend the dimension to 3D (J1xJ2xM)
    '''
    (J1, M) = x1.shape
    (J2, M) = x2.shape
#    print(J1,J2,M)
    xx1 = x1.reshape((J1, 1, M), order='C')  # [J1 1 M] from [J1 M]
#     xx1 = numpy.tile(xx1, (1, J2, 1))  # [J1 J2 M], emulating ndgrid
    xx2 = x2.reshape((1, J2, M), order='C')  # [1 J2 M] from [J2 M]
#     xx2 = numpy.tile(xx2, (J1, 1, 1))  # [J1 J2 M], emulating ndgrid
 
    y = xx1 + xx2
#     y = numpy.einsum('ik, jk->ijk', x1, x2)
    return y  # [J1 J2 M]

def block_outer_prod(x1, x2):
    '''
    Multiply x1 (J1 x M) and x2 (J2xM) and extend the dimension to 3D (J1xJ2xM)
    '''
    (J1, M) = x1.shape
    (J2, M) = x2.shape
#    print(J1,J2,M)
    xx1 = x1.reshape((J1, 1, M), order='C')  # [J1 1 M] from [J1 M]
#     xx1 = numpy.tile(xx1, (1, J2, 1))  # [J1 J2 M], emulating ndgrid
    xx2 = x2.reshape((1, J2, M), order='C')  # [1 J2 M] from [J2 M]
#     xx2 = numpy.tile(xx2, (J1, 1, 1))  # [J1 J2 M], emulating ndgrid
 
    y = xx1 * xx2
#     y = numpy.einsum('ik, jk->ijk', x1, x2)
    return y  # [J1 J2 M]


def block_outer_sum(x1, x2):
    '''
    Update the new index after adding a new axis
    '''
    (M, J1) = x1.shape
    (M, J2) = x2.shape
    xx1 = x1.reshape((M, J1, 1), order='C')  # [J1 1 M] from [J1 M]
#     xx1 = numpy.tile(xx1, (1, J2, 1))  # [J1 J2 M], emulating ndgrid
    xx2 = x2.reshape((M, 1, J2), order='C')  # [1 J2 M] from [J2 M]
#     xx2 = numpy.tile(xx2, (J1, 1, 1))  # [J1 J2 M], emulating ndgrid
#     print('xx1, xx2 shape = ', xx1.shape, xx2.shape)
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
        
