from .._helper.helper import *
import string
alpha_list = list(string.ascii_lowercase)
abc_str = ()
nop_str = ()
for qq in range(0, 12):
    abc_str += (alpha_list[qq],)
    nop_str += (alpha_list[qq+13],)

def create_csr2(uu, kk, Kd, Jd, M):

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
    
def full_kron2(ud, kd, Jd, Kd, M, core):
    udata = khatri_rao_u2(ud, core)
#     Nc = core.shape[-1]
    kindx = khatri_rao_k2(kd)
    CSR  = create_csr2(udata, kindx, Kd, Jd, M) # must have 
    # Dimension reduction: Nd -> 1 
    # Tuple (Nd) -> array (shape = M*prodJd)
    
#     Note: the shape of uu and kk is (M, prodJd)
#     ELL = create_ell(   udata,  kindx)#, Kd, Jd, M)    
    return CSR
def khatri_rao_k2(kd):
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
#     kk = numpy.tile(kk, (Nc, 1))
    return kk

def khatri_rao_u2(ud,  core):
    """
    ud[dimid]: has the shape of (M, J, r)
    M: The number of non-uniform locations
    J: The size of the interpolator
    r: rank
    H_core: core tensor, (r,r,r,..,Nc)
    """
    
    dd = len(ud)
    M = ud[0].shape[0]
#     Nc = core.shape[-1]
    core_shape = len(core.shape)
    core_str = ''
    for pp in range(0, dd):
        core_str += nop_str[pp]

    mstring = numpy.ones((M,))*1.0
    uu = numpy.einsum(core_str + ', m' + '-> m' + core_str, core, mstring)
#     print('uu.shape = ', uu.shape)

    outstr_core = []
    for dimid in range(0, dd):
        outstr_core += [nop_str[pp], ]
    
    Jprod = 1
    for dimid in range(0, dd):
        Jprod *= ud[dimid].shape[1]
        instr = 'm' + abc_str[dimid] + nop_str[dimid] + ', m'
        for pp in range(0, dd):
            instr += outstr_core[pp]
        outstr_core[dimid] = abc_str[dimid] # replace the dimid-th axis with abc_str[dimid
        out = 'm'
        for pp in range(0, dd):
            out += outstr_core[pp]
        uu = numpy.einsum(instr + '->' + out , ud[dimid], uu)

    uu = uu.reshape((M, Jprod), order='C')
     
    return uu

def solve_c2(C, L_bn):
    
    bn2 = numpy.einsum('nj, nmr -> jmr',C.conj(), L_bn)
    C2 = numpy.einsum('nj, nk -> jk ', C.conj(), C)
    C = numpy.linalg.pinv(C2)
    c = numpy.einsum('ij, jmr -> imr', C, bn2)    
    return c    

def QR_process2(om, N, J, K, sn):
    import time
    t0 = time.time()
    M = numpy.size(om)  # 1D size
    gam = 2.0 * numpy.pi / (K * 1.0) # scalar
    nufft_offset0 = nufft_offset(om, J, K)  # om/gam -  nufft_offset , [M,1]
    dk = 1.0 * om / gam - nufft_offset0  # om/gam -  nufft_offset , [M,1]
    phase_scale =  1.0j * gam * (N*1.0 - 1.0) / 2.0
    phase0 = numpy.exp( - 1.0j*om*N/2.0) # M
    
#     arg = outer_sum(-numpy.arange(1, J + 1) * 1.0, dk) #[J, M]
    exp_arg = numpy.einsum('j, m -> jm', numpy.exp(-numpy.arange(1, J + 1)*1.0), numpy.exp(dk))
    ph = exp_arg**phase_scale * phase0
    
    
    C_0 =numpy.einsum('n, j -> nj', numpy.arange(0, N) - N/2,  numpy.arange(1, J + 1))

    C = numpy.exp(1.0j * gam*C_0)
    

    sn2 = numpy.reshape(sn, (N, 1))
    C = C*sn2
    bn =numpy.exp(1.0j*gam* numpy.outer(numpy.arange(0, N) - N/2, dk))

#     ph = OMEGA_phase(N, K, om, arg2)
    print('time = ', time.time() - t0)
    return C, bn, ph

def OMEGA_phase(N, K, omd, arg2):
    gam = 2.0 * numpy.pi / (K * 1.0)
    phase_scale =  1.0j * gam * (N*1.0 - 1.0) / 2.0
    phase = arg2 * numpy.exp( - 1.0j*omd*N/2.0)
#     phase = numpy.exp(phase_scale * arg) * numpy.exp( - 1.0j*omd*N/2.0) # [J? M] linear phase
    
    return phase

def order_reduction(om, N, J, K, sn, L):
    """
    Compute the min-max interpolator consider the factor matrix
    L is the conjugate of the factor matrix of Tucker decomposition
    Note: only suitable for precalculated interpolator
    factor_matrix L: shape = (N, r), r is the rank
    
    """

    M = numpy.size(om)  # 1D size
    gam = 2.0 * numpy.pi / (K * 1.0)
    nufft_offset0 = nufft_offset(om, J, K)  # om/gam -  nufft_offset , [M,1]
    dk = 1.0 * om / gam - nufft_offset0  # om/gam -  nufft_offset , [M,1]
    
    
    arg = outer_sum(-numpy.arange(1, J + 1) * 1.0, dk) #[J, M]
    C_0 =numpy.outer(numpy.arange(0, N) - N/2,  numpy.arange(1, J + 1))

    C = numpy.exp(1.0j * gam*C_0)
    
    sn2 = numpy.reshape(sn, (N, 1))
    C = C*sn2
    CH = C.T.conj()
    C = CH.dot(C)
    
    bn =numpy.exp(1.0j*gam* numpy.outer(numpy.arange(0, N) - N/2, dk))
    '''
    bn: shape = (N, M), M: the number of non-Cartesian locations
    However, factor_matrix has a shape of (N, r)
    First reshape factor matrix to (N, 1, r) and reshape bn to (N, M, 1)
    '''
    r = L.shape[1]
    bn2 = numpy.reshape(bn, (N, M, 1), order='C')
    L2 = numpy.reshape(L, (N, 1, r), order='C')
    bn3 = bn2 * L2 # (N, M, r)
    bn3 = numpy.reshape(bn3, (N, M*r), order='C')
    
    '''
    Now compute L* bn
    '''
    
    bn3 = CH.dot(bn3)

#     t3=time.time()
    C = numpy.linalg.pinv(C)
    c = C.dot(bn3)
    c = numpy.reshape(c, (J, M, r),  order='C')
    u2 = numpy.empty_like(c)
    for pp in range(0, r): # r should be very small, which justifies a for-loop
        u2[:,:,pp] = OMEGA_u(c[:,:,pp], N, K, om, arg, True).T.conj()
    
    return u2

def plan1(om, Nd, Kd, Jd, ft_axes, coil_sense):
    """
    Compute the coil sensitivity aware interpolator
    """
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

    if (len(coil_sense.shape) != dd + 1):
        raise TypeError('Wrong dimension')

    if ft_axes is None:
        ft_axes = tuple(xx for xx in range(0, dd))

#     print('ft_axes = ', ft_axes)
    ft_flag = () # tensor
    
    for pp in range(0, dd):
        if pp in ft_axes:
            ft_flag += (True, )
        else:
            ft_flag += (False, )

    st = {}
    

    st['tol'] = 0
    st['Jd'] = Jd
    st['Nd'] = Nd
    st['Kd'] = Kd
    M = om.shape[0]
    st['M'] = numpy.int32(M)
    st['om'] = om
    
    
    for dimid in range(0, dd):

        (tmp_alpha, tmp_beta) = nufft_alpha_kb_fit(
            Nd[dimid], Jd[dimid], Kd[dimid])
        st.setdefault('alpha', []).append(tmp_alpha)
        st.setdefault('beta', []).append(tmp_beta)
        
    snd = {}
    list_C = {}
    list_arg = {}
    list_bn = {}
    list_ph = {}
    for dimid in range(0, dd):        
        snd[dimid] = nufft_scale(
            Nd[dimid],
            Kd[dimid],
            st['alpha'][dimid],
            st['beta'][dimid]) 
        C, bn, ph = QR_process2(om[:, dimid], Nd[dimid], Jd[dimid], Kd[dimid], snd[dimid]) 
        
        """
        Save the C matrix, arg and bn. They will be reused.
        """
        
        list_C[dimid]= C # Computes the short time inverse Fourier N x J matrix for dimension dimid
#         list_arg += [arg, ] # For each dimension, the arg is an J x M matrix
        list_ph[dimid] = ph # Computes the phase matrix (JxM) for each dimension
        list_bn[dimid] = bn # Computes the bn is an N x M matrix
        
    st['sn'] = Kronector_snd(snd, dd).real # only real scaling is relevant

    # [J? M] interpolation coefficient vectors.
    # Iterate over all dimensions and
    # multiply the coefficients of all dimensions
    from .Nd_tensor import htensor
    Htensor = htensor()
    r = 1
    Nc = coil_sense.shape[-1]
    rank = ()
    for dimid in range(0, dd):
        if ft_flag[dimid] is True:
            rank += (r, )
        else:
            rank += (r, )
            
#     rank += (Nc, )


#     indptr = numpy.zeros( (M + 1, Nc ) , order='C', dtype = numpy.int32)
    csrindices = numpy.empty((M, Nc, numpy.prod(Jd)), order='C', dtype = numpy.int32  )
     
    csrdata = numpy.empty((M, Nc, numpy.prod(Jd)), order='C', dtype = numpy.complex128  )
    
    for nc in range(0, Nc):
#         tmp_image = coil_sense[...,nc]*(-1.0j)
        
        
        Htensor.hosvd( coil_sense[...,nc], rank=rank)

#         uuu, sss,vvvt = scipy.sparse.linalg.svds(coil_sense[...,nc],k=r)
#         Htensor.U[0] = uuu
#         Htensor.U[1] = vvvt.T.conj()
        core_tensor = Htensor.forward(coil_sense[...,nc])
        
        max_core_tensor = numpy.max(abs(core_tensor.flatten()))
#         print('maximum core tensor value=', max_core_tensor)
#         core_tensor[ abs(core_tensor)< 0.1*max_core_tensor] *= 0
        print('core tensor values=', numpy.diag(core_tensor))
#     core_tensor = coil_sense
#     for jj in range(0, dd):
#         core_tensor=  Htensor.nmode(core_tensor, Htensor.U[jj], jj, if_conj = True)
        
    
#     import matplotlib.pyplot
#     matplotlib.pyplot.imshow(Htensor.U[-1].real)
#     matplotlib.pyplot.show()
    
        print('shape of factor matrix (axis 1) =', Htensor.U[0].shape)
        print('shape of factor matrix (axis 2) =', Htensor.U[1].shape)
        print('shape of factor matrix (axis coil) =', Htensor.U[-1].shape)
        print('shape of core tensor =', core_tensor.shape)
        ud = {}
        for dimid in range(0, dd):  # iterate through all dimensions
    
            """
            Compute the coil sensitivity aware interpolator
            """
    
            if ft_flag[dimid] is True:
                L = numpy.reshape(Htensor.U[dimid], (Nd[dimid], 1, Htensor.U[dimid].shape[1]))
                bn = numpy.reshape(list_bn[dimid], (Nd[dimid], M, 1))
                
                c = solve_c2( list_C[dimid], L.conj()*bn) 
                # C: (N, J) 
                # bn: (N, M, r)
                # -> c: (J, M, r)
                
                print('new method: computes the phase separately then multiplies interpolator with the phase')
                u2 = numpy.einsum( 'jmr, jm -> mjr', c, list_ph[dimid]).conj() # take the conjugate transpose
                
                # u2: (M, J, r)
                ud[dimid] = u2 #*(0.70711 + 0.70711j)
    
            else:
                ud[dimid] = numpy.ones((M, 1, 1), dtype = dtype)
    
        """
        Now compute the column indeces for 1D interpolators
        Each length-Jd interpolator includes Jd points, which are linked to Jd k-space locations
        kd is a tuple storing the 1D interpolators. 
        A following Kronecker product will be needed.
        """
        
        kd = {}
        for dimid in range(0, dd):  # iterate over all dimensions
            kd[dimid] = OMEGA_k(Jd[dimid],Kd[dimid], om[:,dimid], Kd, dimid, dd, ft_flag[dimid]).T

        CSR = full_kron2(ud, kd, Jd, Kd, M, core_tensor)
        del core_tensor
        
#         indptr[:, nc] = CSR.indptr + (numpy.arange(0, M+1) * (Nc - 1) + nc ) * numpy.prod(Jd)
        csrindices[:, nc, :] = numpy.reshape(CSR.indices, (M, numpy.prod(Jd)), order='C')
        csrdata[:, nc, :] = numpy.reshape(CSR.data, (M, numpy.prod(Jd)), order='C')
        
#         if nc == 0:
#             st['p'] = CSR
#         else:
#             st['p'] = scipy.sparse.bmat([[st['p']],[CSR]])
    csrdata = numpy.array(csrdata, order='C')
    csrindices = numpy.array(csrindices, order='C')
    st['p'] = scipy.sparse.csr_matrix(  (csrdata.ravel(order='C'), csrindices.ravel(order='C'), numpy.arange(0, M*Nc +1)* numpy.prod(Jd)))
    return st #new