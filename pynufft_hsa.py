'''
@package docstring

Copyright (c) 2014-2017 Pynufft team.
All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer. Redistributions in binary
form must reproduce the above copyright notice, this list of conditions and
the following disclaimer in the documentation and/or other materials provided
with the distribution. Neither the name of Enthought nor the names of the
Pynufft Developers may be used to endorse or promote products derived from
this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 'AS IS'
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

from __future__ import division

from numpy.testing import (run_module_suite, assert_raises, assert_equal,
                           assert_almost_equal)

from unittest import skipIf
import numpy

import scipy.sparse  # TODO: refactor to remove this

from scipy.sparse.csgraph import _validation  # for cx_freeze debug
# import sys
# import scipy.fftpack
import numpy.fft
 
import scipy.linalg
# import pyopencl 
# import pyopencl.array
# import reikna

from reikna import cluda
import reikna.transformations
from reikna.cluda import functions, dtypes


dtype = numpy.complex64
try:
    api = cluda.ocl_api()
    print("using OpenCL API")
except:
    api = cluda.cuda_api()
    print("using Cuda API")
# api = cluda.cuda_api()
try:
    platform = api.get_platforms()[1]

except:
    platform = api.get_platforms()[0]

device = platform.get_devices()[0]
print('device = ', device)
# print('using cl device=',device,device[0].max_work_group_size, device[0].max_compute_units,pyopencl.characterize.get_simd_group_size(device[0], dtype.size))


def preindex_copy(Nd, Kd):
    """
    Building the array index for copying two arrays of sizes Nd and Kd
    
    The output array2 is either truncated (if Nd > Kd) or zero-padded (if Nd < Kd)
    
    input: Nd: tuple, the dimensions of array1
                Kd: tuple, the dimensions of array2
    output: inlist: the index of the input array1
                    outlist: the index of the output array2
                    nelem: the product of all the smaller lengths along each dimension
    example:
    
    
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

def checker(input_var, desire_size):
    '''
    check if debug = 1
    '''

    if input_var is None:
        print('input_variable does not exist!')
    if desire_size is None:
        print('desire_size does not exist!')

    dd = numpy.size(desire_size)
    dims = numpy.shape(input_var)
#     print('dd=',dd,'dims=',dims)
    if numpy.isnan(numpy.sum(input_var[:])):
        print('input has NaN')

    if numpy.ndim(input_var) < dd:
        print('input signal has too few dimensions')

    if dd > 1:
        if dims[0:dd] != desire_size[0:dd]:
            print(dims[0:dd])
            print(desire_size)
            print('input signal has wrong size1')
    elif dd == 1:
        if dims[0] != desire_size:
            print(dims[0])
            print(desire_size)
            print('input signal has wrong size2')

    if numpy.mod(numpy.prod(dims), numpy.prod(desire_size)) != 0:
        print('input signal shape is not multiples of desired size!')



def dirichlet(x):
    return numpy.sinc(x)


def outer_sum(xx, yy):
    nx = numpy.size(xx)
    ny = numpy.size(yy)

    arg1 = numpy.tile(xx, (ny, 1)).T
    arg2 = numpy.tile(yy, (nx, 1))
    return arg1 + arg2


def nufft_offset(om, J, K):
    '''
    For every om points(outside regular grids), find the nearest
    central grid (from Kd dimension)
    '''
    gam = 2.0 * numpy.pi / (K * 1.0)
    k0 = numpy.floor(1.0 * om / gam - 1.0 * J / 2.0)  # new way
    return k0


def nufft_alpha_kb_fit(N, J, K):
    '''
    find out parameters alpha and beta
    of scaling factor st['sn']
    Note, when J = 1 , alpha is hardwired as [1,0,0...]
    (uniform scaling factor)
    '''
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
    X_ant = beta * gam * nlist.reshape((N, 1), order='C')
    X_post = numpy.arange(0, L + 1)
    X_post = X_post.reshape((1, L + 1), order='C')
    X = numpy.dot(X_ant, X_post)  # [N,L]
    X = numpy.cos(X)
    sn_kaiser = sn_kaiser.reshape((N, 1), order='C').conj()
    X = numpy.array(X, dtype=dtype)
    sn_kaiser = numpy.array(sn_kaiser, dtype=dtype)
    coef = numpy.linalg.lstsq(numpy.nan_to_num(X), numpy.nan_to_num(sn_kaiser))[0]  # (X \ sn_kaiser.H);
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
    interpolation weight for given J/alpha/kb-m
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
    calculate image space scaling factor
    '''
#     import types
#     if alpha is types.ComplexType:
    alpha = numpy.real(alpha)
#         print('complex alpha may not work, but I just let it as')

    L = len(alpha) - 1
    if L > 0:
        sn = numpy.zeros((N, 1))
        n = numpy.arange(0, N).reshape((N, 1), order='C')
        i_gam_n_n0 = 1j * (2 * numpy.pi / K) * (n - Nmid) * beta
        for l1 in range(-L, L + 1):
            alf = alpha[abs(l1)]
            if l1 < 0:
                alf = numpy.conj(alf)
            sn = sn + alf * numpy.exp(i_gam_n_n0 * l1)
    else:
        sn = numpy.dot(alpha, numpy.ones((N, 1), dtype=numpy.float32))
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
     equation (29) and (26)Fessler's paper
     create the overlapping matrix CSSC (diagonal dominent matrix)
     of J points
     and then find out the pseudo-inverse of CSSC '''

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
            cssc = cssc + alf1 * numpy.conj(alf2) * tmp
    return mat_inv(cssc)


def iterate_sum(rr, alf, r1):
    rr = rr + alf * r1
    return rr


def iterate_l1(L, alpha, arg, beta, K, N, rr):
    oversample_ratio = (1.0 * K / N)

    for l1 in range(-L, L + 1):
        alf = alpha[abs(l1)] * 1.0
        if l1 < 0:
            alf = numpy.conj(alf)
    #             r1 = numpy.sinc(1.0*(arg+1.0*l1*beta)/(1.0*K/N))
        input_array = (arg + 1.0 * l1 * beta) / oversample_ratio
        r1 = dirichlet(input_array.astype(numpy.float32))
        rr = iterate_sum(rr, alf, r1)
    return rr


def nufft_r(om, N, J, K, alpha, beta):
    '''
    equation (30) of Fessler's paper

    '''

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
    multiply interpolators of different dimensions
    '''
    (J1, M) = x1.shape
    (J2, M) = x2.shape
#    print(J1,J2,M)
    xx1 = x1.reshape((J1, 1, M), order='C')  # [J1 1 M] from [J1 M]
    xx1 = numpy.tile(xx1, (1, J2, 1))  # [J1 J2 M], emulating ndgrid
    xx2 = x2.reshape((1, J2, M), order='C')  # [1 J2 M] from [J2 M]
    xx2 = numpy.tile(xx2, (J1, 1, 1))  # [J1 J2 M], emulating ndgrid

    y = xx1 * xx2

    return y  # [J1 J2 M]


def block_outer_sum(x1, x2):
    (J1, M) = x1.shape
    (J2, M) = x2.shape
    xx1 = x1.reshape((J1, 1, M), order='C')  # [J1 1 M] from [J1 M]
    xx1 = numpy.tile(xx1, (1, J2, 1))  # [J1 J2 M], emulating ndgrid
    xx2 = x2.reshape((1, J2, M), order='C')  # [1 J2 M] from [J2 M]
    xx2 = numpy.tile(xx2, (J1, 1, 1))  # [J1 J2 M], emulating ndgrid
    y = xx1 + xx2
    return y  # [J1 J2 M]


class NUFFT:

    def __init__(self):
        '''
        Construct the pynufft instance
        '''
        pass

    def plan(self, om, Nd, Kd, Jd):
        '''
        Plan pyNufft
        Start from here
        '''
        self.debug = 0  # debug

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
        n_shift = tuple(0*x for x in Nd)
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
        mm = numpy.tile(mm, [numpy.prod(Jd), 1])  # product(Jd)xM
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
        self.st = st
        
        self.Nd = self.st['Nd']  # backup
        self.sn = numpy.asarray(self.st['sn'].astype(dtype)  ,order='C')# backup
        self.ndims = len(self.st['Nd']) # dimension
        self.linear_phase(n_shift)  # calculate the linear phase thing
        
        # Calculate the density compensation function
        self.precompute_sp()
        del self.st['p'], self.st['sn']
        del self.st['p0'] 
        self.reikna() 
        # off-loading success, now delete matrices on the host side
#         del self.st
#         self.st['W_gpu'] = self.pipe_density()
#         self.st['W'] = self.st['W_gpu'].get()
    def precompute_sp(self):
        """
        Precompute matrices, given that self.st['p'] is known
        """ 
        try:
            self.sp = self.st['p']
            self.spH = (self.st['p'].getH().copy()).tocsr()
            self.spHsp =self.st['p'].getH().dot(self.st['p']).tocsr()
        except:
            print("errors occur in self.precompute_sp()")
            raise
#         self.truncate_selfadjoint( 1e-2)
        

    
    def reikna(self):
        """
        Reikna
        Off-load NUFFT to the opencl or cuda device(s)
        """
        """
        Create context from device
        """
        self.thr = api.Thread(device) #pyopencl.create_some_context()
#         self.queue = pyopencl.CommandQueue( self.ctx)

        """
        Wavefront: as warp in cuda. Can control the width in a workgroup
        Wavefront is required in spmv_vector as it improves data coalescence.
        see cSparseMatVec and zSparseMatVec
        """
        self.wavefront = api.DeviceParameters(device).warp_size
#         print(self.wavefront)
#         print(type(self.wavefront))
#          pyopencl.characterize.get_simd_group_size(device[0], dtype.size)
        from re_subroutine import cMultiplyScalar, cCopy, cAddScalar,cAddVec, cSparseMatVec, cSelect, cMultiplyVec, cMultiplyVecInplace, cMultiplyConjVec, cDiff, cSqrt, cAnisoShrink
        # import complex float routines
#         print(dtypes.ctype(dtype))
        prg = self.thr.compile( 
                                cMultiplyScalar.R + #cCopy.R, 
                                cCopy.R + 
                                cAddScalar.R + 
                                cSelect.R +cMultiplyConjVec.R + cAddVec.R+
                                cMultiplyVecInplace.R +cSparseMatVec.R+cDiff.R+ cSqrt.R+ cAnisoShrink.R+ cMultiplyVec.R,
                                render_kwds=dict(
                                    ctype=dtypes.ctype(dtype),
                                    LL =  str(self.wavefront)), fast_math=False)
#                                fast_math = False)
#                                 "#define LL  "+ str(self.wavefront) + "   "+cSparseMatVec.R)
#                                 render_kwds=dict( ctype=dtypes.ctype(dtype),),
#                                 fast_math=False)
#         prg2 = pyopencl.Program(self.ctx, "#define LL "+ str(self.wavefront) + " "+cSparseMatVec.R).build()
        
        self.cMultiplyScalar = prg.cMultiplyScalar
#         self.cMultiplyScalar.set_scalar_arg_dtypes( cMultiplyScalar.scalar_arg_dtypes)
        self.cCopy = prg.cCopy
        self.cAddScalar = prg.cAddScalar
        self.cAddVec = prg.cAddVec
        self.cSparseMatVec = prg.cSparseMatVec     
        self.cSelect = prg.cSelect
        self.cMultiplyVecInplace = prg.cMultiplyVecInplace
        self.cMultiplyVec = prg.cMultiplyVec
        self.cMultiplyConjVec = prg.cMultiplyConjVec
        self.cDiff = prg.cDiff
        self.cSqrt= prg.cSqrt
        self.cAnisoShrink = prg.cAnisoShrink                                 

#         self.xx_Kd = pyopencl.array.zeros(self.queue, self.st['Kd'], dtype=dtype, order="C")
        self.k_Kd = self.thr.to_device(numpy.zeros(self.st['Kd'], dtype=dtype, order="C"))
        self.k_Kd2 = self.thr.to_device(numpy.zeros(self.st['Kd'], dtype=dtype, order="C"))
        self.y =self.thr.to_device( numpy.zeros((self.st['M'],), dtype=dtype, order="C"))
        self.x_Nd = self.thr.to_device(numpy.zeros(self.st['Nd'], dtype=dtype, order="C"))
#         self.xx_Nd =     pyopencl.array.zeros(self.queue, self.st['Nd'], dtype=dtype, order="C")

        self.NdCPUorder, self.KdCPUorder, self.nelem =     preindex_copy(self.st['Nd'], self.st['Kd'])
        self.NdGPUorder = self.thr.to_device( self.NdCPUorder)
        self.KdGPUorder =  self.thr.to_device( self.KdCPUorder)
        self.Ndprod = numpy.int32(numpy.prod(self.st['Nd']))
        self.Kdprod = numpy.int32(numpy.prod(self.st['Kd']))
        self.M = numpy.int32( self.st['M'])
        
        self.SnGPUArray = self.thr.to_device(  self.sn)
        
        self.sp_data = self.thr.to_device( self.sp.data.astype(dtype))
        self.sp_indices =self.thr.to_device( self.sp.indices.astype(numpy.int32))
        self.sp_indptr = self.thr.to_device( self.sp.indptr.astype(numpy.int32))
        self.sp_numrow =  self.M
        del self.sp
        self.spH_data = self.thr.to_device(  self.spH.data.astype(dtype))
        self.spH_indices = self.thr.to_device(  self.spH.indices)
        self.spH_indptr = self.thr.to_device(  self.spH.indptr)
        self.spH_numrow = self.Kdprod
        del self.spH
        self.spHsp_data = self.thr.to_device(  self.spHsp.data.astype(dtype))
        self.spHsp_indices = self.thr.to_device( self.spHsp.indices)
        self.spHsp_indptr =self.thr.to_device(  self.spHsp.indptr)
        self.spHsp_numrow = self.Kdprod
        del self.spHsp
#         import reikna.cluda
        import reikna.fft
#         api = 
#         self.thr = reikna.cluda.ocl_api().Thread(self.queue)        
        self.fft = reikna.fft.FFT(self.k_Kd, numpy.arange(0, self.ndims)).compile(self.thr, fast_math=False)
#         self.fft = reikna.fft.FFT(self.k_Kd).compile(thr, fast_math=True)
#         self.fft = FFT(self.ctx, self.queue,  self.k_Kd, fast_math=True)
#         self.ifft = FFT(self.ctx, self.queue, self.k_Kd2,  fast_math=True)
        self.zero_scalar=dtype(0.0+0.0j)
#     def solver(self,  gy, maxiter):#, solver='cg', maxiter=200):
    def solver(self,y, solver=None, *args, **kwargs):
        import _solver.solver_hsa
        return _solver.solver_hsa.solver(self,  y,  solver, *args, **kwargs)

    def pipe_density(self):
        '''
        Create the density function by iterative solution
        Generate pHp matrix
        '''
#         self.st['W'] = pipe_density(self.st['p'])
#         W = numpy.ones((self.st['M'],),dtype=dtype)
        W_cpu = numpy.ones((self.st['M'],),dtype=dtype)
        self.W_gpu = pyopencl.array.to_device(self.queue, W_cpu)
#         transA=cuda_cffi.cusparse.CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE
#         V1= self.CSR.getH().copy()
    #     VVH = V.dot(V.getH()) 
        
        for pp in range(0,10):
#             E =  self.CSR.mv(self.CSR.mv(W_gpu,transA=transA))
            E = self.forward(self.adjoint(self.W_gpu))
            self.W_gpu = self.W_gpu/E
#         pHp = self.st['p'].getH().dot(self.st['p'])
        # density of the k-space, reduced size
#         return W_gpu #dirichlet
    
    def linear_phase(self, n_shift):
        '''
        Select the center of FOV
        '''
        om = self.st['om']
        M = self.st['M']
        final_shifts = tuple(
            numpy.array(n_shift) +
            numpy.array(self.st['Nd']) / 2)

        phase = numpy.exp(
            1.0j *
            numpy.sum(
                om * numpy.tile(
                    final_shifts,
                    (M,1)),
                1))
        # add-up all the linear phasees in all axes,

        self.st['p'] = scipy.sparse.diags(phase, 0).dot(self.st['p0'])
 

    def truncate_selfadjoint(self, tol):
        
#         for pp in range(1, 8):
#             self.st['pHp'].setdiag(0,k=pp)
#             self.st['pHp'].setdiag(0,k=-pp)
        indix=numpy.abs(self.spHsp.data)< tol
        self.spHsp.data[indix]=0
 
        self.spHsp.eliminate_zeros()
        indix=numpy.abs(self.sp.data)< tol
        self.sp.data[indix]=0
 
        self.sp.eliminate_zeros()
        
    def forward(self, gx):

            self.x_Nd =  self.thr.copy_array(gx)
            
            self._x2xx()

            self._xx2k()

            self._k2y()

            gy =  self.thr.copy_array(self.y)
            return gy
    
    def adjoint(self, gy):
            self.y = self.thr.copy_array(gy) 

            self._y2k()
            self._k2xx()
            self._xx2x()
            gx = self.thr.copy_array(self.x_Nd)
            return gx
#             self.thr.synchronize()


    def selfadjoint(self, gx):
        self.x_Nd = self.thr.copy_array(gx)
        self._x2xx()
        self._xx2k()
        self._k2y2k()
        self._k2xx()
        self._xx2x()
        gx2 = self.thr.copy_array(self.x_Nd)
        return gx2

    def _x2xx(self):
        '''
        scaling of the image, generate Nd array
        input: self.x_Nd
        output: self.xx_Nd

        '''
#         self.cMultiplyVecInplace(self.queue, (self.Ndprod,), None,  self.SnGPUArray.data, self.x_Nd.data)
        self.cMultiplyVecInplace(self.SnGPUArray, self.x_Nd, local_size=None, global_size=int(self.Ndprod))
        self.thr.synchronize()
    def _xx2k(self ):
        '''
        fft of the Nd array
        (1)Nd array is copied to Kd array(zeros) by cSelect
        input: self.xx_Nd,   self.NdGPUorder.data,      self.KdGPUorder.data,
        output: self.xx_Kd
        (2) forward FFT:  
        input: self.xx_Kd, 
        output: self.k_Kd

        '''
        self.cMultiplyScalar(self.zero_scalar, self.k_Kd, local_size=None, global_size=int(self.Kdprod))
        self.cSelect(self.NdGPUorder,      self.KdGPUorder,  self.x_Nd, self.k_Kd, local_size=None, global_size=int(self.Ndprod))
        self.fft( self.k_Kd,self.k_Kd,inverse=False)
        self.thr.synchronize()
    def _k2y(self ):
        """
        interpolation achieved by Sparse Matrix-Vector Multiplication
        input: self.k_Kd (C-order, row-major)
        output: self.y 
        """
        
#         self.cSparseMatVec(self.queue, 
#                                     (self.sp_numrow*self.wavefront,         ), 
#                                     (self.wavefront,          ),
#                                    self.sp_numrow, 
#                                    self.sp_indptr.data,
#                                    self.sp_indices.data,
#                                    self.sp_data.data, 
#                                    self.k_Kd.data,
#                                    self.y.data)
        self.cSparseMatVec(                                
                                   self.sp_numrow, 
                                   self.sp_indptr,
                                   self.sp_indices,
                                   self.sp_data, 
                                   self.k_Kd,
                                   self.y,
                                   local_size=int(self.wavefront),
                                   global_size=int(self.sp_numrow*self.wavefront) 
                                    )
        self.thr.synchronize()
    def _y2k(self):
        '''
        input:
            y:    non-Cartesian data
        output:
            k:    k-space data on Kd grid
        '''

#         self.cSparseMatVec(self.queue, 
#                                     (self.spH_numrow*self.wavefront,), 
#                                     (self.wavefront,),
#                                    self.spH_numrow, 
#                                    self.spH_indptr.data,
#                                    self.spH_indices.data,
#                                    self.spH_data.data, 
#                                    self.y.data,
#                                    self.k_Kd2.data)#,g_times_l=int(csrnumrow))
        self.cSparseMatVec(
                                   self.spH_numrow, 
                                   self.spH_indptr,
                                   self.spH_indices,
                                   self.spH_data, 
                                   self.y,
                                   self.k_Kd2,
                                   local_size=int(self.wavefront),
                                   global_size=int(self.spH_numrow*self.wavefront) 
                                    )#,g_times_l=int(csrnumrow))
#         return k
        self.thr.synchronize()
    def _k2y2k(self):
#         self.cSparseMatVec(self.queue, (self.spHsp_numrow*self.wavefront,), (self.wavefront,),
#                                    self.spHsp_numrow, 
#                                    self.spHsp_indptr.data,
#                                    self.spHsp_indices.data,
#                                    self.spHsp_data.data, 
#                                    self.k_Kd.data,
#                                    self.k_Kd2.data)#,g_times_l=int(csrnumrow))
        self.cSparseMatVec( 
                                    
                                   self.spHsp_numrow, 
                                   self.spHsp_indptr,
                                   self.spHsp_indices,
                                   self.spHsp_data, 
                                   self.k_Kd,
                                   self.k_Kd2,
                                   local_size=int(self.wavefront),
                                   global_size=int(self.spHsp_numrow*self.wavefront) 
                                    )#,g_times_l=int(csrnumrow))

    def _k2xx(self):

#         self.cMultiplyScalar(self.queue, (self.Kdprod,), None, 0.0, self.xx_Kd.data)
        
#         event,= self.ifft.enqueue( forward = False)
#         event.wait()
        self.fft( self.k_Kd2, self.k_Kd2,inverse=True)
#         self.x_Nd._zero_fill()
        self.cMultiplyScalar(self.zero_scalar, self.x_Nd,  local_size=None, global_size=int(self.Ndprod ))
#         self.cSelect(self.queue, (self.Ndprod,), None,   self.KdGPUorder.data,  self.NdGPUorder.data,     self.k_Kd2.data, self.x_Nd.data )
        self.cSelect(  self.KdGPUorder,  self.NdGPUorder,     self.k_Kd2, self.x_Nd, local_size=None, global_size=int(self.Ndprod ))
        
        self.thr.synchronize()
    def _xx2x(self ):
        self.cMultiplyVecInplace( self.SnGPUArray, self.x_Nd, local_size=None, global_size=int(self.Ndprod))
        self.thr.synchronize()

def test_init():
    import cProfile
    import numpy
    import matplotlib.pyplot
    import copy

    cm = matplotlib.cm.gray
    # load example image
    import pkg_resources
    
    DATA_PATH = pkg_resources.resource_filename('pynufft', 'data/')
#     PHANTOM_FILE = pkg_resources.resource_filename('pynufft', 'data/phantom_256_256.txt')
    import numpy
    import matplotlib.pyplot
    import scipy
    # load example image
#     image = numpy.loadtxt(DATA_PATH +'phantom_256_256.txt')
#     image = scipy.misc.face(gray=True)
    image = scipy.misc.ascent()    
    image = scipy.misc.imresize(image, (256,256))
    
    image=image.astype(numpy.float)/numpy.max(image[...])
    #numpy.save('phantom_256_256',image)
    matplotlib.pyplot.subplot(1,3,1)
    matplotlib.pyplot.imshow(image, cmap=matplotlib.cm.gray)
    matplotlib.pyplot.title("Load Scipy \"ascent\" image")
#     matplotlib.pyplot.show()
    print('loading image...')
#     image[128, 128] = 1.0
    Nd = (256, 256)  # image space size
    Kd = (512, 512)  # k-space size
    Jd = (6, 6)  # interpolation size

    # load k-space points
    om = numpy.load(DATA_PATH+'om2D.npz')['arr_0']

    # create object
    
#         else:
#             n_shift=tuple(list(n_shift)+numpy.array(Nd)/2)
    import pynufft
    nfft = pynufft.NUFFT()  # CPU
    nfft.plan(om, Nd, Kd, Jd)
#     nfft.initialize_gpu()
    import scipy.sparse
#     scipy.sparse.save_npz('tests/test.npz', nfft.st['p'])

    NufftObj = NUFFT()

    NufftObj.plan(om, Nd, Kd, Jd)

#     print('sp close? = ', numpy.allclose( nfft.st['p'].data,  NufftObj.st['p'].data, atol=1e-1))
#     NufftObj.initialize_gpu()

    y = nfft.k2y(nfft.xx2k(nfft.x2xx(image)))
    NufftObj.x_Nd = NufftObj.thr.to_device( image.astype(dtype))
    gx = NufftObj.thr.copy_array(NufftObj.x_Nd)
    print('x close? = ', numpy.allclose(image, NufftObj.x_Nd.get() , atol=1e-4))
    NufftObj._x2xx()    
#     ttt2= NufftObj.thr.copy_array(NufftObj.x_Nd)
    print('xx close? = ', numpy.allclose(nfft.x2xx(image), NufftObj.x_Nd.get() , atol=1e-4))        

    NufftObj._xx2k()    
    
#     print(NufftObj.k_Kd.get(queue=NufftObj.queue, async=True).flags)
#     print(nfft.xx2k(nfft.x2xx(image)).flags)
    k = nfft.xx2k(nfft.x2xx(image))
    print('k close? = ', numpy.allclose(nfft.xx2k(nfft.x2xx(image)), NufftObj.k_Kd.get() , atol=1e-3*numpy.linalg.norm(k)))   
    
    NufftObj._k2y()    
    
    
    NufftObj._y2k()
    y2 = NufftObj.y.get(   )
    
    print('y close? = ', numpy.allclose(y, y2 ,  atol=1e-3*numpy.linalg.norm(y)))
#     print(numpy.mean(numpy.abs(nfft.y2k(y)-NufftObj.k_Kd2.get(queue=NufftObj.queue, async=False) )))
    print('k2 close? = ', numpy.allclose(nfft.y2k(y2), NufftObj.k_Kd2.get(), atol=1e-3*numpy.linalg.norm(nfft.y2k(y2)) ))   
    NufftObj._k2xx()
#     print('xx close? = ', numpy.allclose(nfft.k2xx(nfft.y2k(y2)), NufftObj.xx_Nd.get(queue=NufftObj.queue, async=False) , atol=0.1))
    NufftObj._xx2x()
    print('x close? = ', numpy.allclose(nfft.adjoint(y2), NufftObj.x_Nd.get() , atol=1e-3*numpy.linalg.norm(nfft.adjoint(y2))))
    image3 = NufftObj.x_Nd.get() 
    import time
    t0 = time.time()
    for pp in range(0,10):
#         y = nfft.k2y(nfft.xx2k(nfft.x2xx(image)))    
#             x = nfft.adjoint(y)
            y = nfft.forward(image)
#     y2 = NufftObj.y.get(   NufftObj.queue, async=False)
    t_cpu = (time.time() - t0)/10.0 
    print(t_cpu)
    
#     del nfft
        
    
    t0= time.time()
    for pp in range(0,100):
#         pass
#         NufftObj.adjoint()
        gy=NufftObj.forward(gx)        
        
#     NufftObj.thr.synchronize()
    t_cl = (time.time() - t0)/100
    print(t_cl)
    print('gy close? = ', numpy.allclose(y,gy.get(),  atol=1e-1))
    print("acceleration=", t_cpu/t_cl)
    maxiter = 200
    import time
    t0= time.time()
#     x2 = nfft.solver(y2, 'cg',maxiter=maxiter)
    x2 =  nfft.solver(y2, 'L1LAD',maxiter=maxiter, rho = 3)
    t1 = time.time()-t0
#     gy=NufftObj.thr.copy_array(NufftObj.thr.to_device(y2))
    
    t0= time.time()
#     x = NufftObj.solver(gy,'cg', maxiter=maxiter)
    x = NufftObj.solver(gy,'L1LAD', maxiter=maxiter, rho=3)
    
    t2 = time.time() - t0
    print(t1, t2)
    print('acceleration=', t1/t2 )
#     k = x.get()
#     x = nfft.k2xx(k)/nfft.st['sn']
#     return
    
    matplotlib.pyplot.subplot(1, 3, 2)
    matplotlib.pyplot.imshow( NufftObj.x_Nd.get().real, cmap= matplotlib.cm.gray)
    matplotlib.pyplot.subplot(1, 3,3)
    matplotlib.pyplot.imshow(x2.real, cmap= matplotlib.cm.gray)
    matplotlib.pyplot.show()
def test_cAddScalar():

    dtype = numpy.complex64
    
    try:
        device=pyopencl.get_platforms()[1].get_devices()
        
    except:
        device=pyopencl.get_platforms()[0].get_devices()
    print('using cl device=',device,device[0].max_work_group_size, device[0].max_compute_units,pyopencl.characterize.get_simd_group_size(device[0], dtype.size))

    ctx = pyopencl.Context(device) #pyopencl.create_some_context()
    queue = pyopencl.CommandQueue(ctx)
    wavefront = pyopencl.characterize.get_simd_group_size(device[0], dtype.size)

#     B = routine(wavefront)
    import cl_subroutine.cAddScalar
    prg = pyopencl.Program(ctx, cl_subroutine.cAddScalar.R).build()
    
    AddScalar = prg.cAddScalar
    AddScalar.set_scalar_arg_dtypes(cl_subroutine.cAddScalar.scalar_arg_dtypes)
#     indata= numpy.arange(0,128).astype(dtype)
    indata = (numpy.random.randn(128,)+numpy.random.randn(128,)*1.0j).astype(dtype)      
    indata_g = pyopencl.array.to_device(queue, indata)
    scal= 0.1+0.1j
    AddScalar(queue, (128,),None,scal, indata_g.data)
    print(-indata[0]+indata_g.get()[0])
    
if __name__ == '__main__':
    import cProfile
    test_init()
#     test_cAddScalar()
#     cProfile.run('test_init()')
