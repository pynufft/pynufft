'''
@package docstring

Copyright (c) 2014-2016 Pynufft team.
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

# from cuda_cffi.cusparse import *
# from cuda_cffi.cusparse import (_csrgeamNnz, _csrgemmNnz)

# from cuda_cffi import cusparse

import cuda_cffi
import cuda_cffi.cusparse
#import cuda_cffi.cusolver

# import pycuda.autoinit


from numpy.testing import (run_module_suite, assert_raises, assert_equal,
                           assert_almost_equal)

from unittest import skipIf

import pycuda.autoinit
import pycuda.gpuarray
import pycuda.driver as drv
from pycuda.elementwise import ElementwiseKernel


import scipy.sparse  # TODO: refactor to remove this
import numpy
from pycuda.compyte.array import f_contiguous_strides

# cusparse_real_dtypes = []  # [numpy.float32, numpy.float64]
# cusparse_complex_dtypes = [numpy.complex64, ]  # numpy.complex64]
# cusparse_dtypes = cusparse_real_dtypes + cusparse_complex_dtypes
# trans_list = [cuda_cffi.cusparse.CUSPARSE_OPERATION_NON_TRANSPOSE,
#               cuda_cffi.cusparse.CUSPARSE_OPERATION_TRANSPOSE,
#               cuda_cffi.cusparse.CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE]


# descrA = cuda_cffi.cusparse.cusparseCreateMatDescr()
# cuda_cffi.cusparse.cusparseSetMatType(
#     descrA, cuda_cffi.cusparse.CUSPARSE_MATRIX_TYPE_GENERAL)
# cuda_cffi.cusparse.cusparseSetMatIndexBase(
#     descrA, cuda_cffi.cusparse.CUSPARSE_INDEX_BASE_ZERO)


import numpy
import scipy.sparse
from scipy.sparse.csgraph import _validation  # for cx_freeze debug
# import sys
# import scipy.fftpack
import numpy.fft
import skcuda

dtype = numpy.complex64

import scipy.signal

import scipy.linalg

import skcuda.fft as cu_fft
import skcuda.misc as cu_misc
import skcuda.linalg as cu_linalg
# cu_linalg.init()
cuda_cffi.cusparse.init()
#cuda_cffi.cusolver.init()




# def pipe_density(V):
#     '''
#     Compute the density compensation function for non-uniform samples.
#     '''
# 
#     V1 = V.getH()
#     b = numpy.ones((V.get_shape()[0], ), dtype=dtype)
#     from scipy.sparse.linalg import lsmr, bicg, cg
#     V2 = V.dot(V.getH())
#     tmp_W = lsmr(V2, b, atol=1e-3)
#     W = numpy.reshape(tmp_W[0], (V.get_shape()[0], ), order='F')
#     # reshape vector
#     return W
# from string import Template
gpuCrop2DS = ElementwiseKernel( # unwrap the indices from the last dimension
        "pycuda::complex<float> *dest, pycuda::complex<float> *orig, int Nout, int Nin, int Tot",
        " unsigned int a;"+  
        "unsigned int b;"+ 
        "if(i<Tot){"+
        "if(Nout<Nin){a=i/Nout;b=i%Nout;};"+
        "if(Nout>=Nin){a=i/Nin;b=i%Nin;};"+
            "};dest[a*Nout + b] = orig[a*Nin + b];",
        "complex5",
        preamble="#include <pycuda-complex.hpp>",)

gpuCrop2DZ = ElementwiseKernel( # unwrap the indices from the last dimension
        "pycuda::complex<double> *dest, pycuda::complex<double> *orig, int Nout, int Nin, int Tot",
        
        " unsigned int a;"+  
        "unsigned int b;"+ 
        "if(i<Tot){"+
        "if(Nout<Nin){a=i/Nout;b=i%Nout;};"+
        "if(Nout>=Nin){a=i/Nin;b=i%Nin;};"+
            "};dest[a*Nout + b] = orig[a*Nin + b];",
        "complex5",
        preamble="#include <pycuda-complex.hpp>",)
gpuCrop3DS = ElementwiseKernel(
        "pycuda::complex<float> *dest, pycuda::complex<float> *orig, int Nyout, int Nyin, int Nzout, int Nzin, int Tot",
        "unsigned int ay;" + "unsigned int az;" + 
        "unsigned int by;" +  "unsigned int bz;" +
        "if(i<Tot){" + 
        "if(Nzout < Nzin){    az=i/Nzout;     bz=i%Nzout; "   +
            "if(Nyout < Nyin){ay=az/Nyout;    by=az%Nyout;};"  + 
            "if(Nyout >= Nyin){ay=az/Nyin;    by=az%Nyin; };};" +
        "if(Nzout >= Nzin){    az=i/Nzin;     bz=i%Nzin;"    +
            "if(Nyout < Nyin){ ay=az/Nyout;    by=bz%Nyout; };"  + 
            "if(Nyout >= Nyin){ay=az/Nyin;    by=bz%Nyin; }; };"+
            "dest[(ay*Nyout + by)*Nzout + bz] = orig[(ay*Nyin + by)*Nzin + bz];};",
        "complex5",
        preamble="#include <pycuda-complex.hpp>",)

gpuCrop3DZ = ElementwiseKernel(
        "pycuda::complex<double> *dest, pycuda::complex<double> *orig, int Nout, int Nin, int Tot",
                "unsigned int ay;" + "unsigned int az;" + 
        "unsigned int by;" +  "unsigned int bz;" +
        "if(i<Tot){" + 
        "if(Nzout < Nzin){    az=i/Nzout;     bz=i%Nzout; "   +
            "if(Nyout < Nyin){ay=az/Nyout;    by=az%Nyout;};"  + 
            "if(Nyout >= Nyin){ay=az/Nyin;    by=az%Nyin; };};" +
        "if(Nzout >= Nzin){    az=i/Nzin;     bz=i%Nzin;"    +
            "if(Nyout < Nyin){ ay=az/Nyout;    by=bz%Nyout; };"  + 
            "if(Nyout >= Nyin){ay=az/Nyin;    by=bz%Nyin; }; };"+
            "dest[(ay*Nyout + by)*Nzout + bz] = orig[(ay*Nyin + by)*Nzin + bz];};",
        "complex5",
        preamble="#include <pycuda-complex.hpp>",)

 
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
        L = numpy.ceil(N / 3)

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
        n = numpy.arange(0, N).reshape((N, 1), order='F')
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
    xx1 = x1.reshape((J1, 1, M), order='F')  # [J1 1 M] from [J1 M]
    xx1 = numpy.tile(xx1, (1, J2, 1))  # [J1 J2 M], emulating ndgrid
    xx2 = x2.reshape((1, J2, M), order='F')  # [1 J2 M] from [J2 M]
    xx2 = numpy.tile(xx2, (J1, 1, 1))  # [J1 J2 M], emulating ndgrid

    y = xx1 * xx2

    return y  # [J1 J2 M]


def block_outer_sum(x1, x2):
    (J1, M) = x1.shape
    (J2, M) = x2.shape
    xx1 = x1.reshape((J1, 1, M), order='F')  # [J1 1 M] from [J1 M]
    xx1 = numpy.tile(xx1, (1, J2, 1))  # [J1 J2 M], emulating ndgrid
    xx2 = x2.reshape((1, J2, M), order='F')  # [1 J2 M] from [J2 M]
    xx2 = numpy.tile(xx2, (J1, 1, 1))  # [J1 J2 M], emulating ndgrid
    y = xx1 + xx2
    return y  # [J1 J2 M]


def crop_slice_ind(Nd):
    return [slice(0, Nd[_ss]) for _ss in range(0, len(Nd))]


class gpuNUFFT:

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
        st['M'] = M
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
            st['sn'] = numpy.reshape(st['sn'], (dimid_cnt, 1), order='F')
            # JML do not apply scaling

        # order = 'F' is for fortran order
        st['sn'] = st['sn'].reshape(Nd, order='F')  # [(Nd)]
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
#             if dimid > 0:  # trick: pre-convert these indices into offsets!
#                 #            ('trick: pre-convert these indices into offsets!')
#                 kd[dimid] = kd[dimid] * numpy.prod(Kd[0:dimid]) - 1
            if dimid < dd - 1:  # trick: pre-convert these indices into offsets!
                #            ('trick: pre-convert these indices into offsets!')
                kd[dimid] = kd[dimid] * numpy.prod(Kd[dimid+1:dd]) - 1 

        kk = kd[0]  # [J1 M] # pointers to indices
        uu = ud[0]  # [J1 M]
        Jprod = Jd[0]
        Kprod = Kd[0]
        for dimid in range(1, dd):
            Jprod = numpy.prod(Jd[:dimid + 1])
            Kprod = numpy.prod(Kd[:dimid + 1])
            kk = block_outer_sum(kk, kd[dimid]) + 1  # outer sum of indices
            kk = kk.reshape((Jprod, M), order='F')
            # outer product of coefficients
            uu = block_outer_prod(uu, ud[dimid])
            uu = uu.reshape((Jprod, M), order='F')
            # now kk and uu are [*Jd M]
            # now kk and uu are [*Jd M]
        # *numpy.tile(phase,[numpy.prod(Jd),1]) #    product(Jd)xM
        uu = uu.conj()
        mm = numpy.arange(0, M)  # indices from 0 to M-1
        mm = numpy.tile(mm, [numpy.prod(Jd), 1])  # product(Jd)xM
        # Now build sparse matrix from uu, mm, kk

        # convert array to list
        csrdata = numpy.reshape(uu, (Jprod * M, ), order='F')

        # row indices, from 1 to M convert array to list
        rowindx = numpy.reshape(mm, (Jprod * M, ), order='F')

        # colume indices, from 1 to prod(Kd), convert array to list
        colindx = numpy.reshape(kk, (Jprod * M, ), order='F')

        # The shape of sparse matrix
        csrshape = (M, numpy.prod(Kd))

        # Build sparse matrix (interpolator)
        st['p0'] = scipy.sparse.csr_matrix((csrdata, (rowindx, colindx)),
                                           shape=csrshape)
        # Note: the sparse matrix requires the following linear phase,
        #       which moves the image to the center of the image
        
        self.st = st
#         self.truncate_selfadjoint( 1e-5)
        self.Nd = self.st['Nd']  # backup
        self.sn = self.st['sn']  # backup
        self.ndims = len(self.st['Nd']) # dimension
        self.linear_phase(n_shift)  # calculate the linear phase thing
        
        # Calculate the density compensation function

        self.finalization()
        self.st['W_gpu'] = self.pipe_density()
        self.st['W'] = self.st['W_gpu'].get()
        self.precompute()
    
    def precompute(self):
        
#         CSR_W = cuda_cffi.cusparse.CSR.to_CSR(self.st['W_gpu'],diag_type=True)

#         Dia_W_cpu = scipy.sparse.dia_matrix( (self.st['M'], self.st['M']),dtype=dtype)
#         Dia_W_cpu = scipy.sparse.dia_matrix( ( self.st['W'], 0 ), shape=(self.st['M'], self.st['M']) )
#         Dia_W_cpu = scipy.sparse.diags(self.st['W'], format="csr", dtype=dtype)
#         CSR_W = cuda_cffi.cusparse.CSR.to_CSR(Dia_W_cpu)

        
        self.st['pHp_gpu'] = self.CSRH.gemm(self.CSR)
        self.st['pHp']=self.st['pHp_gpu'].get()
        print('untrimmed',self.st['pHp'].nnz)
        self.truncate_selfadjoint(1e-5)
        print('trimmed', self.st['pHp'].nnz)
        self.st['pHp_gpu'] = cuda_cffi.cusparse.CSR.to_CSR(self.st['pHp'])
#         self.st['pHWp_gpu'] = self.CSR.conj().gemm(CSR_W,transA=cuda_cffi.cusparse.CUSPARSE_OPERATION_TRANSPOSE)
#         self.st['pHWp_gpu'] = self.st['pHWp_gpu'].gemm(self.CSR, transA=cuda_cffi.cusparse.CUSPARSE_OPERATION_NON_TRANSPOSE)         
        
    def pipe_density(self):
        '''
        Create the density function by iterative solution
        Generate pHp matrix
        '''
#         self.st['W'] = pipe_density(self.st['p'])
#         W = numpy.ones((self.st['M'],),dtype=dtype)
        W_gpu = skcuda.misc.ones((self.st['M'],),dtype=dtype)
#         transA=cuda_cffi.cusparse.CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE
#         V1= self.CSR.getH().copy()
    #     VVH = V.dot(V.getH()) 
        
        for pp in range(0,10):
#             E =  self.CSR.mv(self.CSR.mv(W_gpu,transA=transA))
            E = self.forward(self.adjoint(W_gpu))
            W_gpu = W_gpu/E
#         pHp = self.st['p'].getH().dot(self.st['p'])
        # density of the k-space, reduced size
        return W_gpu #dirichlet
    
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
        # multiply the diagonal, linear phase before the gridding matrix

    def finalization(self):
        '''
        Add sparse matrix multiplication on GPU
        Note: use "python-cuda-cffi" generated interface to access cusparse

        '''
        self.gpu_flag = 0

        self.CSR = cuda_cffi.cusparse.CSR.to_CSR(self.st['p'].astype(dtype), )
        self.CSRH = cuda_cffi.cusparse.CSR.to_CSR(self.st['p'].getH().tocsr().astype(dtype), )
        
        self.scikit_plan = cu_fft.Plan(self.st['Kd'], dtype, dtype)
#         self.pHp = cuda_cffi.cusparse.CSR.to_CSR(
#             self.st['pHp'].astype(dtype))
        
        self.gpu_flag = 1
        self.sn_gpu = pycuda.gpuarray.to_gpu(self.sn.astype(dtype))
#         tmp_array = skcuda.misc.ones((numpy.prod(self.st['Kd']),1),dtype=dtype)
#         tmp = cuda_cffi.cusolver.csrlsvqr(self.CSR, tmp_array)
    def truncate_selfadjoint(self, tol):
        indix=numpy.abs(self.st['pHp'].data)< tol
        self.st['pHp'].data[indix]=0

        self.st['pHp'].eliminate_zeros()

    def forward(self, x):

        y = self.k2y(
            self.xx2k(
                self.x2xx(x)
            )
        )

        return y

    def adjoint(self, y):

        x = self.xx2x(
            self.k2xx(
                self.y2k(y)
            )
        )

        return x

    def selfadjoint(self, x):

        x2 = self.xx2x(self.k2xx(self.k2k(self.xx2k(self.x2xx(x)))))

        return x2
#     def forward_modulate_adjoint(self, x):
# 
#         x2 = self.xx2x(self.k2xx(self.kWk(self.xx2k(self.x2xx(x)))))
# #         self.backward(self.st['W_gpu']*self.forward(x))
#         return x2

    def x2xx(self, x):
        '''
        scaling of the image, generate Nd
        input:
            x: 2D image
        output:
            xx: scaled 2D image
        '''

#         xx = x*self.st['sn']
        xx = cu_misc.multiply(x, self.sn_gpu)
        return xx

    def xx2k(self, xx):
        '''
        fft of the image
        input:
            xx:    scaled 2D image
        output:
            k:    k-space grid
        '''
        dd = numpy.size(self.st['Kd'])


#         if dd == 2: # 2D case
#
# #             inflat_matrix_L = numpy.eye(self.st['Kd'][0], self.st['Nd'][0], dtype=dtype).astype(dtype=dtype,order='F')
# #             inflat_matrix_R = numpy.eye(self.st['Nd'][1], self.st['Kd'][1], dtype=dtype).astype(dtype=dtype,order='F')
# #             output_x = numpy.dot(xx, inflat_matrix_R)
# #             output_x = numpy.dot(inflat_matrix_L, output_x).astype(dtype=dtype,order='F')
# #             print(numpy.shape(output_x), type(output_x))
#             output_x=numpy.zeros(self.st['Kd'], dtype=dtype, order='F')
#             output_x[   0:self.st['Nd'][0],    0:self.st['Nd'][1]   ] = xx[   0:self.st['Nd'][0],    0:self.st['Nd'][1]   ]
#         elif dd == 1: # 1D case
# #             inflat_matrix_L = numpy.eye(self.st['Kd'][0], self.st['Nd'][0], dtype=dtype)
# #             output_x = numpy.dot(inflat_matrix_L, output_x).astype(dtype)
#             output_x=numpy.zeros(self.st['Kd'], dtype=dtype, order='F')
#             output_x[0:self.st['Nd'][0]] = xx[0:self.st['Nd'][0]]
#         elif dd == 3:
#             output_x=numpy.zeros(self.st['Kd'], dtype=dtype, order='F')
#             output_x[0:self.st['Nd'][0],0:self.st['Nd'][1],0:self.st['Nd'][2]] = xx[0:self.st['Nd'][0],0:self.st['Nd'][1],0:self.st['Nd'][2]]

#         input_gpu = pycuda.gpuarray.to_gpu(xx.astype(dtype))

        dd = numpy.size(self.st['Nd'])
        if dd == 2:
#             output_gpu_row = pycuda.gpuarray.zeros(
#                 shape=(self.st['Kd'][0], self.st['Nd'][1]), dtype=dtype, order='C')
#   
#             input_gpu = xx
#             drv.memcpy_dtod(output_gpu_row.ptr,
#                             input_gpu.ptr, input_gpu.nbytes)
#   
#             k_gpu_T = pycuda.gpuarray.zeros(
#                 shape=(self.st['Kd'][1], self.st['Kd'][0]), dtype=dtype, order='C')
#   
#             output_gpu_col = cu_linalg.transpose(output_gpu_row)
#   
#             drv.memcpy_dtod(k_gpu_T.ptr, output_gpu_col.ptr,
#                             output_gpu_col.nbytes)
#   
#             k_gpu = cu_linalg.transpose(k_gpu_T)
            
            k_gpu = pycuda.gpuarray.zeros(
                shape=(self.st['Kd'][0], self.st['Kd'][1]), dtype=dtype, order='C')
            
            if dtype == numpy.complex64:
                gpuCrop2DS(k_gpu, xx, self.st['Kd'][1], self.st['Nd'][1], numpy.prod(self.st['Nd']))
                
            elif dtype == numpy.complex128:
                gpuCrop2DZ(k_gpu, xx, self.st['Kd'][1], self.st['Nd'][1], numpy.prod(self.st['Nd']))
            
        elif dd == 1:
            k_gpu = pycuda.gpuarray.zeros(
                shape=(self.st['Kd'][0], ), dtype=dtype, order='C')

#             input_gpu_T = cu_linalg.transpose(input_gpu)
            drv.memcpy_dtod(k_gpu.ptr,
                            input_gpu.ptr, input_gpu.nbytes)

        elif dd == 3:
            k_gpu = pycuda.gpuarray.zeros(
                shape=(self.st['Kd'][0], self.st['Kd'][1],  self.st['Kd'][2]), dtype=dtype, order='C')
            
            if dtype == numpy.complex64:
                gpuCrop3DS(k_gpu, xx, self.st['Kd'][1], self.st['Nd'][1], self.st['Kd'][2], self.st['Nd'][2], numpy.prod(self.st['Nd']))
                
            elif dtype == numpy.complex128:
                gpuCrop3DZ(k_gpu, xx, self.st['Kd'][1], self.st['Nd'][1], self.st['Kd'][2], self.st['Nd'][2], numpy.prod(self.st['Nd']))


#         if numpy.size(self.st['Nd']) == 2:
#             output_x[0:self.st['Nd'][0],    0:self.st['Nd'][1]] =input_x[0:self.st['Nd'][0],    0:self.st['Nd'][1]]
#         elif numpy.size(self.st['Nd']) == 3:
#             output_x[0:self.st['Nd'][0],    0:self.st['Nd'][1],    0:self.st['Nd'][2]] =input_x[0:self.st['Nd'][0],    0:self.st['Nd'][1] ,    0:self.st['Nd'][2]]
#         elif numpy.size(self.st['Nd']) == 1:
#             output_x[0:self.st['Nd'][0]] =input_x[0:self.st['Nd'][0] ]


#         k_gpu = pycuda.gpuarray.to_gpu(output_x)

#         if self.gpu_flag == 1:
#             output_gpu = pycuda.gpuarray.zeros_like(A_gpu)
        cu_fft.fft(k_gpu, k_gpu, self.scikit_plan, False)

        return k_gpu
    def k2vec(self, k):
#         k = cu_linalg.transpose(k)
        
        k_vec = k.reshape((numpy.prod(self.st['Kd']),))

        return k_vec
    
    def k2y(self, Xk_gpu):
        '''
        2D k-space grid to 1D array
        input:
            k:    k-space grid,
        output:
            y: non-Cartesian data
        '''
#         print(type(k))
#
#         Xk_gpu = pycuda.gpuarray.to_gpu(k.astype( dtype) )

        '''
        GPUarray "reshape" is in C-order. (row major matrix)
        Transposition before reshape 
        '''

#         Xk_gpu_tmp = cu_linalg.transpose(Xk_gpu)
# 
#         Xk_gpu = Xk_gpu_tmp.reshape((numpy.prod(self.st['Kd']),))
        Xk_gpu = self.k2vec(Xk_gpu)
#         y_gpu = cuda_cffi.cusparse.csrmv(self.descrA, self.csrValA, self.csrRowPtrA, self.csrColIndA,
#                   self.st['M'],numpy.prod(self.st['Kd']),
#                   Xk_gpu, transA=cuda_cffi.cusparse.CUSPARSE_OPERATION_NON_TRANSPOSE, alpha=1, beta=0)
#         y_gpu = self.CSR.mv(Xk_gpu)
        y_gpu = self.vec2y(Xk_gpu)
#         y_gpu2 = y_gpu.reshape( (self.st['M'],))
#         print(type(y_gpu), y_gpu.dtype)

#         y = y_gpu.get() #         y_gpu2 = y_gpu.reshape( (self.st['M'],))

#         y = numpy.reshape(self.st['p'].dot(Xk),(self.st['M'],),order='F')

        return y_gpu

    def vec2y(self, Xk_gpu):

        #         self.y_gpu = self.CSR.mv(Xk_gpu)
        #         m,n = self.csrshape
        #
        #         self.y_gpu = cuda_cffi.cusparse.csrmv(descrA, self.csrValA, self.csrRowPtrA, self.csrColIndA, m,
        # n, Xk_gpu,
        # transA=cuda_cffi.cusparse.CUSPARSE_OPERATION_NON_TRANSPOSE,
        # alpha=1.0, beta=0.0, y=self.y_gpu)

        y_gpu = self.CSR.mv(Xk_gpu)

        return y_gpu

    def y2vec(self, y_gpu):
        
#         k_gpu = self.CSR.mv(
#             y_gpu, transA=cuda_cffi.cusparse.CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE)
        k_gpu = self.CSRH.mv(y_gpu)
        return k_gpu
    def vec2k(self, k_vec):
        k_tmp = k_vec.reshape(self.st['Kd'])  # .transpose()
#         k = cu_linalg.transpose(k_tmp)
        return k_tmp
    def y2k(self, y):
        '''
        input:
            y:    non-Cartesian data
        output:
            k:    k-space data on Kd grid
        '''

        k_vec = self.y2vec(y)
#         k_tmp = k.reshape(self.st['Kd'])  # .transpose()
#         k = cu_linalg.transpose(k_tmp)
        k = self.vec2k(k_vec)
        return k

    def k2k(self, k_gpu):
        #         k = numpy.reshape(k, (numpy.prod(self.st['Kd']),),order='F')
        #         k_gpu = pycuda.gpuarray.to_gpu(k.astype(dtype))

#         k_gpu = cu_linalg.transpose(k_gpu)
#         k_gpu = k_gpu.reshape((numpy.prod(self.st['Kd']),))
        k_gpu2 = self.k2vec(k_gpu)
        k_gpu = self.st['pHp_gpu'].mv(k_gpu2)
#         y_gpu = self.CSR.mv(k_gpu2)
#         k_gpu = self.CSRH.mv(y_gpu)

#         print(k_gpu2.shape)
#         print(k_gpu2.shape)
#         k_tmp = k_gpu2.reshape(self.st['Kd'])  # .transpose()
#         k_gpu = cu_linalg.transpose(k_tmp)
        k_gpu2 = self.vec2k(k_gpu)
#         k=k_gpu.get()
#         k = numpy.reshape(k, self.st['Kd'], order='F')
        return k_gpu2
#     def kWk(self, k_gpu):
#         #         k = numpy.reshape(k, (numpy.prod(self.st['Kd']),),order='F')
#         #         k_gpu = pycuda.gpuarray.to_gpu(k.astype(dtype))
# 
#         k_gpu = cu_linalg.transpose(k_gpu)
#         k_gpu = k_gpu.reshape((numpy.prod(self.st['Kd']),))
# 
#         k_gpu2 = self.st['pHWp_gpu'].mv(k_gpu)
# 
# #         print(k_gpu2.shape)
# #         print(k_gpu2.shape)
#         k_tmp = k_gpu2.reshape(self.st['Kd'])  # .transpose()
#         k_gpu = cu_linalg.transpose(k_tmp)
# 
# #         k=k_gpu.get()
# #         k = numpy.reshape(k, self.st['Kd'], order='F')
#         return k_gpu

    def k2xx(self, k):

        dd = numpy.size(self.st['Kd'])

        cu_fft.ifft(k, k, self.scikit_plan, True)

        if dd == 2:
            '''
            Now remove zeros and crop the center from the zero padded image
            '''
#  
#             output_gpu_row = pycuda.gpuarray.zeros(
#                 shape=(self.st['Nd'][0], self.st['Kd'][1]), dtype=dtype, order='C')
            '''
            Allocate a new matrix, C-order
            '''

#             drv.memcpy_dtod(output_gpu_row.ptr, k.ptr, output_gpu_row.nbytes)
#   
#             output_gpu_row_T = cu_linalg.transpose(output_gpu_row)
#   
#             xx_gpu_T = pycuda.gpuarray.zeros(
#                 shape=(self.st['Nd'][1], self.st['Nd'][0]), dtype=dtype, order='C')
#   
#             drv.memcpy_dtod(xx_gpu_T.ptr, output_gpu_row_T.ptr,
#                             xx_gpu_T.nbytes)
#             xx_gpu = cu_linalg.transpose(xx_gpu_T)
# #             
            
            xx_gpu = pycuda.gpuarray.zeros(
                shape=(self.st['Nd'][0], self.st['Nd'][1]), dtype=dtype, order='C')
            if dtype == numpy.complex64:  
                gpuCrop2DS(xx_gpu, k ,self.st['Nd'][1], self.st['Kd'][1] ,numpy.prod(self.st['Nd']))
            elif dtype == numpy.complex128:
                gpuCrop2DZ(xx_gpu, k ,self.st['Nd'][1], self.st['Kd'][1] ,numpy.prod(self.st['Nd']))

        elif dd == 1:
            xx_gpu = pycuda.gpuarray.zeros(
                shape=(self.st['Nd'][0], ), dtype=dtype, order='C')
            '''
            Allocate a new matrix, C-order
            '''

#             xx_gpu_T = cu_linalg.transpose(xx_gpu)
#             input_gpu_T = cu_linalg.transpose(input_gpu)
            drv.memcpy_dtod(xx_gpu.ptr, k.ptr, xx_gpu.nbytes)

        elif dd == 3:
            
            xx_gpu = pycuda.gpuarray.zeros(
                shape=(self.st['Nd'][0], self.st['Nd'][1], self.st['Nd'][2]), dtype=dtype, order='C')
            if dtype == numpy.complex64:  
                gpuCrop3DS(xx_gpu, k ,self.st['Nd'][1], self.st['Kd'][1] , self.st['Nd'][2], self.st['Kd'][2] , numpy.prod(self.st['Nd']))
            elif dtype == numpy.complex128:
                gpuCrop3DZ(xx_gpu, k ,self.st['Nd'][1], self.st['Kd'][1] , self.st['Nd'][2], self.st['Kd'][2] ,numpy.prod(self.st['Nd']))

#         xx = xx_gpu.get()

        return xx_gpu

    def xx2x(self, xx):
        x = self.x2xx(xx)
        return x


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
    # load example image
#     image = numpy.loadtxt(DATA_PATH +'phantom_256_256.txt')
    image = scipy.misc.face(gray=True)
    
    image = scipy.misc.imresize(image, (256,256))
    
    image=image.astype(numpy.float)/numpy.max(image[...])
    #numpy.save('phantom_256_256',image)
    matplotlib.pyplot.imshow(image, cmap=matplotlib.cm.gray)
    matplotlib.pyplot.show()
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

    NufftObj = gpuNUFFT()

    NufftObj.plan(om, Nd, Kd, Jd)


#     NufftObj.initialize_gpu()

    y = nfft.k2y(nfft.xx2k(nfft.x2xx(image)))
    image_gpu = pycuda.gpuarray.to_gpu(image.astype(dtype))
    _y2 = NufftObj.k2y(NufftObj.xx2k(NufftObj.x2xx(image_gpu)))
#     y =  (nfft.xx2k(nfft.x2xx(image)))
#
#     _y2 = (NufftObj.xx2k(NufftObj.x2xx(image)))
    y2 = _y2.get()

    print('forward close? = ', numpy.allclose(y,y2,  atol=1e-0))

    x1 = nfft.xx2x(nfft.k2xx(nfft.y2k(y2)))
    x2 = NufftObj.xx2x(NufftObj.k2xx(NufftObj.y2k(_y2)))

    print('adjoint close? = ', numpy.allclose(x1,x2.get(),   atol=1e-1))

    import time
    t0 = time.time()
    for pp in range(4):
        image2 = nfft.selfadjoint((image))
    t1 = (time.time() - t0) / 4.0

#     image2 = NufftObj.xx2x(NufftObj.k2xx(NufftObj.y2k(y)))

    t0 = time.time()
    for pp in range(200):
#         image3_gpu = NufftObj.xx2x(NufftObj.k2xx(NufftObj.y2k(
#             NufftObj.st['W_gpu']*_y2)))
        image3_gpu =  NufftObj.selfadjoint(image_gpu)
    t2 = (time.time() - t0) / 200.0
    image3 = image3_gpu.get()
    print('CPU', t1)
    print('GPU', t2)

    print(numpy.shape(image2))
    matplotlib.pyplot.subplot(1, 2, 1)
    matplotlib.pyplot.imshow(x1.real)
    matplotlib.pyplot.subplot(1, 2, 2)
    matplotlib.pyplot.imshow(image3.real)
    matplotlib.pyplot.show()

if __name__ == '__main__':
    import cProfile
#    test_init()
    cProfile.run('test_init()')
