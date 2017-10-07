"""
Class NUFFT on heterogeneous platforms
=================================
"""

# '''
# @package docstring
# 
# Copyright (c) 2014-2017 Pynufft team.
# All rights reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer. Redistributions in binary
# form must reproduce the above copyright notice, this list of conditions and
# the following disclaimer in the documentation and/or other materials provided
# with the distribution. Neither the name of Enthought nor the names of the
# Pynufft Developers may be used to endorse or promote products derived from
# this software without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 'AS IS'
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# '''

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
except:
    api = cluda.cuda_api()
# api = cluda.cuda_api()
try:
    platform = api.get_platforms()[0]

except:
    platform = api.get_platforms()[0]

device = platform.get_devices()[0]
print('device = ', device)
# print('using cl device=',device,device[0].max_work_group_size, device[0].max_compute_units,pyopencl.characterize.get_simd_group_size(device[0], dtype.size))

from src._helper.helper import *


class NUFFT:
#     """
#     The class pynufft computes Non-Uniform Fast Fourier Transform (NUFFT).
# 
#     Methods
#     ----------
#     __init__() : constructor
#         Input: 
#             None
#         Return: 
#             pynufft instance
#         Example: MyNufft = pynufft.pynufft()
#     plan(om, Nd, Kd, Jd) : to plan the pynufft object
#         Input:
#             om: M * ndims array: The locations of M non-uniform points in the ndims dimension. Normalized between [-pi, pi]
#             Nd: tuple with ndims elements. Image matrix size. Example: (256,256)
#             Kd: tuple with ndims elements. Oversampling k-space matrix size. Example: (512,512)
#             Jd: tuple with ndims elements. The number of adjacent points in the interpolator. Example: (6,6)
#         Return:
#             None
#     forward(x) : perform NUFFT
#         Input:
#             x: numpy.array. The input image on the regular grid. The size must be Nd. 
#         Output:
#             y: M array.The output M points array.
#              
#     adjoint(y): adjoint NUFFT (Hermitian transpose (a.k.a. conjugate transpose) of NUFFT)
#         Input:
#             y: M array.The input M points array.
#         Output:
#             x: numpy.array. The output image on the regular grid.
#             
#     inverse_DC(y) deprecate: inverse NUFFT using Pipe's sampling density compensation (James Pipe, Magn. Res. Med., 1999)
#         Input: 
#             y: M array.The input M points array.
#         Output:
#             x: numpy.array. The output image on the regular grid.
# 
#     """
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

        n_shift = tuple(0*x for x in Nd)
        self.st = plan(om, Nd, Kd, Jd)
        
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
        self.reikna():
        
        Off-load NUFFT to the opencl or cuda device(s)
        """
        
#         Create context from device
        self.thr = api.Thread(device) #pyopencl.create_some_context()
#         self.queue = pyopencl.CommandQueue( self.ctx)

#         """
#         Wavefront: as warp in cuda. Can control the width in a workgroup
#         Wavefront is required in spmv_vector as it improves data coalescence.
#         see cSparseMatVec and zSparseMatVec
#         """
        self.wavefront = api.DeviceParameters(device).warp_size
        print(api.DeviceParameters(device).max_work_group_size)
#         print(self.wavefront)
#         print(type(self.wavefront))
#          pyopencl.characterize.get_simd_group_size(device[0], dtype.size)
        from src.re_subroutine import cMultiplyScalar, cCopy, cAddScalar,cAddVec, cSparseMatVec, cSelect, cMultiplyVec, cMultiplyVecInplace, cMultiplyConjVec, cDiff, cSqrt, cAnisoShrink
        # import complex float routines
#         print(dtypes.ctype(dtype))
        prg = self.thr.compile( 
                                cMultiplyScalar.R + #cCopy.R, 
                                cCopy.R + 
                                cAddScalar.R + 
                                cSelect.R +cMultiplyConjVec.R + cAddVec.R+
                                cMultiplyVecInplace.R +cSparseMatVec.R+cDiff.R+ cSqrt.R+ cAnisoShrink.R+ cMultiplyVec.R,
                                render_kwds=dict(
                                    LL =  str(self.wavefront)), fast_math=False)
#                                fast_math = False)
#                                 "#define LL  "+ str(self.wavefront) + "   "+cSparseMatVec.R)
#                                ),
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
        import src._solver.solver_hsa
        return src._solver.solver_hsa.solver(self,  y,  solver, *args, **kwargs)

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
def benchmark():
    import cProfile
    import numpy
    #import matplotlib.pyplot
    import copy

    #cm = matplotlib.cm.gray
    # load example image
    import pkg_resources
    
    DATA_PATH = pkg_resources.resource_filename('pynufft', 'data/')
#     PHANTOM_FILE = pkg_resources.resource_filename('pynufft', 'data/phantom_256_256.txt')
    import numpy
    #import matplotlib.pyplot
    import scipy
    import scipy.misc
    # load example image
#     image = numpy.loadtxt(DATA_PATH +'phantom_256_256.txt')
    image = scipy.misc.face(gray=True)
#    image = scipy.misc.ascent()    
    image = scipy.misc.imresize(image, (256,256))
    
    image=image.astype(numpy.float)/numpy.max(image[...])
    #numpy.save('phantom_256_256',image)
    #matplotlib.pyplot.subplot(1,3,1)
    #matplotlib.pyplot.imshow(image, cmap=matplotlib.cm.gray)
    #matplotlib.pyplot.title("Load Scipy \"ascent\" image")
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
    print("create NUFFT gpu object")
    NufftObj = NUFFT()
    print("plan nufft on gpu")
    NufftObj.plan(om, Nd, Kd, Jd)
    print("NufftObj planed")
#     print('sp close? = ', numpy.allclose( nfft.st['p'].data,  NufftObj.st['p'].data, atol=1e-1))
#     NufftObj.initialize_gpu()

    y = nfft.k2y(nfft.xx2k(nfft.x2xx(image)))
    print("send image to device")
    NufftObj.x_Nd = NufftObj.thr.to_device( image.astype(dtype))
    print("copy image to gx")
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
    for pp in range(0,20):
#         pass
#         NufftObj.adjoint()
        gy=NufftObj.forward(gx)        

#     NufftObj.thr.synchronize()
    t_cl = (time.time() - t0)/20
    print(t_cl)
    print("forward acceleration=", t_cpu/t_cl)

    t0 = time.time()
    for pp in range(0,10):
#         y = nfft.k2y(nfft.xx2k(nfft.x2xx(image)))    
            x = nfft.adjoint(y)
#             y = nfft.forward(image)
#     y2 = NufftObj.y.get(   NufftObj.queue, async=False)
    t_cpu = (time.time() - t0)/10.0 
    print(t_cpu)
    
#     del nfft

    
    t0= time.time()
    for pp in range(0,20):
#         pass
#         NufftObj.adjoint()
        gx=NufftObj.adjoint(gy)        

#     NufftObj.thr.synchronize()
    t_cl = (time.time() - t0)/20
    print(t_cl)
    print("adjoint acceleration=", t_cpu/t_cl)

    t0 = time.time()
    for pp in range(0,10):
#         y = nfft.k2y(nfft.xx2k(nfft.x2xx(image)))    
#             x = nfft.adjoint(y)
            x = nfft.selfadjoint(image)
#     y2 = NufftObj.y.get(   NufftObj.queue, async=False)
    t_cpu = (time.time() - t0)/10.0 
    print(t_cpu)
    
   
#     del nfft

    
    t0= time.time()
    for pp in range(0,20):
#         pass
#         NufftObj.adjoint()
        g2=NufftObj.selfadjoint(gx)        

#     NufftObj.thr.synchronize()
    t_cl = (time.time() - t0)/20
    print(t_cl)
    print("selfadjoint acceleration=", t_cpu/t_cl)



    maxiter = 100
    import time
    t0= time.time()
    x2 = nfft.solver(y2, 'cg',maxiter=maxiter)
#    x2 =  nfft.solver(y2, 'L1LAD',maxiter=maxiter, rho = 1)
    t1 = time.time()-t0
#     gy=NufftObj.thr.copy_array(NufftObj.thr.to_device(y2))
    
    t0= time.time()
    x = NufftObj.solver(gy,'cg', maxiter=maxiter)
#    x = NufftObj.solver(gy,'L1LAD', maxiter=maxiter, rho=1)
    
    t2 = time.time() - t0
    print(t1, t2)
    print('acceleration=', t1/t2 )

    t0= time.time()
#     x = NufftObj.solver(gy,'cg', maxiter=maxiter)
    x = NufftObj.solver(gy,'L1OLS', maxiter=maxiter, rho=2)

    
    t3 = time.time() - t0
    print(t2, t3)
    print('Speed of LAD vs OLS =', t3/t2 )


#     k = x.get()
#     x = nfft.k2xx(k)/nfft.st['sn']
#     return
    
    #matplotlib.pyplot.subplot(1, 3, 2)
    #matplotlib.pyplot.imshow( NufftObj.x_Nd.get().real, cmap= matplotlib.cm.gray)
    #matplotlib.pyplot.subplot(1, 3,3)
    #matplotlib.pyplot.imshow(x2.real, cmap= matplotlib.cm.gray)
    #matplotlib.pyplot.show()


def test_init():
    import cProfile
    import numpy
    import matplotlib.pyplot
    import copy

    cm = matplotlib.cm.gray
    # load example image
    import pkg_resources
    
    DATA_PATH = pkg_resources.resource_filename('pynufft', 'src/data/')
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
    x2 =  nfft.solver(y2, 'L1LAD',maxiter=maxiter, rho = 2)
    t1 = time.time()-t0
#     gy=NufftObj.thr.copy_array(NufftObj.thr.to_device(y2))
    
    t0= time.time()
#     x = NufftObj.solver(gy,'cg', maxiter=maxiter)
    x = NufftObj.solver(gy,'L1LAD', maxiter=maxiter, rho=2)
    
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
#     cProfile.run('benchmark()')
    test_init()
#     test_cAddScalar()
#     cProfile.run('test_init()')
