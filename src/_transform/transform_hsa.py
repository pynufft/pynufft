"""
Class NUFFT on heterogeneous platforms
==================================================================
"""
from __future__ import division

from numpy.testing import (run_module_suite, assert_raises, assert_equal,
                           assert_almost_equal)

from unittest import skipIf
import numpy

import scipy.sparse  # TODO: refactor to remove this

from scipy.sparse.csgraph import _validation  # for cx_freeze debug

import numpy.fft
 
import scipy.linalg

dtype = numpy.complex64

from .._helper.helper import *
from .._transform.transform_cpu import NUFFT as NUFFT_c 
# import transform_cpu.NUFFT as NUFFT_cpu

class NUFFT(NUFFT_c):
    """
    The class NUFFT belongs to pynufft_hsa, which offloads Non-Uniform Fast Fourier Transform (NUFFT) to heterogeneous devices.
   """

    def __init__(self):
        """
        Constructor.
        
        :param None:
        :type None: Python NoneType
        :return: NUFFT: the pynufft_hsa.NUFFT instance
        :rtype: NUFFT: the pynufft_hsa.NUFFT class
        :Example:

        >>> import pynufft.pynufft
        >>> NufftObj = pynufft.pynufft.NUFFT_hsa()


        .. note:: requires plan() and offload()
        .. seealso:: :method:`plan()' 'offload()'
        .. todo:: test 3D case
        """
        
        pass
        NUFFT_c.__init__(self)


#     def plan(self, om, Nd, Kd, Jd):
#         """
#         Design the min-max interpolator.
#          
#         :param om: The M off-grid locations in the frequency domain. Normalized between [-pi, pi]
#         :param Nd: The matrix size of equispaced image. Example: Nd=(256,256) for a 2D image; Nd = (128,128,128) for a 3D image
#         :param Kd: The matrix size of the oversampled frequency grid. Example: Kd=(512,512) for 2D image; Kd = (256,256,256) for a 3D image
#         :param Jd: The interpolator size. Example: Jd=(6,6) for 2D image; Jd = (6,6,6) for a 3D image
#         :type om: numpy.float array, matrix size = M * ndims
#         :type Nd: tuple, ndims integer elements. 
#         :type Kd: tuple, ndims integer elements. 
#         :type Jd: tuple, ndims integer elements. 
#         :returns: 0
#         :rtype: int, float
#         :Example:
#  
#         >>> import pynufft
#         >>> NufftObj = pynufft_hsa.NUFFT()
#         >>> NufftObj.plan(om, Nd, Kd, Jd) 
#          
#         """         
#         self.debug = 0  # debug
#  
#         n_shift = tuple(0*x for x in Nd)
#         self.st = plan(om, Nd, Kd, Jd)
#          
#         self.Nd = self.st['Nd']  # backup
#         self.sn = numpy.asarray(self.st['sn'].astype(dtype)  ,order='C')# backup
#         self.ndims = len(self.st['Nd']) # dimension
#         self._linear_phase(n_shift)  # calculate the linear phase thing
#          
#         # Calculate the density compensation function
#         self._precompute_sp()
#         del self.st['p'], self.st['sn']
#         del self.st['p0'] 
#         self.NdCPUorder, self.KdCPUorder, self.nelem =     preindex_copy(self.st['Nd'], self.st['Kd'])
#  
#         return 0

#     def _precompute_sp(self):
#         """
# 
#         Private: Precompute adjoint (gridding) and Toepitz interpolation matrix.
#         
#         :param None: 
#         :type None: Python Nonetype
#         :return: self: instance
#         """
#         try:
#             self.sp = self.st['p']
#             self.spH = (self.st['p'].getH().copy()).tocsr()
#             self.spHsp =self.st['p'].getH().dot(self.st['p']).tocsr()
#         except:
#             print("errors occur in self.precompute_sp()")
#             raise
# #         self.truncate_selfadjoint( 1e-2)

    def offload(self, API, platform_number=0, device_number=0):
        """
        self.offload():
        
        Off-load NUFFT to the opencl or cuda device(s)
        
        :param API: define the device type, which can be 'cuda' or 'ocl'
        :param platform_number: define which platform to be used. The default platform_number = 0.
        :param device_number: define which device to be used. The default device_number = 0.
        :type API: string
        :type platform_number: int
        :type device_number: int
        :return: self: instance

        """
        from reikna import cluda
        import reikna.transformations
        from reikna.cluda import functions, dtypes
        try: # try to create api/platform/device using the given parameters
            if 'cuda' == API:
                api = cluda.cuda_api()
            elif 'ocl' == API:
                api = cluda.ocl_api()
     
            platform = api.get_platforms()[platform_number]
            
            device = platform.get_devices()[device_number]
        except: # if failed, find out what's going wrong?
            diagnose()
            
            return 1

        
#         print('device = ', device)
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
        from ..re_subroutine import cMultiplyScalar, cCopy, cAddScalar,cAddVec, cSparseMatVec, cSelect, cMultiplyVec, cMultiplyVecInplace, cMultiplyConjVec, cDiff, cSqrt, cAnisoShrink
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

#         self.NdCPUorder, self.KdCPUorder, self.nelem =     preindex_copy(self.st['Nd'], self.st['Kd'])
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
    def solve(self,gy, solver=None, *args, **kwargs):
        """
        The solver of NUFFT_hsa
        
        :param gy: data, reikna array, (M,) size
        :param solver: could be 'cg', 'L1TVOLS', 'L1TVLAD' 
        :param maxiter: the number of iterations
        :type gy: reikna array, dtype = numpy.complex64
        :type solver: string
        :type maxiter: int
        :return: reikna array with size Nd
        """
        from .._nonlin.solve_hsa import solve
        
            
        try:
            return solve(self,  gy,  solver, *args, **kwargs)
        except:
            if numpy.ndarray==type(gy):
                print("input gy must be a reikna array with dtype = numpy.complex64")
                raise TypeError
            else:
                print("wrong")
                raise TypeError

    def _pipe_density(self,maxiter):
        """
        Private: create the density function by iterative solution
        Generate pHp matrix
        """


        try:
            if maxiter < self.last_iter:
            
                W = self.st['W']
            else: #maxiter > self.last_iter
                W = self.st['W']
                for pp in range(0,maxiter - self.last_iter):
 
    #             E = self.st['p'].dot(V1.dot(W))
                    E = self.forward(self.adjoint(W))
                    W = (W/E)             
                self.last_iter = maxiter   
        except:
            W = self.thr.copy_array(self.y)
            self.cMultiplyScalar(self.zero_scalar, W, local_size=None, global_size=int(self.M))
    #         V1= self.st['p'].getH()
        #     VVH = V.dot(V.getH()) 
            
            for pp in range(0,1):
    #             E = self.st['p'].dot(V1.dot(W))
 
                E = self.forward(self.adjoint(W))
                W /= E
#                 self.cMultiplyVecInplace(self.SnGPUArray, self.x_Nd, local_size=None, global_size=int(self.Ndprod))


            self.last_iter = maxiter
        
        return W    
    def _linear_phase(self, n_shift):
        """
        Private: Select the center of FOV
        """
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
 

    def _truncate_selfadjoint(self, tol):
        """
        Yet to be tested.
        """
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
            """
            Forward NUFFT on the heterogeneous device
            
            :param gx: The input gpu array, with size=Nd
            :type: reikna gpu array with dtype =numpy.complex64
            :return: gy: The output gpu array, with size=(M,)
            :rtype: reikna gpu array with dtype =numpy.complex64
            """
            self.x_Nd =  self.thr.copy_array(gx)
            
            self._x2xx()

            self._xx2k()

            self._k2y()

            gy =  self.thr.copy_array(self.y)
            return gy
    
    def adjoint(self, gy):
            """
            Adjoint NUFFT on the heterogeneous device
            
            :param gy: The input gpu array, with size=(M,)
            :type: reikna gpu array with dtype =numpy.complex64
            :return: gx: The output gpu array, with size=Nd
            :rtype: reikna gpu array with dtype =numpy.complex64
            """        
            self.y = self.thr.copy_array(gy) 

            self._y2k()
            self._k2xx()
            self._xx2x()
            gx = self.thr.copy_array(self.x_Nd)
            return gx
    def selfadjoint(self, gx):
        """
        selfadjoint NUFFT (Teplitz) on the heterogeneous device
        
        :param gx: The input gpu array, with size=Nd
        :type: reikna gpu array with dtype =numpy.complex64
        :return: gx: The output gpu array, with size=Nd
        :rtype: reikna gpu array with dtype =numpy.complex64
        """                
        self.x_Nd = self.thr.copy_array(gx)
        self._x2xx()
        self._xx2k()
        self._k2y2k()
        self._k2xx()
        self._xx2x()
        gx2 = self.thr.copy_array(self.x_Nd)
        return gx2

    def _x2xx(self):
        """
        Private: Scaling on the heterogeneous device
        Inplace multiplication of self.x_Nd by the scaling factor self.SnGPUArray.
        """                
#         self.cMultiplyVecInplace(self.queue, (self.Ndprod,), None,  self.SnGPUArray.data, self.x_Nd.data)
        self.cMultiplyVecInplace(self.SnGPUArray, self.x_Nd, local_size=None, global_size=int(self.Ndprod))
        self.thr.synchronize()
    def _xx2k(self ):
        
        """
        Private: oversampled FFT on the heterogeneous device
        
        Firstly, zeroing the self.k_Kd array
        Second, copy self.x_Nd array to self.k_Kd array by cSelect
        Third: inplace FFT
        """
        
        self.cMultiplyScalar(self.zero_scalar, self.k_Kd, local_size=None, global_size=int(self.Kdprod))
        self.cSelect(self.NdGPUorder,      self.KdGPUorder,  self.x_Nd, self.k_Kd, local_size=None, global_size=int(self.Ndprod))
        self.fft( self.k_Kd,self.k_Kd,inverse=False)
        self.thr.synchronize()
    def _k2y(self ):
        """
        Private: interpolation by the Sparse Matrix-Vector Multiplication
        """
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
        """
        Private: gridding by the Sparse Matrix-Vector Multiplication
        """
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
        """
        Private: the integrated interpolation-gridding by the Sparse Matrix-Vector Multiplication
        """
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
        """
        Private: the inverse FFT and image cropping (which is the reverse of _xx2k() method)
        """
        self.fft( self.k_Kd2, self.k_Kd2,inverse=True)
#         self.x_Nd._zero_fill()
        self.cMultiplyScalar(self.zero_scalar, self.x_Nd,  local_size=None, global_size=int(self.Ndprod ))
#         self.cSelect(self.queue, (self.Ndprod,), None,   self.KdGPUorder.data,  self.NdGPUorder.data,     self.k_Kd2.data, self.x_Nd.data )
        self.cSelect(  self.KdGPUorder,  self.NdGPUorder,     self.k_Kd2, self.x_Nd, local_size=None, global_size=int(self.Ndprod ))
        
        self.thr.synchronize()
    def _xx2x(self ):
        """
        Private: rescaling, which is identical to the  _x2xx() method
        """
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
    import transform_cpu as pynufft
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
#    x2 =  nfft.solver(y2, 'L1TVLAD',maxiter=maxiter, rho = 1)
    t1 = time.time()-t0
#     gy=NufftObj.thr.copy_array(NufftObj.thr.to_device(y2))
    
    t0= time.time()
    x = NufftObj.solver(gy,'cg', maxiter=maxiter)
#    x = NufftObj.solver(gy,'L1TVLAD', maxiter=maxiter, rho=1)
    
    t2 = time.time() - t0
    print(t1, t2)
    print('acceleration=', t1/t2 )

    t0= time.time()
#     x = NufftObj.solver(gy,'cg', maxiter=maxiter)
    x = NufftObj.solver(gy,'L1TVOLS', maxiter=maxiter, rho=2)

    
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
#     from transform_cpu import NUFFT as NUFFT_c
    nfft = NUFFT_c()  # CPU
    nfft.plan(om, Nd, Kd, Jd)
#     nfft.initialize_gpu()
    import scipy.sparse
#     scipy.sparse.save_npz('tests/test.npz', nfft.st['p'])

    NufftObj = NUFFT()

    NufftObj.plan(om, Nd, Kd, Jd)
    NufftObj.offload(API='ocl',   platform_number= 0 , device_number= 0)
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
    maxiter =100
    import time
    t0= time.time()
#     x2 = nfft.solver(y2, 'cg',maxiter=maxiter)
    x2 =  nfft.solver(y2, 'L1TVLAD',maxiter=maxiter, rho = 2)
    t1 = time.time()-t0
#     gy=NufftObj.thr.copy_array(NufftObj.thr.to_device(y2))
    
    t0= time.time()
#     x = NufftObj.solver(gy,'dc', maxiter=maxiter)
    x = NufftObj.solver(gy,'L1TVLAD', maxiter=maxiter, rho=2)
    
    t2 = time.time() - t0
    print(t1, t2)
    print('acceleration=', t1/t2 )
#     k = x.get()
#     x = nfft.k2xx(k)/nfft.st['sn']
#     return
    
    matplotlib.pyplot.subplot(1, 3, 2)
    matplotlib.pyplot.imshow( x.get().real, cmap= matplotlib.cm.gray)
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
    
# if __name__ == '__main__':
#     import cProfile
# #     cProfile.run('benchmark()')
#     test_init()
#     test_cAddScalar()
#     cProfile.run('test_init()')
