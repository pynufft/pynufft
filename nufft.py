"""
Main classes (NUFFT_cpu, NUFFT_hsa)
=================================
"""
from __future__ import absolute_import
import numpy
import scipy.sparse
import numpy.fft
import scipy.signal
import scipy.linalg
import scipy.special

# from .... import src
import sys
if sys.version_info[0] == 3:
    try:
        from .src._helper import helper
    except:
        from pynufft.src._helper import helper
        
if sys.version_info[0] == 2:
    
#     import os, sys
#     sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    from src._helper import helper
class NUFFT_cpu:
    """
    Class NUFFT_cpu
   """    
    def __init__(self):
        """
        Constructor.
        
        :param None:
        :type None: Python NoneType
        :return: NUFFT: the pynufft_hsa.NUFFT instance
        :rtype: NUFFT: the pynufft_hsa.NUFFT class
        :Example:

        >>> import pynufft
        >>> NufftObj = pynufft.NUFFT_cpu()


        .. note:: requires plan() 
        .. seealso:: :method:`plan()' 
        .. todo:: test 3D case
        """        
        self.dtype=numpy.complex64
        pass

    def plan(self, om, Nd, Kd, Jd):
        """
        Design the min-max interpolator.
        
        :param om: The M off-grid locations in the frequency domain. Normalized between [-pi, pi]
        :param Nd: The matrix size of equispaced image. Example: Nd=(256,256) for a 2D image; Nd = (128,128,128) for a 3D image
        :param Kd: The matrix size of the oversampled frequency grid. Example: Kd=(512,512) for 2D image; Kd = (256,256,256) for a 3D image
        :param Jd: The interpolator size. Example: Jd=(6,6) for 2D image; Jd = (6,6,6) for a 3D image
        :type om: numpy.float array, matrix size = M * ndims
        :type Nd: tuple, ndims integer elements. 
        :type Kd: tuple, ndims integer elements. 
        :type Jd: tuple, ndims integer elements. 
        :returns: 0
        :rtype: int, float
        :Example:

        >>> import pynufft
        >>> NufftObj = pynufft.NUFFT_cpu()
        >>> NufftObj.plan(om, Nd, Kd, Jd) 
        
        """         
        self.debug = 0  # debug

        n_shift = tuple(0*x for x in Nd)
        self.st = helper.plan(om, Nd, Kd, Jd)
        
        self.Nd = self.st['Nd']  # backup
        self.Kd = self.st['Kd']
        self.sn = numpy.asarray(self.st['sn'].astype(self.dtype)  ,order='C')# backup
        self.ndims = len(self.Nd) # dimension
        self._linear_phase(n_shift)  # calculate the linear phase thing
        
        # Calculate the density compensation function
        self._precompute_sp()
        del self.st['p'], self.st['sn']
        del self.st['p0'] 
        self.NdCPUorder, self.KdCPUorder, self.nelem =     helper.preindex_copy(self.st['Nd'], self.st['Kd'])
        return 0
        
#         print('untrimmed',self.st['pHp'].nnz)
#         self.truncate_selfadjoint(1e-1)
#         print('trimmed', self.st['pHp'].nnz)
 
    def _precompute_sp(self):
        """

        Private: Precompute adjoint (gridding) and Toepitz interpolation matrix.
        
        :param None: 
        :type None: Python Nonetype
        :return: self: instance
        """
        try:
            self.sp = self.st['p']
            self.spH = (self.st['p'].getH().copy()).tocsr()
            self.spHsp =self.st['p'].getH().dot(self.st['p']).tocsr()
        except:
            print("errors occur in self.precompute_sp()")
            raise
#         self.truncate_selfadjoint( 1e-2)

    def _matvec(self, x_vec):
        """
        (To be tested): dot operation provided for scipy.sparse.linalg
        wrapper of self.forward()
        """
        
        x2 = numpy.reshape(x_vec, self.Nd, order='F')
        
        return self.forward(x2) 

    def solve(self,y, solver=None, *args, **kwargs):
        """
        Solve NUFFT_cpu
        
        :param y: data, numpy array, (M,) size
        :param solver: could be 'cg', 'L1TVOLS', 'L1TVLAD' 
        :param maxiter: the number of iterations
        :type y: numpy array, dtype = numpy.complex64
        :type solver: string
        :type maxiter: int
        :return: numpy array with size Nd
        """        
        from .src._nonlin.solve_cpu import solve
        return solve(self,  y,  solver, *args, **kwargs)

    
    def _pipe_density(self,maxiter):
        '''
        Create the density function by iterative solution
        Generate pHp matrix
        '''
#         W = pipe_density(self.st['p'])
        # sampling density function

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
       
                   
            W = numpy.ones((self.st['M'],),dtype=self.dtype)
    #         V1= self.st['p'].getH()
        #     VVH = V.dot(V.getH()) 
            
            for pp in range(0,maxiter):
    #             E = self.st['p'].dot(V1.dot(W))
 
                E = self.forward(self.adjoint(W))
                W = (W/E)

            self.last_iter = maxiter
        
        return W
        # density of the k-space, reduced size

    def _linear_phase(self, n_shift):
        """
        Private: Select the center of FOV
        """
        om = self.st['om']
        M = self.st['M']
        final_shifts = tuple(
            numpy.array(n_shift) +
            numpy.array(
                self.Nd) /
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

        self.st['p'] = scipy.sparse.diags(phase, 0).dot(self.st['p0'])
        return 0  # shifted sparse matrix
    def truncate_selfadjoint(self, tol):
        """
        Yet to be tested.
        """   
        indix=numpy.abs(self.st['pHp'].data)< tol
        self.st['pHp'].data[indix]=0

        self.st['pHp'].eliminate_zeros()

    def forward(self, x):
        """
        Forward NUFFT on CPU
        
        :param x: The input numpy array, with size=Nd
        :type: numpy array with dtype =numpy.complex64
        :return: gy: The output numpy array, with size=(M,)
        :rtype: numpy array with dtype =numpy.complex64
        """
        y = self.k2y(self.xx2k(self.x2xx(x)))

        return y

    def adjoint(self, y):
        """
        Adjoint NUFFT on CPU
        
        :param y: The input numpy array, with size=(M,)
        :type: numpy array with dtype =numpy.complex64
        :return: x: The output numpy array, with size=Nd
        :rtype: numpy array with dtype =numpy.complex64
        """     
        x = self.xx2x(self.k2xx(self.y2k(y)))

        return x

    def selfadjoint(self, x):
        """
        selfadjoint NUFFT (Teplitz) on CPU
        
        :param x: The input numpy array, with size=Nd
        :type: numpy array with dtype =numpy.complex64
        :return: x: The output numpy array, with size=Nd
        :rtype: numpy array with dtype =numpy.complex64
        """       
#         x2 = self.adjoint(self.forward(x))
        
        x2 = self.xx2x(self.k2xx(self.k2y2k(self.xx2k(self.x2xx(x)))))
#         x2 = self.k2xx(self.k2y2k(self.xx2k(x)))
        
        return x2

    def x2xx(self, x):
        """
        Private: Scaling on CPU
        Inplace multiplication of self.x_Nd by the scaling factor self.sn.
        """    
        xx = x * self.sn
        return xx

    def xx2k(self, xx):
        """
        Private: oversampled FFT on CPU
        
        Firstly, zeroing the self.k_Kd array
        Second, copy self.x_Nd array to self.k_Kd array by cSelect
        Third: inplace FFT
        """
#         dd = numpy.size(self.Kd)      
        output_x = numpy.zeros(self.Kd, dtype=self.dtype,order='C')
#         output_x[crop_slice_ind(xx.shape)] = xx
        output_x.flat[self.KdCPUorder]=xx.flat[self.NdCPUorder]
        k = numpy.fft.fftn(output_x, self.Kd, range(0, self.ndims))

        return k
    def k2vec(self,k):
        
        k_vec = numpy.reshape(k, (numpy.prod(self.Kd), ), order='C')
   
        return k_vec
    
    def vec2y(self,k_vec):
        '''
        gridding: 
        
        '''
        y = self.sp.dot(k_vec)
        
        return y
    def k2y(self, k):
        """
        Private: interpolation by the Sparse Matrix-Vector Multiplication
        """
        
        y = self.vec2y(self.k2vec(k)) #numpy.reshape(self.st['p'].dot(Xk), (self.st['M'], ), order='F')
        
        return y
    def y2vec(self, y):
        '''
       regridding non-uniform data, (unsorted vector)
        '''
#         k_vec = self.st['p'].getH().dot(y)
        k_vec = self.spH.dot(y)
        
        return k_vec
    def vec2k(self, k_vec):
        '''
        Sorting the vector to k-spectrum Kd array
        '''
        k = numpy.reshape(k_vec, self.Kd, order='C')
        
        return k
    
    def y2k(self, y):
        """
        Private: gridding by the Sparse Matrix-Vector Multiplication
        """
        
        k_vec = self.y2vec(y)
        
        k = self.vec2k(k_vec)
        
        return k

    def k2xx(self, k):
        """
        Private: the inverse FFT and image cropping (which is the reverse of _xx2k() method)
        """
#         dd = numpy.size(self.Kd)
        
        k = numpy.fft.ifftn(k, self.Kd, range(0, self.ndims))
        xx= numpy.zeros(self.Nd,dtype=self.dtype, order='C')
        xx.flat[self.NdCPUorder]=k.flat[self.KdCPUorder]
#         xx = xx[crop_slice_ind(self.Nd)]
        return xx

    def xx2x(self, xx):
        """
        Private: rescaling, which is identical to the  _x2xx() method
        """
        x = self.x2xx(xx)
        return x
    def k2y2k(self, k):
        """
        Private: the integrated interpolation-gridding by the Sparse Matrix-Vector Multiplication
        """

        Xk = self.k2vec(k)
         
        k = self.spHsp.dot(Xk)
        k = self.vec2k(k)
        return k


class NUFFT_hsa(NUFFT_cpu):
    """
    Class NUFFT_hsa for heterogeneous systems.
   """

    def __init__(self):
        """
        Constructor.
        
        :param None:
        :type None: Python NoneType
        :return: NUFFT: the pynufft_hsa.NUFFT instance
        :rtype: NUFFT: the pynufft_hsa.NUFFT class
        :Example:

        >>> import pynufft
        >>> NufftObj = pynufft.NUFFT_hsa()


        .. note:: requires plan() and offload()
        .. seealso:: :method:`plan()' 'offload()'
        .. todo:: test 3D case
        """
        
        pass
        NUFFT_cpu.__init__(self)


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
        if sys.version_info[0] == 3:
            from .src.re_subroutine import cMultiplyScalar, cCopy, cAddScalar,cAddVec, cSparseMatVec, cSelect, cMultiplyVec, cMultiplyVecInplace, cMultiplyConjVec, cDiff, cSqrt, cAnisoShrink
        if sys.version_info[0] == 2:
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

#         self.xx_Kd = pyopencl.array.zeros(self.queue, self.st['Kd'], dtype=self.dtype, order="C")
        self.k_Kd = self.thr.to_device(numpy.zeros(self.st['Kd'], dtype=self.dtype, order="C"))
        self.k_Kd2 = self.thr.to_device(numpy.zeros(self.st['Kd'], dtype=self.dtype, order="C"))
        self.y =self.thr.to_device( numpy.zeros((self.st['M'],), dtype=self.dtype, order="C"))
        self.x_Nd = self.thr.to_device(numpy.zeros(self.st['Nd'], dtype=self.dtype, order="C"))
#         self.xx_Nd =     pyopencl.array.zeros(self.queue, self.st['Nd'], dtype=self.dtype, order="C")

#         self.NdCPUorder, self.KdCPUorder, self.nelem =     preindex_copy(self.st['Nd'], self.st['Kd'])
        self.NdGPUorder = self.thr.to_device( self.NdCPUorder)
        self.KdGPUorder =  self.thr.to_device( self.KdCPUorder)
        self.Ndprod = numpy.int32(numpy.prod(self.st['Nd']))
        self.Kdprod = numpy.int32(numpy.prod(self.st['Kd']))
        self.M = numpy.int32( self.st['M'])
        
        self.SnGPUArray = self.thr.to_device(  self.sn)
        
        self.sp_data = self.thr.to_device( self.sp.data.astype(self.dtype))
        self.sp_indices =self.thr.to_device( self.sp.indices.astype(numpy.int32))
        self.sp_indptr = self.thr.to_device( self.sp.indptr.astype(numpy.int32))
        self.sp_numrow =  self.M
        del self.sp
        self.spH_data = self.thr.to_device(  self.spH.data.astype(self.dtype))
        self.spH_indices = self.thr.to_device(  self.spH.indices)
        self.spH_indptr = self.thr.to_device(  self.spH.indptr)
        self.spH_numrow = self.Kdprod
        del self.spH
        self.spHsp_data = self.thr.to_device(  self.spHsp.data.astype(self.dtype))
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
        self.zero_scalar=self.dtype(0.0+0.0j)
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
        from .src._nonlin.solve_hsa import solve
        
            
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
    
