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
# import sys
# if sys.version_info[0] == 3:
#     try:
from .src._helper import helper

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
        self.debug = 0  # debug
        pass

    def plan(self, om, Nd, Kd, Jd, ft_axes = None):
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
        

#         n_shift = tuple(0*x for x in Nd)
        self.ndims = len(Nd) # dimension
        if ft_axes is None:
            ft_axes = range(0, self.ndims)
        self.ft_axes = ft_axes
#     
        self.st = helper.plan(om, Nd, Kd, Jd, ft_axes = ft_axes)
        st_tmp = helper.plan0(om, Nd, Kd, Jd)
        if self.debug is 1:
            print('error between current and old interpolators=', scipy.sparse.linalg.norm(self.st['p'] - st_tmp['p'])/scipy.sparse.linalg.norm(self.st['p']))
            print('error between current and old scaling=', numpy.linalg.norm(self.st['sn'] - st_tmp['sn']))
        
        self.Nd = self.st['Nd']  # backup
        self.Kd = self.st['Kd']
        self.sn = numpy.asarray(self.st['sn'].astype(self.dtype)  ,order='C')# backup
            
        
        # Calculate the density compensation function
        self.sp = self.st['p'].copy().tocsr()
        self.spH = (self.st['p'].getH().copy()).tocsr()        
        self.Kdprod = numpy.int32(numpy.prod(self.st['Kd']))
        self.Jdprod = numpy.int32(numpy.prod(self.st['Jd']))
        del self.st['p'], self.st['sn']
#         self._precompute_sp()        
#         del self.st['p0'] 
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
#             self.sp = self.st['p']
#             self.spH = (self.st['p'].getH().copy()).tocsr()
            self.spHsp =self.spH.dot(self.sp).tocsr()
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
        from .linalg.solve_cpu import solve
        return solve(self,  y,  solver, *args, **kwargs)

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
        output_x.ravel()[self.KdCPUorder]=xx.ravel()[self.NdCPUorder]
        k = numpy.fft.fftn(output_x, axes = self.ft_axes)
#         k = numpy.fft.fftn(a=xx, s=tuple(self.Kd[ax] for ax in self.ft_axes), axes = self.ft_axes)
        return k
    
    def k2vec(self,k):
        k_vec = numpy.reshape(k, (numpy.prod(self.Kd), ), order='C')
        return k_vec
    
    def vec2y(self,k_vec):
        '''
        gridding: 
        '''
        y = self.sp.dot(k_vec)
#         y = self.st['ell'].spmv(k_vec)
        
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
#         k_vec = self.st['ell'].spmvH(y)
        
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
        
        k = numpy.fft.ifftn(k, axes = self.ft_axes)
        xx= numpy.zeros(self.Nd,dtype=self.dtype, order='C')
        xx.ravel()[self.NdCPUorder]=k.ravel()[self.KdCPUorder]
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
         
#         k = self.spHsp.dot(Xk)
#         k = self.spH.dot(self.sp.dot(Xk))
        k = self.y2vec(self.vec2y(Xk))
        k = self.vec2k(k)
        return k
    def lsmr(self, y, maxiter=50):
        vec = scipy.sparse.linalg.lsmr(self.sp, y, maxiter = maxiter)[0]
        k = self.vec2k(vec)
        xx = self.k2xx(k)
        x = xx/self.sn
        return x
        
    def vectorize(self, input_array):
        vec = input_array.flatten(order='C')
        return vec
    
    def tensorize(self, input_vec):
        arr = numpy.reshape(input_vec, self.tensor_shape, order='C')
        return arr     

    def matvec(self, input_vec):
        """
        mathematicians' way
        equivalent to forward, but both the input and output are vectors
        """
        input_array = self.tensorize(input_vec)
        vec = self.vectorize(self.forward(input_array))
        return vec
    
    def rmatvec(self, input_vec):
        """
        mathematicians' way
        equivalent to adjoint, but both the input and output are vectors
        """        
        tensor = self.adjoint(input_vec)
        vec = tensor.flatten(order='C')
        return vec

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
            helper.diagnose()
            
            return 1

#         Create context from device
        self.thr = api.Thread(device) #pyopencl.create_some_context()
        print('Using opencl or cuda = ', self.thr.api)
        
#         print('Using opencl?  ', self.thr.api is reikna.cluda.ocl)
#         """
#         Wavefront: as warp in cuda. Can control the width in a workgroup
#         Wavefront is required in spmv_vector as it improves data coalescence.
#         see cCSR_spmv and zSparseMatVec
#         """
        self.wavefront = api.DeviceParameters(device).warp_size

        print('wavefront = ',self.wavefront)

        from .src.re_subroutine import cMultiplyScalar, cCopy, cAddScalar,cAddVec,  cSelect, cMultiplyVec, cMultiplyVecInplace, cMultiplyConjVec, cDiff, cSqrt, cAnisoShrink, cHypot, cSpmv, cSpmvh, atomic_add


        kernel_sets = ( cMultiplyScalar.R + 
                                cCopy.R + cHypot.R +
                                cAddScalar.R + 
                                cSelect.R + 
                                cMultiplyConjVec.R + 
                                cAddVec.R+  
                                cMultiplyVecInplace.R + 
                                cDiff.R+ cSqrt.R+ cAnisoShrink.R+ cMultiplyVec.R + cSpmv.R + cSpmvh.R)
        
        try:
            if self.thr.api is cluda.cuda:
                kernel_sets =  atomic_add.cuda_add + kernel_sets
        except:
            try:
                print("No cuda device, trying ocl")
                if self.thr.api is cluda.ocl:
                    kernel_sets =  atomic_add.ocl_add + kernel_sets
            except:
                print('no ocl interface')
                
        prg = self.thr.compile(kernel_sets, 
                                render_kwds=dict(LL =  str(self.wavefront)), 
                                fast_math=False)

        self.cMultiplyScalar = prg.cMultiplyScalar
        self.cCopy = prg.cCopy
        self.cAddScalar = prg.cAddScalar
        self.cAddVec = prg.cAddVec
        self.cCSR_spmv_vector = prg.cCSR_spmv_vector
        self.cCSR_spmvh_scalar = prg.cCSR_spmvh_scalar     
        self.cSelect = prg.cSelect
        self.cMultiplyVecInplace = prg.cMultiplyVecInplace
        self.cMultiplyVec = prg.cMultiplyVec
        self.cMultiplyConjVec = prg.cMultiplyConjVec
        self.cDiff = prg.cDiff
        self.cSqrt= prg.cSqrt
        self.cAnisoShrink = prg.cAnisoShrink        
        self.cHypot = prg.cHypot               
#         self.cELL_spmv_scalar = prg.cELL_spmv_scalar
#         self.cELL_spmv_vector = prg.cELL_spmv_vector
#         self.cELL_spmvh_scalar = prg.cELL_spmvh_scalar
                      
        self.pELL_spmv_scalar = prg.pELL_spmv_scalar
        self.pELL_spmv_vector = prg.pELL_spmv_vector
        self.pELL_spmvh_scalar = prg.pELL_spmvh_scalar        

        self.pELL_nRow = numpy.uint32(self.st['pELL'].nRow)
        self.pELL_prodJd = numpy.uint32(self.st['pELL'].prodJd)
        self.pELL_sumJd = numpy.uint32(self.st['pELL'].sumJd)
        self.pELL_dim   = numpy.uint32(self.st['pELL'].dim)
        self.pELL_Jd= self.thr.to_device(self.st['pELL'].Jd.astype(numpy.uint32))
        self.pELL_currsumJd = self.thr.to_device(self.st['pELL'].curr_sumJd.astype(numpy.uint32))
        self.pELL_meshindex = self.thr.to_device(self.st['pELL'].meshindex.astype(numpy.uint32))
        self.pELL_kindx = self.thr.to_device(self.st['pELL'].kindx.astype(numpy.uint32))
        self.pELL_udata = self.thr.to_device(self.st['pELL'].udata.astype(self.dtype))
        
#         print('dim = ', self.pELL_dim )
#         self.ellcol = self.thr.to_device(self.st['ell'].col)
#         self.elldata = self.thr.to_device(self.st['ell'].data.astype(self.dtype))
        
        
        self.NdGPUorder = self.thr.to_device( self.NdCPUorder)
        self.KdGPUorder =  self.thr.to_device( self.KdCPUorder)
        self.Ndprod = numpy.int32(numpy.prod(self.st['Nd']))
        self.Kdprod = numpy.int32(numpy.prod(self.st['Kd']))
        self.M = numpy.int32( self.st['M'])
        
        self.SnGPUArray = self.thr.to_device(  self.sn)
        
        self.sp_data = self.thr.to_device( self.sp.data.astype(self.dtype))
        self.sp_indices =self.thr.to_device( self.sp.indices.astype(numpy.uint32))
        self.sp_indptr = self.thr.to_device( self.sp.indptr.astype(numpy.uint32))
        self.sp_numrow =  self.M
        self.sp_numcol = self.Kdprod
        del self.sp
        self.spH_data = self.thr.to_device(  self.spH.data.astype(self.dtype))
        self.spH_indices = self.thr.to_device(  self.spH.indices.astype(numpy.uint32))
        self.spH_indptr = self.thr.to_device(  self.spH.indptr.astype(numpy.uint32))
        self.spH_numrow = self.Kdprod
        del self.spH
#         self.spHsp_data = self.thr.to_device(  self.spHsp.data.astype(self.dtype))
#         self.spHsp_indices = self.thr.to_device( self.spHsp.indices)
#         self.spHsp_indptr =self.thr.to_device(  self.spHsp.indptr)
#         self.spHsp_numrow = self.Kdprod
#         del self.spHsp

        import reikna.fft

        self.fft = reikna.fft.FFT(numpy.empty(self.st['Kd'], dtype=self.dtype), self.ft_axes).compile(self.thr, fast_math=True)

        self.zero_scalar=self.dtype(0.0+0.0j)

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
        from .linalg.solve_hsa import solve
        
            
        try:
            return solve(self,  gy,  solver, *args, **kwargs)
        except:
            if numpy.ndarray==type(gy):
                print("input gy must be a reikna array with dtype = numpy.complex64")
                raise TypeError
            else:
                print("wrong")
                raise TypeError
          
    def forward(self, gx):
            """
            Forward NUFFT on the heterogeneous device
            
            :param gx: The input gpu array, with size=Nd
            :type: reikna gpu array with dtype =numpy.complex64
            :return: gy: The output gpu array, with size=(M,)
            :rtype: reikna gpu array with dtype =numpy.complex64
            """
            xx = self.x2xx(gx)
            k = self.xx2k(xx)
            del xx
            gy = self.k2y(k)
            del k
            return gy
    
    def adjoint(self, gy):
            """
            Adjoint NUFFT on the heterogeneous device
            
            :param gy: The input gpu array, with size=(M,)
            :type: reikna gpu array with dtype =numpy.complex64
            :return: gx: The output gpu array, with size=Nd
            :rtype: reikna gpu array with dtype =numpy.complex64
            """        
            k = self.y2k(gy)
            xx = self.k2xx(k)
            del k
            gx = self.xx2x(xx)
            del xx
            return gx
        
    def selfadjoint(self, gx):
        """
        selfadjoint NUFFT (Teplitz) on the heterogeneous device
        
        :param gx: The input gpu array, with size=Nd
        :type: reikna gpu array with dtype =numpy.complex64
        :return: gx: The output gpu array, with size=Nd
        :rtype: reikna gpu array with dtype =numpy.complex64
        """                
        gy = self.forward(gx)
        gx2 = self.adjoint(gy)
        del gy
        return gx2
    
    def x2xx(self, x):
        """
        Private: Scaling on the heterogeneous device
        Inplace multiplication of self.x_Nd by the scaling factor self.SnGPUArray.
        """           
        xx = self.thr.array(self.st['Nd'], dtype=self.dtype)
        self.thr.copy_array(x, xx, )#src_offset, dest_offset, size)
        self.cMultiplyVecInplace(self.SnGPUArray, xx, local_size=None, global_size=int(self.Ndprod))
        self.thr.synchronize()
        return xx
    
    def xx2k(self, xx):
        """
        Private: oversampled FFT on the heterogeneous device
        
        Firstly, zeroing the self.k_Kd array
        Second, copy self.x_Nd array to self.k_Kd array by cSelect
        Third: inplace FFT
        """
        k = self.thr.array(self.st['Kd'], dtype = self.dtype)
        k.fill(0)
#         self.cMultiplyScalar(self.zero_scalar, k, local_size=None, global_size=int(self.Kdprod))
        self.cSelect(self.NdGPUorder,      self.KdGPUorder,  xx, k, local_size=None, global_size=int(self.Ndprod))
        self.fft( k, k,inverse=False)
        self.thr.synchronize()
        return k
    
    def k2y(self, k):
        """
        Private: interpolation by the Sparse Matrix-Vector Multiplication
        """
        y =self.thr.array( (self.st['M'],), dtype=self.dtype).fill(0)
#         self.cCSR_spmv_vector(                                
#                            self.sp_numrow, 
#                            self.sp_indptr,
#                            self.sp_indices,
#                            self.sp_data, 
#                            k,
#                            y,
#                            local_size=int(self.wavefront),
#                            global_size=int(self.sp_numrow*self.wavefront) 
#                             )
#         self.cELL_spmv_scalar(                                
#                            self.sp_numrow, 
#                            numpy.int32(self.st['ell'].colWidth),
#                            self.sp_indices,
#                            self.sp_data, 
#                            k,
#                            y,
#                            local_size=None,
#                            global_size=int(self.sp_numrow) 
#                             )

#         self.cELL_spmv_vector(                                
#                           self.sp_numrow, 
#                           numpy.uint32(self.st['ell'].colWidth),
#                           self.sp_indices,
#                           self.sp_data, 
#                           k,
#                           y,
#                           local_size=int(self.wavefront),
#                           global_size=int(self.sp_numrow*self.wavefront) 
#                            )
        
#         self.pELL_spmv_scalar(
#                             self.pELL_nRow,
#                             self.pELL_prodJd,
#                             self.pELL_sumJd, 
#                             self.pELL_dim,
#                             self.pELL_Jd,
#                             self.pELL_currsumJd,
#                             self.pELL_meshindex,
#                             self.pELL_kindx,
#                             self.pELL_udata, 
#                             k,
#                             y,
#                             local_size=None,
#                             global_size= int(self.pELL_nRow)             
#                             )                 
        self.pELL_spmv_vector(
                            self.pELL_nRow,
                            self.pELL_prodJd,
                            self.pELL_sumJd, 
                            self.pELL_dim,
                            self.pELL_Jd,
                            self.pELL_currsumJd,
                            self.pELL_meshindex,
                            self.pELL_kindx,
                            self.pELL_udata, 
                            k,
                            y,
                            local_size= int(self.wavefront),
                            global_size= int(self.pELL_nRow*self.wavefront)             
                            )           
        self.thr.synchronize()
        return y
    
    def y2k(self, y):
        """
        Private: gridding by the Sparse Matrix-Vector Multiplication
        However, serial atomic add is far too slow and inaccurate.
        """
        k = self.thr.array(self.st['Kd'], dtype = self.dtype)
        kx = self.thr.array(self.st['Kd'], dtype = numpy.float32)
        ky = self.thr.array(self.st['Kd'], dtype = numpy.float32)
        
#         gk = self.thr.array(self.st['Kd'], dtype = self.dtype)
        
        k.fill(0.0 + 0.0j)
        kx.fill(0.0)
        ky.fill(0.0)
#         g_err = self.thr.array(self.st['Kd'],dtype = self.dtype)
#         g_err.fill(0.0)
#         self.zeroing(k, local_size=None,
#                            global_size=int(self.sp_numcol) )
#         self.zeroing(g_err, local_size=None,
#                            global_size=int(self.sp_numcol) )
#         self.thr.synchronize()


#         self.cCSR_spmvh_scalar(
#         self.sp_numrow,
#         self.sp_indptr,
#         self.sp_indices,
#         self.sp_data,
#         kx,
#         ky,
#         y,
# #         g_err,
#         local_size=None,
#         global_size=int(self.sp_numrow) )       

#         self.cELL_spmvh_scalar(
#                           self.sp_numrow, 
#                           numpy.uint32(self.st['ell'].colWidth),
#                           self.sp_indices,
#                           self.sp_data, 
#                         kx,
#                         ky,
#                         y,
#                         local_size=None,
#                         global_size=int(self.sp_numrow) )      
        self.pELL_spmvh_scalar(
                            self.pELL_nRow,
                            self.pELL_prodJd,
                            self.pELL_sumJd, 
                            self.pELL_dim,
                            self.pELL_Jd,
                            self.pELL_currsumJd,
                            self.pELL_meshindex,
                            self.pELL_kindx,
                            self.pELL_udata, 
                            kx,
                            ky,
                            y,
                            local_size=None,
                            global_size= int(self.pELL_nRow)             
                            )         
#         self.cAddVec(k, g_err, gk, local_size=None, global_size=(self.sp_numcol) )
#         self.add(k, g_err, local_size=None, global_size=(self.sp_numcol) )
        k = kx+1.0j* ky
        
#         self.thr.synchronize()
        
        return k
    def y2k_old(self, y):
        """
        Private: gridding by the Sparse Matrix-Vector Multiplication
        """
        k = self.thr.array(self.st['Kd'], dtype = self.dtype)
#         g_err = self.thr.array(self.st['Kd'],dtype = self.dtype)
#         self.zeroing(k, local_size=None,
#                            global_size=int(self.sp_numcol) )
#         self.zeroing(g_err, local_size=None,
#                            global_size=int(self.sp_numcol) )
#         self.thr.synchronize()
#         self.cCSR_spmv_scalar(
#         self.sp_numrow,
#         self.sp_indptr,
#         self.sp_indices,
#         self.sp_data,
#         k,
#         y,
#         g_err,
#         local_size=None,
#         global_size=int(self.sp_numrow) )        
        self.cCSR_spmv_vector(
                           self.spH_numrow, 
                           self.spH_indptr,
                           self.spH_indices,
                           self.spH_data, 
                           y,
                           k,
                           local_size=int(self.wavefront),
                           global_size=int(self.spH_numrow*self.wavefront) 
                            )#,g_times_l=int(csrnumrow))
#         self.thr.synchronize()
#         self.add(k, g_err, local_size=None, global_size=(self.sp_numcol) )
        self.thr.synchronize()
        return k    
    def k2xx(self, k):
        """
        Private: the inverse FFT and image cropping (which is the reverse of _xx2k() method)
        """        
        xx = self.thr.array(self.st['Nd'], dtype = self.dtype)
        self.fft( k, k, inverse=True)
        self.thr.synchronize()
#         self.x_Nd._zero_fill()
#         self.cMultiplyScalar(self.zero_scalar, xx,  local_size=None, global_size=int(self.Ndprod ))
        xx.fill(0)
#         self.cSelect(self.queue, (self.Ndprod,), None,   self.KdGPUorder.data,  self.NdGPUorder.data,     self.k_Kd2.data, self.x_Nd.data )
        self.cSelect(  self.KdGPUorder,  self.NdGPUorder,     k, xx, local_size=None, global_size=int(self.Ndprod ))
        
        return xx
    def xx2x(self, xx):
        """
        Private: rescaling, which is identical to the  _x2xx() method
        """
        x = self.x2xx(xx)
        return x
