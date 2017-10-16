"""
HSA solvers
======================================
"""
import numpy, scipy
dtype = numpy.complex64
# def  L1TVLAD():
import scipy
import numpy
from .._helper.helper import *

def cDiff(x, d_indx):
        """
        Compute image gradient
        Work with indxmap_diff(Nd)
        """    
        a2=numpy.asarray(x.copy(),order='C')
        a2.flat =   a2 .flat[d_indx] - a2 .flat
        return a2
def _create_kspace_sampling_density(nufft):
        """
        Compute kspace sampling density from the nufft object
        """    
        y = numpy.ones((nufft.st['M'],),dtype = numpy.complex64)
        nufft.y = nufft.thr.to_device(y)
        nufft._y2k()
        w =  numpy.abs( nufft.k_Kd2.get())#**2) ))
    
        nufft.st['w'] = w#self.nufftobj.vec2k(w)
        RTR=nufft.st['w'] # see __init__() in class "nufft"
        return RTR
# def _create_laplacian_kernel(nufft):
# #===============================================================================
# # #        # Laplacian oeprator, convolution kernel in spatial domain
# #         # related to constraint
# #===============================================================================
#     uker = numpy.zeros(nufft.st['Kd'][:],dtype=numpy.complex64,order='C')
#     n_dims= numpy.size(nufft.st['Nd'])
# 
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
# 
#     uker =numpy.fft.fftn(uker) #, self.nufftobj.st['Kd'], range(0,numpy.ndim(uker)))
#     return uker  
 
def L1TVLAD(nufft, gy, maxiter, rho  ): # main function of solver
    """
    L1-total variation regularized least absolute deviation 
    """
    mu = 1.0
    LMBD = rho*mu

    def AHA(x):
        x2 = nufft.selfadjoint(x)
        return x2
    def AH(gy):
        x2 = nufft.adjoint(gy)
        return x2
    
    uker_cpu = mu*_create_kspace_sampling_density(nufft)   - LMBD* create_laplacian_kernel(nufft) # on cpu
    uker = nufft.thr.to_device(uker_cpu.astype(numpy.complex64))
    AHy = AH(gy) # on  device?
    z = numpy.zeros(nufft.st['Nd'],dtype = numpy.complex64,order='C')
    z_gpu = nufft.thr.to_device(z)
    xkp1 = nufft.thr.copy_array(z_gpu)
    AHyk = nufft.thr.copy_array(z_gpu)
           
#         self._allo_split_variables()        
    zz= []
    bb = []
    dd = []
    d_indx, dt_indx = indxmap_diff(nufft.st['Nd'])
    ndims = len(nufft.st['Nd'])
    s_tmp = []
    for pp in range(0, ndims):
        s_tmp += [0, ]
    for jj in range(    0,  ndims): # n_dims + 1 for wavelets
        d_indx[jj] = nufft.thr.to_device(d_indx[jj])
        dt_indx[jj] = nufft.thr.to_device(dt_indx[jj])
#     z=numpy.zeros(nufft.st['Nd'], dtype = nufft.dtype, order='C')
    
#     ndims = len(nufft.st['Nd'])
    
    for jj in range(    0,  ndims): # n_dims + 1 for wavelets
        
        zz += [nufft.thr.copy_array(z_gpu),]
        bb += [nufft.thr.copy_array(z_gpu),]
        dd +=  [nufft.thr.copy_array(z_gpu),]
    zf = nufft.thr.copy_array(z_gpu)
    bf = nufft.thr.copy_array(z_gpu)
    df = nufft.thr.copy_array(z_gpu)

    n_dims = len(nufft.st['Nd'])#numpy.size(uf.shape)
    
    tmp_gpu = nufft.thr.copy_array(z_gpu) 
    
    
    for outer in numpy.arange(0, maxiter):
#             for inner in numpy.arange(0,nInner):
            
        # solve Ku = rhs
#                 rhs = (mu*(AHyk + df - bf) +  # right hand side
#                 LMBD*(cDiff(dd[0] - bb[0],  dt_indx[0])) + 
#                 LMBD*(cDiff(dd[1] - bb[1],  dt_indx[1]))  )      
        rhs = nufft.thr.copy_array(AHyk)
        
        rhs += df
        
        rhs -= bf
        
#         rhs *= mu
        nufft.cMultiplyScalar( dtype(mu), rhs, local_size=None, global_size=int(nufft.Ndprod))
        
#         print(rhs.get())
        for pp in range(0, ndims): 
            in_cDiff = nufft.thr.copy_array(dd[pp])
            
            in_cDiff -= bb[pp]
            
    #         out_cDiff = nufft.thr.empty_like(in_cDiff) 
            nufft.cDiff(dt_indx[pp], in_cDiff, tmp_gpu, local_size=None, global_size=int(nufft.Ndprod))
            
    #         tmp_gpu *= LMBD
            nufft.cMultiplyScalar( dtype(LMBD), tmp_gpu, local_size=None, global_size=int(nufft.Ndprod))
            
            rhs += tmp_gpu
        
#         print(rhs.get())
#         in_cDiff = nufft.thr.copy_array(dd[1])
#         
#         in_cDiff -= bb[1]
#         
# #         out_cDiff = nufft.thr.empty_like(in_cDiff) 
#         nufft.cDiff(dt_indx[1], in_cDiff, tmp_gpu, local_size=None, global_size=int(nufft.Ndprod))
#         
# #         tmp_gpu *= LMBD
#         nufft.cMultiplyScalar( dtype(LMBD), tmp_gpu, local_size=None, global_size=int(nufft.Ndprod))
#         
#         rhs += tmp_gpu
        
#         print(rhs.get())
    
        # Note K = F' uker F
        # so K-1 ~ F
#         xkp1 = nufft.k2xx(nufft.xx2k(rhs) / uker) 
        nufft.x_Nd = nufft.thr.copy_array(rhs)
        
        nufft._xx2k()
        
        nufft.k_Kd2 = nufft.k_Kd/uker
        
#         nufft.k_Kd2 = nufft.thr.copy_array(nufft.k_Kd)
        nufft._k2xx()
        xkp1 = nufft.thr.copy_array(nufft.x_Nd)
#         print(xkp1.get())
#                 self._update_d(xkp1)


#         zz[0] = cDiff(xkp1,  d_indx[0])
#         zz[1] = cDiff(xkp1,  d_indx[1])
        for pp in range(0, ndims):
            nufft.cDiff(d_indx[pp],  xkp1, zz[pp],  local_size=None, global_size=int(nufft.Ndprod)) 
#         nufft.cDiff(d_indx[0],  xkp1, zz[0],  local_size=None, global_size=int(nufft.Ndprod))
#         nufft.cDiff(d_indx[1],  xkp1, zz[1],  local_size=None, global_size=int(nufft.Ndprod))
        

#         zf = AHA(xkp1)  -AHy
        zf = AHA(xkp1)
        
        zf -= AHy 
        

        '''
        soft-thresholding the edges
        '''
        for pp in range(0, ndims):
            s_tmp[pp] = zz[pp] + bb[pp]
#         s1 = zz[0] + bb[0]
#         
#         s2 = zz[1] + bb[1]
        
#         s = s1**2 + s2**2
#         s1 *= s1
            nufft.cMultiplyConjVec(s_tmp[pp], s_tmp[pp], tmp_gpu, local_size=None, global_size=int(nufft.Ndprod))
            if pp > 0:
                s += tmp_gpu
            else: # pp == 0
                s = nufft.thr.copy_array(tmp_gpu)
#         nufft.cMultiplyConjVec(s1, s1, tmp_gpu, local_size=None, global_size=int(nufft.Ndprod))
#         s = nufft.thr.copy_array(tmp_gpu)
# #         s2 *= s2
#         nufft.cMultiplyConjVec(s2, s2, tmp_gpu, local_size=None, global_size=int(nufft.Ndprod))
#         s += tmp_gpu
        
#         s = s1 + s2
        
        
        nufft.cSqrt(s, local_size=None, global_size=int(nufft.Ndprod))
        
        s += 1e-6
        
        threshold_value = dtype(1/LMBD)
#         r =(s > threshold_value)*(s-threshold_value)/s#numpy.maximum(s - threshold_value ,  0.0)/s
        nufft.cAnisoShrink(threshold_value, s, tmp_gpu, local_size=None, global_size=int(nufft.Ndprod))
        tmp_gpu /=s
        
#         dd[0] = s1*r
#         dd[1] = s2*r
#         dd[0] = s1*tmp_gpu
        for pp in range(0, ndims):
            nufft.cMultiplyVec(s_tmp[pp], tmp_gpu, dd[pp], local_size=None, global_size=int(nufft.Ndprod)) 
#         nufft.cMultiplyVec(s1, tmp_gpu, dd[0], local_size=None, global_size=int(nufft.Ndprod)) 
        
#         dd[1] = s2*tmp_gpu
        
#         nufft.cMultiplyVec(s2, tmp_gpu, dd[1], local_size=None, global_size=int(nufft.Ndprod))
        
        tmp_gpu = zf+bf
        
        threshold_value=dtype(1.0/mu)
        
#         df.real =0.0+ (df.real>threshold_value)*(df.real - threshold_value) +(df.real<= - threshold_value)*(df.real+threshold_value)
#         df.imag = 0.0+(df.imag>threshold_value)*(df.imag - threshold_value) +(df.imag<= - threshold_value)*(df.imag+threshold_value)
    
        nufft.cAnisoShrink(threshold_value,  tmp_gpu, df, local_size=None, global_size=int(nufft.Ndprod))
        
         
#                 df =     sy
        # end of shrinkage
        for pp in range(0, ndims):
            bb[pp] += zz[pp] - dd[pp]
#         bb[0] += zz[0] - dd[0] 
#         bb[1] += zz[1] - dd[1] 
        bf += zf - df 
#                 self._update_b() # update b based on the current u
#         print(outer)
        
        AHyk -= zf # Linearized Bregman iteration f^k+1 = f^k + f - Au
        
#     print(xkp1.get())
#             print(outer)
#     print('here')
    nufft.x_Nd = nufft.thr.copy_array(xkp1)
    return 0
def L1TVOLS(nufft, gy, maxiter, rho  ): # main function of solver
    """
    L1-total variation regularized ordinary least square 
    """
    mu = 1.0
    LMBD = rho*mu

    def AHA(x):
        x2 = nufft.selfadjoint(x)
        return x2
    def AH(gy):
        x2 = nufft.adjoint(gy)
        return x2
    
    uker_cpu = mu*_create_kspace_sampling_density(nufft)   - LMBD* create_laplacian_kernel(nufft) # on cpu
    uker = nufft.thr.to_device(uker_cpu.astype(numpy.complex64))
    AHy = AH(gy) # on  device?
    z = numpy.zeros(nufft.st['Nd'],dtype = numpy.complex64,order='C')
    z_gpu = nufft.thr.to_device(z)
    xkp1 = nufft.thr.copy_array(z_gpu)
    AHyk = nufft.thr.copy_array(z_gpu)
           
#         self._allo_split_variables()        
    zz= []
    bb = []
    dd = []
    d_indx, dt_indx = indxmap_diff(nufft.st['Nd'])
    ndims = len(nufft.st['Nd'])
    s_tmp = []
    for pp in range(0, ndims):
        s_tmp += [0, ]
    for jj in range(    0,  ndims): # n_dims + 1 for wavelets
        d_indx[jj] = nufft.thr.to_device(d_indx[jj])
        dt_indx[jj] = nufft.thr.to_device(dt_indx[jj])
#     z=numpy.zeros(nufft.st['Nd'], dtype = nufft.dtype, order='C')
    
#     ndims = len(nufft.st['Nd'])
    
    for jj in range(    0,  ndims): # n_dims + 1 for wavelets
        
        zz += [nufft.thr.copy_array(z_gpu),]
        bb += [nufft.thr.copy_array(z_gpu),]
        dd +=  [nufft.thr.copy_array(z_gpu),]
#     zf = nufft.thr.copy_array(z_gpu)
#     bf = nufft.thr.copy_array(z_gpu)
#     df = nufft.thr.copy_array(z_gpu)

    n_dims = len(nufft.st['Nd'])#numpy.size(uf.shape)
    
    tmp_gpu = nufft.thr.copy_array(z_gpu) 
    
    
    for outer in numpy.arange(0, maxiter):
#             for inner in numpy.arange(0,nInner):
            
        # solve Ku = rhs
#                 rhs = (mu*(AHyk + df - bf) +  # right hand side
#                 LMBD*(cDiff(dd[0] - bb[0],  dt_indx[0])) + 
#                 LMBD*(cDiff(dd[1] - bb[1],  dt_indx[1]))  )      
        rhs = nufft.thr.copy_array(AHyk)
        
#         rhs += df
#         
#         rhs -= bf
        
#         rhs *= mu
        nufft.cMultiplyScalar( dtype(mu), rhs, local_size=None, global_size=int(nufft.Ndprod))
        
#         print(rhs.get())
        for pp in range(0, ndims): 
            in_cDiff = nufft.thr.copy_array(dd[pp])
            
            in_cDiff -= bb[pp]
            
    #         out_cDiff = nufft.thr.empty_like(in_cDiff) 
            nufft.cDiff(dt_indx[pp], in_cDiff, tmp_gpu, local_size=None, global_size=int(nufft.Ndprod))
            
    #         tmp_gpu *= LMBD
            nufft.cMultiplyScalar( dtype(LMBD), tmp_gpu, local_size=None, global_size=int(nufft.Ndprod))
            
            rhs += tmp_gpu
        
#         print(rhs.get())
#         in_cDiff = nufft.thr.copy_array(dd[1])
#         
#         in_cDiff -= bb[1]
#         
# #         out_cDiff = nufft.thr.empty_like(in_cDiff) 
#         nufft.cDiff(dt_indx[1], in_cDiff, tmp_gpu, local_size=None, global_size=int(nufft.Ndprod))
#         
# #         tmp_gpu *= LMBD
#         nufft.cMultiplyScalar( dtype(LMBD), tmp_gpu, local_size=None, global_size=int(nufft.Ndprod))
#         
#         rhs += tmp_gpu
        
#         print(rhs.get())
    
        # Note K = F' uker F
        # so K-1 ~ F
#         xkp1 = nufft.k2xx(nufft.xx2k(rhs) / uker) 
        nufft.x_Nd = nufft.thr.copy_array(rhs)
        
        nufft._xx2k()
        
        nufft.k_Kd2 = nufft.k_Kd/uker
        
#         nufft.k_Kd2 = nufft.thr.copy_array(nufft.k_Kd)
        nufft._k2xx()
        xkp1 = nufft.thr.copy_array(nufft.x_Nd)
#         print(xkp1.get())
#                 self._update_d(xkp1)


#         zz[0] = cDiff(xkp1,  d_indx[0])
#         zz[1] = cDiff(xkp1,  d_indx[1])
        for pp in range(0, ndims):
            nufft.cDiff(d_indx[pp],  xkp1, zz[pp],  local_size=None, global_size=int(nufft.Ndprod)) 
#         nufft.cDiff(d_indx[0],  xkp1, zz[0],  local_size=None, global_size=int(nufft.Ndprod))
#         nufft.cDiff(d_indx[1],  xkp1, zz[1],  local_size=None, global_size=int(nufft.Ndprod))
        

#         zf = AHA(xkp1)  -AHy
        zf = AHA(xkp1)
        
        zf -= AHy 
        

        '''
        soft-thresholding the edges
        '''
        for pp in range(0, ndims):
            s_tmp[pp] = zz[pp] + bb[pp]
#         s1 = zz[0] + bb[0]
#         
#         s2 = zz[1] + bb[1]
        
#         s = s1**2 + s2**2
#         s1 *= s1
            nufft.cMultiplyConjVec(s_tmp[pp], s_tmp[pp], tmp_gpu, local_size=None, global_size=int(nufft.Ndprod))
            if pp > 0:
                s += tmp_gpu
            else: # pp == 0
                s = nufft.thr.copy_array(tmp_gpu)
#         nufft.cMultiplyConjVec(s1, s1, tmp_gpu, local_size=None, global_size=int(nufft.Ndprod))
#         s = nufft.thr.copy_array(tmp_gpu)
# #         s2 *= s2
#         nufft.cMultiplyConjVec(s2, s2, tmp_gpu, local_size=None, global_size=int(nufft.Ndprod))
#         s += tmp_gpu
        
#         s = s1 + s2
        
        
        nufft.cSqrt(s, local_size=None, global_size=int(nufft.Ndprod))
        
        s += 1e-6
        
        threshold_value = dtype(1/LMBD)
#         r =(s > threshold_value)*(s-threshold_value)/s#numpy.maximum(s - threshold_value ,  0.0)/s
        nufft.cAnisoShrink(threshold_value, s, tmp_gpu, local_size=None, global_size=int(nufft.Ndprod))
        tmp_gpu /=s
        
#         dd[0] = s1*r
#         dd[1] = s2*r
#         dd[0] = s1*tmp_gpu
        for pp in range(0, ndims):
            nufft.cMultiplyVec(s_tmp[pp], tmp_gpu, dd[pp], local_size=None, global_size=int(nufft.Ndprod)) 
#         nufft.cMultiplyVec(s1, tmp_gpu, dd[0], local_size=None, global_size=int(nufft.Ndprod)) 
        
#         dd[1] = s2*tmp_gpu
        
#         nufft.cMultiplyVec(s2, tmp_gpu, dd[1], local_size=None, global_size=int(nufft.Ndprod))
        
#         tmp_gpu = zf+bf
        
#         threshold_value=dtype(1.0/mu)
        
#         df.real =0.0+ (df.real>threshold_value)*(df.real - threshold_value) +(df.real<= - threshold_value)*(df.real+threshold_value)
#         df.imag = 0.0+(df.imag>threshold_value)*(df.imag - threshold_value) +(df.imag<= - threshold_value)*(df.imag+threshold_value)
    
#         nufft.cAnisoShrink(threshold_value,  tmp_gpu, df, local_size=None, global_size=int(nufft.Ndprod))
        
         
#                 df =     sy
        # end of shrinkage
        for pp in range(0, ndims):
            bb[pp] += zz[pp] - dd[pp]
#         bb[0] += zz[0] - dd[0] 
#         bb[1] += zz[1] - dd[1] 
#         bf += zf - df 
#                 self._update_b() # update b based on the current u
#         print(outer)
        
        AHyk -= zf # Linearized Bregman iteration f^k+1 = f^k + f - Au
        
#     print(xkp1.get())
#             print(outer)
#     print('here')
    nufft.x_Nd = nufft.thr.copy_array(xkp1)
    return 0

def solve(nufft,gy, solver=None,  maxiter=30, *args, **kwargs):
    """
    Solve NUFFT.
    The current version supports solvers = 'cg' or 'L1TVOLS' or 'L1TVLAD'.
    
    :param nufft: NUFFT_hsa object
    :param y: (M,) array, non-uniform data
    :return: x: image
        
    """
    # define the reduction kernel on the device
#     if None ==  solver:
#         solver  =   'cg'
    if 'L1TVLAD' == solver:
        L1TVLAD(nufft, gy, maxiter=maxiter, *args, **kwargs  )
        x2 = nufft.thr.copy_array(nufft.x_Nd)
        return x2
    elif 'L1TVOLS' == solver:
        L1TVOLS(nufft, gy, maxiter=maxiter, *args, **kwargs  )
        x2 = nufft.thr.copy_array(nufft.x_Nd)
        return x2
    elif 'dc'   ==  solver:
         """
         Density compensation method
         nufft.st['W'] will be computed if doesn't exist
         If nufft.st['W'] exist then x2 = nufft.adjoint(nufft.st['W']*y)
         input: 
             y: (M,) array
         output:
             x2: Nd array
         """
         print(solver)

         nufft.st['W'] = nufft._pipe_density(maxiter=maxiter,*args, **kwargs)
 
         x2 = nufft.adjoint(nufft.st['W']*gy)
         return x2
#             return gx        
    elif 'cg' == solver:

        from reikna.algorithms import Reduce, Predicate, predicate_sum
        
        nufft.reduce_sum = Reduce(nufft.k_Kd, predicate_sum(dtype))
        nufft.reduce_sum  = nufft.reduce_sum.compile(nufft.thr)        
        
 
        # update: b = spH * gy              
        b= nufft.thr.empty_like( nufft.k_Kd)
        nufft.cSparseMatVec(
                                   nufft.spH_numrow, 
                                   nufft.spH_indptr,
                                   nufft.spH_indices,
                                   nufft.spH_data, 
                                   gy,
                                   b,
                                   local_size=int(nufft.wavefront),
                                   global_size=int(nufft.spH_numrow*nufft.wavefront) 
                                    )#,g_times_l=int(csrnumrow))
#         print('b',numpy.sum(b.get()))     
 
        # Initialize x = b
        x   =   nufft.thr.copy_array( b)
        rsold = nufft.thr.empty_like(nufft.reduce_sum.parameter.output)
        nufft.reduce_sum(rsold, x)
#         print('x',rsold) 
        
        # initialize r = b - A * x
        r   =   nufft.thr.empty_like( b)
        Ax   =   nufft.thr.empty_like( b)
        nufft.cSparseMatVec( 
                                   nufft.spHsp_numrow, 
                                   nufft.spHsp_indptr,
                                   nufft.spHsp_indices,
                                   nufft.spHsp_data, 
                                   x,
                                   Ax,
                                   local_size=int(nufft.wavefront),
                                   global_size=int(nufft.spHsp_numrow*nufft.wavefront) 
                                    )#,g_times_l=int(csrnumrow))     
         
        rsold = nufft.thr.empty_like(nufft.reduce_sum.parameter.output)
        nufft.reduce_sum(rsold, Ax)
#         print('Ax',rsold) 
        nufft.cAddVec(b, - Ax, r , local_size=None, global_size = int(nufft.Kdprod))
        
        # p = r
        p   =   nufft.thr.copy_array(r)

        # rsold = r' * r
        tmp_array = nufft.thr.empty_like( r)
        nufft.cMultiplyConjVec(r, r, tmp_array, local_size=None, global_size=int(nufft.Kdprod))
        rsold = nufft.thr.empty_like(nufft.reduce_sum.parameter.output)
        nufft.reduce_sum(rsold, tmp_array)

        # allocate Ap
        Ap  =   nufft.thr.empty_like( b)     

        rsnew = nufft.thr.empty_like(nufft.reduce_sum.parameter.output)
        tmp_sum = nufft.thr.empty_like(nufft.reduce_sum.parameter.output)
        
        for pp in range(0, maxiter):
            
            # Ap = A*p
            nufft.cSparseMatVec( 
                                   nufft.spHsp_numrow, 
                                   nufft.spHsp_indptr,
                                   nufft.spHsp_indices,
                                   nufft.spHsp_data, 
                                   p,
                                   Ap,
                                   local_size=int(nufft.wavefront),
                                   global_size=int(nufft.spHsp_numrow*nufft.wavefront) 
                                    )#,g_times_l=int(csrnumrow))     
            
#             alpha = rs_old/(p'*Ap)
            nufft.cMultiplyConjVec(p, Ap, tmp_array, local_size=None, global_size=int(nufft.Kdprod))
            nufft.reduce_sum(tmp_sum, tmp_array)
            
                        
            alpha = rsold / tmp_sum
#             print(pp,rsold , alpha, numpy.sum(tmp_array.get()) )
            # x = x + alpha*p
            p2 = nufft.thr.copy_array(p)
            nufft.cMultiplyScalar(alpha.get(), p2,  local_size=None, global_size=int(nufft.Kdprod))
#             nufft.cAddVec(x, alpha, local_size=None, global_size=int(nufft.Kdprod))
            x += p2

            # r = r - alpha * Ap
            p2= nufft.thr.copy_array(Ap)
            nufft.cMultiplyScalar(alpha.get(), p2,  local_size=None, global_size=int(nufft.Kdprod))
            r -= p2
#             print(pp, numpy.sum(x.get()), numpy.sum(r.get()))
            # rs_new = r'*r
            
            nufft.cMultiplyConjVec(r,    r,  tmp_array, local_size=None, global_size=int(nufft.Kdprod))
            nufft.reduce_sum(rsnew, tmp_array)        
            
            # tmp_sum = p = r + (rs_new/rs_old)*p
            beta = rsnew/rsold
            p2= nufft.thr.copy_array(p)
            nufft.cMultiplyScalar(beta,   p2,  local_size=None, global_size=int(nufft.Kdprod))
            nufft.cAddVec(r, p2, p, local_size=None, global_size=int(nufft.Kdprod))
            p = r + p2
            
            rsold =nufft.thr.copy_array( rsnew)
        # end of iteration    
        
        # copy result to k_Kd2
        nufft.k_Kd2 = nufft.thr.copy_array(x)
        
        # inverse FFT: k_Kd2 -> x_Nd
        nufft._k2xx()
        
        # rescale the SnGPUArray
        x2 = nufft.thr.copy_array(nufft.x_Nd)/nufft.SnGPUArray
        
        
        return x2