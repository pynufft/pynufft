import scipy
import numpy
def indxmap_diff(Nd):
    """
    Build indexes for image gradient
    input: 
                Nd: tuple, the size of image
    output: 
                d_indx: tuple
                        Diff(x) = x.flat[d_indx[0]] - x.flat
                dt_indx: tuple,  index of the adjoint Diff
                    Diff_t(x) =  x.flat[dt_indx[0]] - x.flat
    """
    ndims = len(Nd)
    Ndprod = numpy.prod(Nd)
    mylist = numpy.arange(0, Ndprod).astype(numpy.int32)
    mylist = numpy.reshape(mylist, Nd)
    d_indx = ()
    dt_indx = ()
    for pp in range(0, ndims):
        d_indx = d_indx + ( numpy.reshape(   numpy.roll(  mylist, +1 , pp  ), (Ndprod,)  ,order='C').astype(numpy.int32) ,)
        dt_indx = dt_indx + ( numpy.reshape(   numpy.roll(  mylist, -1 , pp  ) , (Ndprod,) ,order='C').astype(numpy.int32) ,)

    return d_indx,  dt_indx  
def cDiff(x, d_indx):
    a2=numpy.asarray(x.copy(),order='C')
    a2.flat =   a2 .flat[d_indx] - a2 .flat
    return a2
def _create_kspace_sampling_density(nufft):
    y = numpy.ones((nufft.st['M'],),dtype = numpy.complex64)
    w =  numpy.abs( nufft.y2k(y))#**2) ))

    nufft.st['w'] = w#self.nufftobj.vec2k(w)
    RTR=nufft.st['w'] # see __init__() in class "nufft"
    return RTR
def _create_laplacian_kernel(nufft):
#===============================================================================
# #        # Laplacian oeprator, convolution kernel in spatial domain
#         # related to constraint
#===============================================================================
    uker = numpy.zeros(nufft.st['Kd'][:],dtype=numpy.complex64,order='C')
    n_dims= numpy.size(nufft.st['Nd'])

    if n_dims == 1:
        uker[0] = -2.0
        uker[1] = 1.0
        uker[-1] = 1.0
    elif n_dims == 2:
        uker[0,0] = -4.0
        uker[1,0] = 1.0
        uker[-1,0] = 1.0
        uker[0,1] = 1.0
        uker[0,-1] = 1.0
    elif n_dims == 3:  
        uker[0,0,0] = -6.0
        uker[1,0,0] = 1.0
        uker[-1,0,0] = 1.0
        uker[0,1,0] = 1.0
        uker[0,-1,0] = 1.0
        uker[0,0,1] = 1.0
        uker[0,0,-1] = 1.0                      

    uker =numpy.fft.fftn(uker) #, self.nufftobj.st['Kd'], range(0,numpy.ndim(uker)))
    return uker  
 
def L1LAD(nufft, y, maxiter, rho  ): # main function of solver
    print("L1LAD")
    mu = 1.0
    LMBD = rho*mu

    def AHA(x):
        x2 = nufft.selfadjoint(x)
        return x2
    def AH(y):
        x2 = nufft.adjoint(y)
        return x2
    
    uker = mu*_create_kspace_sampling_density(nufft)   - LMBD* _create_laplacian_kernel(nufft)
    
    AHy = AH(y)
    
    xkp1 = numpy.zeros_like(AHy)
    AHyk = numpy.zeros_like(AHy)
           
#         self._allo_split_variables()        
    zz= []
    bb = []
    dd = []
    d_indx, dt_indx = indxmap_diff(nufft.st['Nd'])
    z=numpy.zeros(nufft.st['Nd'], dtype = nufft.dtype, order='C')
    
    ndims = len(nufft.st['Nd'])
    
    for jj in range(    0,  ndims): # n_dims + 1 for wavelets
        
        zz += [z.copy(),]
        bb += [z.copy(),]
        dd +=  [z.copy(),]
    zf = z.copy()
    bf = z.copy()
    df = z.copy()

    n_dims = len(nufft.st['Nd'])#numpy.size(uf.shape)
    
    for outer in numpy.arange(0, maxiter):
#             for inner in numpy.arange(0,nInner):
            
        # solve Ku = rhs
            
        rhs = (mu*(AHyk + df - bf) +  # right hand side
                LMBD*(cDiff(dd[0] - bb[0],  dt_indx[0])) + 
                LMBD*(cDiff(dd[1] - bb[1],  dt_indx[1]))  )          
        # Note K = F' uker F
        # so K-1 ~ F
        xkp1 = nufft.k2xx(nufft.xx2k(rhs) / uker) 
#                 self._update_d(xkp1)

        zz[0] = cDiff(xkp1,  d_indx[0])
        zz[1] = cDiff(xkp1,  d_indx[1])
        zf = AHA(xkp1)  -AHy 

        '''
        soft-thresholding the edges
        '''

        s1 = zz[0] + bb[0]
        s2 = zz[1] + bb[1]
#         s = sum((self.zz[pj] + self.bb[pj])**2 for pj in range(0,n_dims))
        s = s1**2 + s2**2
        s = s**0.5 +1e-3

        threshold_value = 1/LMBD
        r =(s > threshold_value)*(s-threshold_value)/s#numpy.maximum(s - threshold_value ,  0.0)/s
        dd[0] = s1*r
        dd[1] = s2*r
        df = zf+bf
        
        threshold_value=1.0/mu
    
        df.real =0.0+ (df.real>threshold_value)*(df.real - threshold_value) +(df.real<= - threshold_value)*(df.real+threshold_value)
        df.imag = 0.0+(df.imag>threshold_value)*(df.imag - threshold_value) +(df.imag<= - threshold_value)*(df.imag+threshold_value) 
#                 df =     sy
        # end of shrinkage

        bb[0] += zz[0] - dd[0] 
        bb[1] += zz[1] - dd[1] 
        bf += zf - df 
#                 self._update_b() # update b based on the current u

        AHyk = AHyk - zf # Linearized Bregman iteration f^k+1 = f^k + f - Au
#             print(outer)

    return xkp1 #(u,u_stack)

def L1OLS(nufft, y, maxiter, rho ): # main function of solver

    mu = 1.0
    LMBD = rho*mu

    def AHA(x):
        x2 = nufft.selfadjoint(x)
        return x2
    def AH(y):
        x2 = nufft.adjoint(y)
        return x2
    
    uker = mu*_create_kspace_sampling_density(nufft)   - LMBD* _create_laplacian_kernel(nufft)
    
    AHy = AH(y)
    
    xkp1 = numpy.zeros_like(AHy)
    AHyk = numpy.zeros_like(AHy)
           
#         self._allo_split_variables()        
    zz= []
    bb = []
    dd = []
    d_indx, dt_indx = indxmap_diff(nufft.st['Nd'])
    z=numpy.zeros(nufft.st['Nd'], dtype = nufft.dtype, order='C')
    
    ndims = len(nufft.st['Nd'])
    
    for jj in range(    0,  ndims): # n_dims + 1 for wavelets
        
        zz += [z.copy(),]
        bb += [z.copy(),]
        dd +=  [z.copy(),]
    zf = z.copy()
    bf = z.copy()
    df = z.copy()

    n_dims = len(nufft.st['Nd'])#numpy.size(uf.shape)
    
    for outer in numpy.arange(0, maxiter):
#             for inner in numpy.arange(0,nInner):
            
        # solve Ku = rhs
            
        rhs = (mu*AHyk + # df - bf) +  # right hand side
                LMBD*(cDiff(dd[0] - bb[0],  dt_indx[0])) + 
                LMBD*(cDiff(dd[1] - bb[1],  dt_indx[1]))  )          
        # Note K = F' uker F
        # so K-1 ~ F
        xkp1 = nufft.k2xx(nufft.xx2k(rhs) / uker) 
#                 self._update_d(xkp1)

        zz[0] = cDiff(xkp1,  d_indx[0])
        zz[1] = cDiff(xkp1,  d_indx[1])
        zf = AHA(xkp1)  -AHy 

        '''
        soft-thresholding the edges
        '''

        s1 = zz[0] + bb[0]
        s2 = zz[1] + bb[1]
#         s = sum((self.zz[pj] + self.bb[pj])**2 for pj in range(0,n_dims))
        s = s1**2 + s2**2
        s = s**0.5 +1e-3

        threshold_value = 1/LMBD
        r =(s > threshold_value)*(s-threshold_value)/s#numpy.maximum(s - threshold_value ,  0.0)/s
        dd[0] = s1*r
        dd[1] = s2*r
#         df = zf+bf
        
#         threshold_value=1.0/mu
#     
#         df.real =0.0+ (df.real>threshold_value)*(df.real - threshold_value) +(df.real<= - threshold_value)*(df.real+threshold_value)
#         df.imag = 0.0+(df.imag>threshold_value)*(df.imag - threshold_value) +(df.imag<= - threshold_value)*(df.imag+threshold_value) 
#                 df =     sy
        # end of shrinkage

        bb[0] += zz[0] - dd[0] 
        bb[1] += zz[1] - dd[1] 
#         bf += zf - df 
#                 self._update_b() # update b based on the current u

        AHyk = AHyk - zf # Linearized Bregman iteration f^k+1 = f^k + f - Au
#             print(outer)

    return xkp1 #(u,u_stack)     
def solver(nufft,   y,  solver=None, *args, **kwargs):
    if ('cgs' == solver) or ('qmr' ==solver) or ('minres'==solver):
        raise TypeError(solver +' requires real symmetric matrix')
    

    if None ==  solver:
#         solver  =   'cg'
        print("solver must be one of the following solvers; \newline dc, cg, bicg, bicgstab, gmres, lgmres, L1OLS, L1LAD")
 

    if 'dc'   ==  solver:
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
#             try:
# 
#                 x = nufft.adjoint(nufft.st['W']*y)
# 
#             except: 

        nufft.st['W'] = nufft.pipe_density( *args, **kwargs)

        x = nufft.adjoint(nufft.st['W']*y)

        return x
    elif ('lsmr'==solver) or ('lsqr'==solver):
            """
            Assymetric matrix A
            Minimize L2 norm of |y-Ax|_2 or |y-Ax|_2+|x|_2
            Very stable 
            """
            A = nufft.st['p']
            methods={'lsqr':scipy.sparse.linalg.lsqr,
                                'lsmr':scipy.sparse.linalg.lsmr,}
            k2 = methods[solver](A,  y, *args, **kwargs)#,show=True)

 
            xx = nufft.k2xx(nufft.vec2k(k2[0]))
            x= xx/nufft.st['sn']
            return x#, k2[1:]        
    elif 'L1OLS' == solver:
        return  L1OLS(nufft, y, *args, **kwargs)
    elif 'L1LAD' == solver:
        return  L1LAD(nufft, y, *args, **kwargs)
    else:
        """
        Hermitian matrix A
        cg: conjugate gradient
        bicgstab: biconjugate gradient stablizing
        bicg: biconjugate gradient
        gmres: 
        lgmres:
        """
        A =nufft.st['p'].getH().dot(nufft.st['p'])
        
        methods={'cg':scipy.sparse.linalg.cg,   
                             'bicgstab':scipy.sparse.linalg.bicgstab, 
                             'bicg':scipy.sparse.linalg.bicg, 
#                                  'cgs':scipy.sparse.linalg.cgs, 
                             'gmres':scipy.sparse.linalg.gmres, 
                             'lgmres':scipy.sparse.linalg.lgmres, 
#                                  'minres':scipy.sparse.linalg.minres, 
#                                  'qmr':scipy.sparse.linalg.qmr, 
                             }
        k2 = methods[solver](A,  nufft.st['p'].getH().dot(y), *args, **kwargs)#,show=True)


        xx = nufft.k2xx(nufft.vec2k(k2[0]))
        x= xx/nufft.st['sn']
        return x#     , k2[1:]       