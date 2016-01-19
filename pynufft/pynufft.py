'''@package docstring
@author: Jyh-Miin Lin
@address: jyhmiinlin@gmail.com
@date: 2016/1/18
   
    Pythonic non-uniform fast Fourier transform (NUFFT)
    Please cite http://scitation.aip.org/content/aapm/journal/medphys/42/10/10.1118/1.4929560 
    
    This algorithm was proposed by
    Fessler JA, Sutton BP. Nonuniform fast Fourier transforms using min-max interpolation. IEEE Trans Signal Process 2003;51(2):560-574.

    Installation: 
    pip install pynufft
    
    test:
    import pynufft.pynufft
    pynufft.pynufft.test_1D()
    pynufft.pynufft.test_2D()
    
    Check test_1D() and test_2D() for usage
    
'''

import numpy
import scipy.sparse

import numpy.fft
 

 
dtype = numpy.complex64
 


import scipy.signal

import scipy.linalg

try: 
    xrange 
except NameError: 
    xrange = range

def pipe_density(V): 
    '''
    An lsmr iterative solution. 
    '''
    
    V1=V.getH()
#     E = V.dot( V1.dot(    W   )   )
#     W = W*(E+1.0e-17)/(E*E+1.0e-17)    
    b = numpy.ones( (V.get_shape()[0] ,1) ,dtype  = numpy.complex64)  
    from scipy.sparse.linalg import lsqr, lsmr
        
#     x1 =  lsqr(V, b , iter_lim=20, calc_var = True, damp = 0.001)
    x1 =  lsmr(V, b , maxiter=100,  damp = 0.001)
    
    my_k_dens = x1[0]    # the first element is the answer
    
#     tmp_W =  lsqr(V1, my_k_dens, iter_lim=20, calc_var = True, damp = 0.001)
    tmp_W =  lsmr(V1, my_k_dens , maxiter=10,  damp = 0.001)
    
    W = numpy.reshape( tmp_W[0], (V.get_shape()[0] ,1),order='F' ) # reshape vector

 
    return W
def checker(input_var,desire_size):
    '''
    check if debug = 1
    '''

    if input_var is None:
        print('input_variable does not exist!')
      
    if desire_size is None:
        print('desire_size does not exist!')  
             
    dd=numpy.size(desire_size)
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
       
    if numpy.mod(numpy.prod(dims),numpy.prod(desire_size)) != 0:
        print('input signal shape is not multiples of desired size!')
        
def dirichlet(x):
    return numpy.sinc(x)

def outer_sum(xx,yy):
    nx=numpy.size(xx)
    ny=numpy.size(yy)
    
    arg1 = numpy.tile(xx,(ny,1)).T
    arg2 = numpy.tile(yy,(nx,1))
    #cc = arg1 + arg2
    
    return arg1 + arg2

def nufft_offset(om, J, K):
    '''
    For every om points(outside regular grids), find the nearest
    central grid (from Kd dimension)  
    '''
    gam = 2.0*numpy.pi/(K*1.0);
    k0 = numpy.floor(1.0*om / gam - 1.0*J/2.0) # new way
    return k0

def nufft_alpha_kb_fit(N, J, K):
    '''
    find out parameters alpha and beta
    of scaling factor st['sn']
    Note, when J = 1 , alpha is hardwired as [1,0,0...] 
    (uniform scaling factor)
    '''
    beta=1
    #chat=0
    Nmid=(N-1.0)/2.0
    
    if N > 40:
        #empirical L
        L= 13
    else:
        L=numpy.ceil(N/3)
        
    nlist = numpy.arange(0,N)*1.0-Nmid
#    print(nlist)
    (kb_a,kb_m)=kaiser_bessel('string', J, 'best', 0, K/N)
#    print(kb_a,kb_m)
    if J > 1:
        sn_kaiser = 1 / kaiser_bessel_ft(nlist/K, J, kb_a, kb_m, 1.0)
    elif J ==1:  # cases on grids
        sn_kaiser = numpy.ones((1,N),dtype=dtype)
#            print(sn_kaiser)
    gam = 2*numpy.pi/K;
    X_ant =beta*gam*nlist.reshape((N,1),order='F')
    X_post= numpy.arange(0,L+1)
    X_post=X_post.reshape((1,L+1),order='F') 
    X=numpy.dot(X_ant, X_post) # [N,L]
    X=numpy.cos(X)
    sn_kaiser=sn_kaiser.reshape((N,1),order='F').conj()
#    print(numpy.shape(X),numpy.shape(sn_kaiser))
#   print(X)
    #sn_kaiser=sn_kaiser.reshape(N,1)
    X=numpy.array(X,dtype=dtype)
    sn_kaiser=numpy.array(sn_kaiser,dtype=dtype)
    coef = numpy.linalg.lstsq(X,sn_kaiser)[0] #(X \ sn_kaiser.H);
#            print('coef',coef)
    #alphas=[]
    alphas=coef
    if J > 1:
        alphas[0]=alphas[0]
        alphas[1:]=alphas[1:]/2.0      
    elif J ==1: # cases on grids
        alphas[0]=1.0
        alphas[1:]=0.0                  
    alphas=numpy.real(alphas)
    return (alphas, beta)

def kaiser_bessel(x, J, alpha, kb_m, K_N):
    if K_N != 2 : 
        kb_m = 0
        alpha = 2.34 * J
    else:
        kb_m = 0    # hardwritten in Fessler's code, because it was claimed as the best!
        jlist_bestzn={2: 2.5, 
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
                        16: 2.33 }
        if J in jlist_bestzn:
#            print('demo key',jlist_bestzn[J])
            alpha = J*jlist_bestzn[J]
            #for jj in tmp_key:
            #tmp_key=abs(tmp_key-J*numpy.ones(len(tmp_key)))
#            print('alpha',alpha)
        else:
            #sml_idx=numpy.argmin(J-numpy.arange(2,17))
            tmp_key=(jlist_bestzn.keys())
            min_ind=numpy.argmin(abs(tmp_key-J*numpy.ones(len(tmp_key))))
            p_J=tmp_key[min_ind]
            alpha = J * jlist_bestzn[p_J]
            print('well, this is not the best though',alpha)
    kb_a=alpha
    return (kb_a, kb_m)

def kaiser_bessel_ft(u, J, alpha, kb_m, d):
    '''
    interpolation weight for given J/alpha/kb-m 
    '''
#     import types
    
    # scipy.special.jv (besselj in matlab) only accept complex
#     if u is not types.ComplexType:
#         u=numpy.array(u,dtype=numpy.complex64)
    u = u*(1.0+0.0j)
    import scipy.special
    z = numpy.sqrt( (2*numpy.pi*(J/2)*u)**2.0 - alpha**2.0 );
    nu = d/2 + kb_m;
    y = ((2*numpy.pi)**(d/2))* ((J/2)**d) * (alpha**kb_m) / scipy.special.iv(kb_m, alpha) * scipy.special.jv(nu, z) / (z**nu)
    y = numpy.real(y);
    return y

def nufft_scale1(N, K, alpha, beta, Nmid):
    '''
    calculate image space scaling factor
    '''
#     import types
#     if alpha is types.ComplexType:
    alpha=numpy.real(alpha)
#         print('complex alpha may not work, but I just let it as')
        
    L = len(alpha) - 1
    if L > 0:
        sn = numpy.zeros((N,1))
        n = numpy.arange(0,N).reshape((N,1),order='F')
        i_gam_n_n0 = 1j * (2*numpy.pi/K)*( n- Nmid)* beta
        for l1 in xrange(-L,L+1):
            alf = alpha[abs(l1)];
            if l1 < 0:
                alf = numpy.conj(alf)
            sn = sn + alf*numpy.exp(i_gam_n_n0 * l1)
    else:
        sn = numpy.dot(alpha , numpy.ones((N,1),dtype=numpy.float32))
    return sn

def nufft_scale(Nd, Kd, alpha, beta):
    dd=numpy.size(Nd)
    Nmid = (Nd-1)/2.0
    if dd == 1:
        sn = nufft_scale1(Nd, Kd, alpha, beta, Nmid);
#    else:
#        sn = 1
#        for dimid in numpy.arange(0,dd):
#            tmp =  nufft_scale1(Nd[dimid], Kd[dimid], alpha[dimid], beta[dimid], Nmid[dimid])
#            sn = numpy.dot(list(sn), tmp.H)
    return sn
def mat_inv(A):
    '''
    Abstraction for Penrose-Moore pseudo-inverse
    '''
    B = scipy.linalg.pinv2(A)  
 
    
    return B

def nufft_T(N, J, K , alpha, beta):
    '''
     equation (29) and (26)Fessler's paper
     create the overlapping matrix CSSC (diagonal dominent matrix)
     of J points 
     and then find out the pseudo-inverse of CSSC 
     '''

#     import scipy.linalg
    L = numpy.size(alpha) - 1
#     print('L = ', L, 'J = ',J, 'a b', alpha,beta )
    cssc = numpy.zeros((J,J));
    [j1, j2] = numpy.mgrid[1:J+1, 1:J+1]
    overlapping_mat = j2 - j1 
    
    for l1 in xrange(-L,L+1):
        for l2 in xrange(-L,L+1):
            alf1 = alpha[abs(l1)]
#             if l1 < 0: alf1 = numpy.conj(alf1)
            alf2 = alpha[abs(l2)]
#             if l2 < 0: alf2 = numpy.conj(alf2)
            tmp = overlapping_mat + beta * (l1 - l2)
#             tmp = numpy.sinc(1.0*tmp/(1.0*K/N)) # the interpolator
            tmp = dirichlet(1.0*tmp/(1.0*K/N))
            cssc = cssc + alf1 * numpy.conj(alf2) * tmp;

#     cssc = scipy.linalg.inv(cssc )
#     q,r  = scipy.linalg.qr(cssc,mode='full')
#     cssc =  r.conj().T.dot(scipy.linalg.inv(q))
#     cssc =  scipy.linalg.inv(r).dot(q.T.conj())
#     T,Z = scipy.linalg.schur(cssc)
#     cssc = Z.conj().T.dot(scipy.linalg.inv(T))*Z
    
    return mat_inv(cssc) 

def nufft_r(om, N, J, K, alpha, beta):
    '''
    equation (30) of Fessler's paper
    
    '''
  
    M = numpy.size(om) # 1D size
    gam = 2.0*numpy.pi / (K*1.0)
    nufft_offset0 = nufft_offset(om, J, K) # om/gam -  nufft_offset , [M,1]
    dk = 1.0*om/gam -  nufft_offset0 # om/gam -  nufft_offset , [M,1]
    arg = outer_sum( -numpy.arange(1,J+1)*1.0, dk)
    L = numpy.size(alpha) - 1
#     print('alpha',alpha)
    rr = numpy.zeros((J,M))
#     if L > 0: 
#         rr = numpy.zeros((J,M))
#                if J > 1:
    for l1 in xrange(-L,L+1):
        alf = alpha[abs(l1)]*1.0
        if l1 < 0: alf = numpy.conj(alf) 
#             r1 = numpy.sinc(1.0*(arg+1.0*l1*beta)/(1.0*K/N))
        r1 = dirichlet(1.0*(arg+1.0*l1*beta)/(1.0*K/N))
        rr = 1.0*rr + alf * r1;            # [J,M]
#                elif J ==1:
#                    rr=rr+1.0
#     else: #L==0
# #         rr = numpy.sinc(1.0*(arg+1.0*l1*beta)/(1.0*K/N))
#         rr = dirichlet(1.0*(arg+1.0*l1*beta)/(1.0*K/N))
    return (rr,arg)
def SC(om, N, J, K, alpha, beta):
    '''
    equation (30) of Fessler's paper
    
    '''
  
    M = numpy.size(om) # 1D size
    gam = 2.0*numpy.pi / (K*1.0)
    nufft_offset0 = nufft_offset(om, J, K) # om/gam -  nufft_offset , [M,1]
    dk = 1.0*om/gam -  nufft_offset0 # om/gam -  nufft_offset , [M,1] phase shifts for M points
    arg = outer_sum( -numpy.arange(1,J+1)*1.0, dk) # phase shifts for JxM points, [J, M]
#     print(numpy.shape(arg))
    L = numpy.size(alpha) - 1
#     print('alpha',alpha)
    rr = numpy.zeros((J,M))
#     if L > 0: 
#         rr = numpy.zeros((J,M))
#                if J > 1:
    for l1 in xrange(-L,L+1):
        alf = alpha[abs(l1)]*1.0
        if l1 < 0: alf = numpy.conj(alf) 
#             r1 = numpy.sinc(1.0*(arg+1.0*l1*beta)/(1.0*K/N))
#         r1 = dirichlet(1.0*( 1.0*l1*beta)/(1.0*K/N))
        r1 = dirichlet(1.0*(1.0*l1*beta)/(1.0*K/N))
        rr = 1.0*rr + alf * r1;            # [J,M]
#                elif J ==1:
#                    rr=rr+1.0
#     else: #L==0
# #         rr = numpy.sinc(1.0*(arg+1.0*l1*beta)/(1.0*K/N))
#         rr = dirichlet(1.0*(arg+1.0*l1*beta)/(1.0*K/N))
    SC = rr.conj().T # [M, J]
    return (SC,arg)
def block_outer_prod(x1, x2):
    '''
    multiply interpolators of different dimensions
    '''
    (J1,M)=x1.shape
    (J2,M)=x2.shape
#    print(J1,J2,M)
    xx1 = x1.reshape((J1,1,M),order='F') #[J1 1 M] from [J1 M]
    xx1 = numpy.tile(xx1,(1,J2,1)) #[J1 J2 M], emulating ndgrid
    xx2 = x2.reshape((1,J2,M),order='F') # [1 J2 M] from [J2 M]
    xx2 = numpy.tile(xx2,(J1,1,1)) # [J1 J2 M], emulating ndgrid
    
#     ang_xx1=xx1/numpy.abs(xx1)
#     ang_xx2=xx2/numpy.abs(xx2)
    
    y= xx1* xx2
#     y= ang_xx1*ang_xx2*numpy.sqrt(xx1*xx1.conj())*numpy.sqrt( xx2*xx2.conj())
    
    # RMS
    return y # [J1 J2 M]

def block_outer_sum(x1, x2):
    (J1,M)=x1.shape
    (J2,M)=x2.shape
#    print(J1,J2,M)
    xx1 = x1.reshape((J1,1,M),order='F') #[J1 1 M] from [J1 M]
    xx1 = numpy.tile(xx1,(1,J2,1)) #[J1 J2 M], emulating ndgrid
    xx2 = x2.reshape((1,J2,M),order='F') # [1 J2 M] from [J2 M]
    xx2 = numpy.tile(xx2,(J1,1,1)) # [J1 J2 M], emulating ndgrid
    y= xx1+ xx2
    return y # [J1 J2 M]

def crop_slice_ind(Nd):
    return [slice(0, Nd[_ss]) for _ss in xrange(0,len(Nd))]

class pynufft:

    def __init__(self,om, Nd, Kd,Jd,n_shift ):
        '''
       constructor of pyNufft

        
        '''       

        '''
        Constructor: Start from here
        '''



        if n_shift == None:
#             n_shift=tuple(numpy.array(Nd)/2)
            n_shift=tuple(numpy.array(Nd)*0)
        self.debug = 0 # debug          
        Nd=tuple(Nd) # convert Nd to tuple for consistent structure 
        Jd=tuple(Jd) # convert Jd to tuple for consistent structure
        Kd=tuple(Kd) # convert Kd to tuple for consistent structure
        # n_shift: the fftshift position, it must be at center
#         n_shift=tuple(numpy.array(Nd)/2)
          

        # dimensionality of input space (usually 2 or 3)
        dd=numpy.size(Nd)
        
    #=====================================================================
    # check input errors
    #=====================================================================
        st={}
        ud={}
        kd={}
 
        st['n_shift']=n_shift
    #=======================================================================
    # First, get alpha and beta: the weighting and freq
    # of formula (28) of Fessler's paper 
    # in order to create slow-varying image space scaling 
    #=======================================================================
        for dimid in xrange(0,dd):

            (tmp_alpha,tmp_beta)=nufft_alpha_kb_fit(Nd[dimid], Jd[dimid], Kd[dimid])
                
            st.setdefault('alpha', []).append(tmp_alpha)
            st.setdefault('beta', []).append(tmp_beta)

        
        st['tol'] = 0
        st['Jd'] = Jd
        st['Nd'] = Nd
        st['Kd'] = Kd
        M = om.shape[0]
        st['M'] = M
        st['om'] = om
        st['sn'] = numpy.array(1.0+0.0j)
        dimid_cnt=1 
    #=======================================================================
    # create scaling factors st['sn'] given alpha/beta
    # higher dimension implementation
    #=======================================================================
        for dimid in xrange(0,dd):
            tmp = nufft_scale(Nd[dimid], Kd[dimid], st['alpha'][dimid], st['beta'][dimid])

            dimid_cnt=Nd[dimid]*dimid_cnt
    #=======================================================================
    # higher dimension implementation: multiply over all dimension
    #=======================================================================

            st['sn'] =  numpy.dot(st['sn'] , tmp.T )
            st['sn'] =   numpy.reshape(st['sn'],(dimid_cnt,1),order='F') # JML do not apply scaling

        #=======================================================================
        # if numpy.size(Nd) > 1: 
        #=======================================================================
            # high dimension, reshape for consistent out put
            # order = 'F' is for fortran order otherwise it is C-type array 
        st['sn'] = st['sn'].reshape(Nd,order='F') # [(Nd)]
        #=======================================================================
        # else:
        #     st['sn'] = numpy.array(st['sn'],order='F')
#         #=======================================================================
            
        st['sn']=numpy.real(st['sn']) # only real scaling is relevant 

        # [J? M] interpolation coefficient vectors.  will need kron of these later
        for dimid in xrange(0,dd): # loop over dimensions
            N = Nd[dimid]
            J = Jd[dimid]
            K = Kd[dimid]
            alpha = st['alpha'][dimid]
            beta = st['beta'][dimid]
            #===================================================================
            # formula 29 , 26 of Fessler's paper
            #===================================================================
            
            T = nufft_T(N, J, K, alpha, beta) # pseudo-inverse of CSSC using large N approx [J? J?]
            #==================================================================
            # formula 30  of Fessler's paper
            #==================================================================

            (r,arg)=  nufft_r(om[:,dimid], N, J, K, alpha, beta) # large N approx [J? M]
            
            #==================================================================
            # formula 25  of Fessler's paper
            #==================================================================            
            c=numpy.dot(T,r)

            #===================================================================
            # grid intervals in radius
            #===================================================================
            gam = 2.0*numpy.pi/(K*1.0);
            
            phase_scale = 1.0j * gam * (N-1.0)/2.0
            phase = numpy.exp(phase_scale * arg) # [J? M] linear phase
            ud[dimid] = phase * c
            # indices for the [J? M] interpolation kernel
            # FORMULA 7
            koff=nufft_offset(om[:,dimid], J, K)
            # FORMULA 9, find the indexes on Kd grids, of each M point
            kd[dimid]= numpy.mod(outer_sum( numpy.arange(1,J+1)*1.0, koff),K)
            if self.debug > 0:
                print('kd[',dimid,']',kd[dimid].shape)
#             phase_kd[dimid] = - ( numpy.arange(0,1.0, 1.00/K) - 0.5 )*2.0*numpy.pi*n_shift[dimid]# new phase2
             
            if dimid > 0: # trick: pre-convert these indices into offsets!
    #            ('trick: pre-convert these indices into offsets!')
                kd[dimid] = kd[dimid]*numpy.prod(Kd[0:dimid])-1

        kk = kd[0] # [J1 M] # pointers to indices
        uu = ud[0] # [J1 M]
        Jprod = Jd[0]
        Kprod = Kd[0]
            
        for dimid in xrange(1,dd):
            Jprod = numpy.prod(Jd[:dimid+1])
            Kprod = numpy.prod(Kd[:dimid+1])
            if self.debug > 0:
                print('Kprod',Kprod)
            kk = block_outer_sum(kk, kd[dimid])+1 # outer sum of indices
            kk = kk.reshape((Jprod, M),order='F')
 
              
            uu = block_outer_prod(uu, ud[dimid]) # outer product of coefficients
            uu = uu.reshape((Jprod, M),order='F')
 
        uu = uu.conj()#*numpy.tile(phase,[numpy.prod(Jd),1]) #    product(Jd)xM
        mm = numpy.arange(0,M) # indices from 0 to M-1
        mm = numpy.tile(mm,[numpy.prod(Jd),1]) #    product(Jd)xM

        st['p0'] = scipy.sparse.coo_matrix( # build sparse matrix from uu, mm, kk
                                          (     numpy.reshape(uu,(Jprod*M,),order='F'), # convert array to list
                                                (numpy.reshape(mm,(Jprod*M,),order='F'), # row indices, from 1 to M convert array to list
                                                 numpy.reshape(kk,(Jprod*M,),order='F') # colume indices, from 1 to prod(Kd), convert array to list
                                                           )
                                                    ),
                                        shape=(M,numpy.prod(Kd)) # shape of the matrix
                                          ).tocsc() #  order = 'F'  
        self.st=st
        self.Nd = self.st['Nd'] # backup
        self.sn = self.st['sn'] # backup
        self.linear_phase( n_shift  ) # calculate the linear phase thing
#         self.pipe_density() # recalculate the density compensation

       

    def pipe_density(self):
        '''
        Create the density function by the iterative solution 
        '''
        self.st['W'] = pipe_density(self.st['p'])
        self.st['w'] =  numpy.abs(( self.st['p'].conj().T.dot(numpy.ones(self.st['W'].shape,dtype = numpy.float32))))#**2) ))
        tmp_max = numpy.max(self.st['w'])
        self.st['w'] = self.st['w'] /tmp_max

        if self.debug > 0:
            print('shape of tmp',numpy.shape(self.st['w'] ))

    def linear_phase(self, n_shift ):
        '''
        Select the center of FOV 
        '''
        om = self.st['om']
        M = self.st['M']
        phase=numpy.exp(1.0j* numpy.sum(om*numpy.tile( tuple( numpy.array(n_shift) + numpy.array(self.st['Nd'])/2  ),(M,1)) , 1))  # add-up all the linear phasees in all axes,
        
        self.st['p'] = scipy.sparse.diags(phase ,0).dot(self.st['p0'] ) # multiply the diagonal, linear phase before the gridding matrix
  
    def forward(self,x):
        '''
        foward(x): forward transform of pynufft
        
        Compute dd-dimensional Non-uniform transform of signal/image x
        where d is the dimension of the data x.
        
        INPUT: 
          case 1:  x: ndarray, [Nd[0], Nd[1], ... , Kd[dd-1] ]
          case 2:  x: ndarray, [Nd[0], Nd[1], ... , Kd[dd-1], Lprod]
            
        OUTPUT: 
          case 1: X: ndarray, [M, 1], where M is the number of non-uniform points M =st['M']
          case 2: X: ndarray, [M, Lprod] (Lprod=1 in case 1)
                    where M is the number of non-uniform points M =st['M']
        '''
        
        st=self.st
        Nd = st['Nd']

        dd = numpy.size(Nd)

        # exceptions
        if self.debug==0:
            pass
        else:
            checker(x,Nd)
        
        if numpy.ndim(x) == dd:
            Lprod = 1
        elif numpy.ndim(x) > dd: # multi-channel data
            Lprod = numpy.size(x)/numpy.prod(Nd)
        '''
        Now transform Nd grids to Kd grids(not be reshaped)
        '''
        Xk=self.Nd2Kd(x, 1)
        
        # interpolate using precomputed sparse matrix
        if Lprod > 1:
            X = numpy.reshape(st['p'].dot(Xk),(st['M'],)+( Lprod,),order='F')
        else:
            X = numpy.reshape(st['p'].dot(Xk),(st['M'],1),order='F')
        if self.debug==0:
            pass
        else:
            checker(X,st['M']) # check output
        return X
    
    def adjoint(self,X):
        '''
        adjoint(x): adjoint operator of pynufft
        
        from [M x Lprod] non-uinform points, compute its adjoint of 
        Non-uniform Fourier transform 

        
        INPUT: 
          X: ndarray, [M, Lprod] (Lprod=1 in case 1)
            where M =st['M']
          
        OUTPUT: 
          x: ndarray, [Nd[0], Nd[1], ... , Kd[dd-1], Lprod]
            
        '''
#     extract attributes from structure
        st=self.st

        Nd = st['Nd']
#         Kd = st['Kd']
        if self.debug==0:
            pass
        else:
            checker(X,st['M']) # check X of correct shape


        Xk_all = st['p'].getH().dot(X) 
        # Multiply X with interpolator st['p'] [prod(Kd) Lprod]
        '''
        Now transform Kd grids to Nd grids(not be reshaped)
        '''
        x = self.Kd2Nd(Xk_all, 1)
        
        if self.debug==0:
            pass
        else:        
            checker(x,Nd) # check output
        
        return x

    def Nd2Kd(self,x, weight_flag):
        '''
        Now transform Nd grids to Kd grids(not be reshaped)
        
        '''
        #print('661 x.shape',x.shape)        
        
        st=self.st
        Nd = st['Nd']
        Kd = st['Kd']
        dims = numpy.shape(x)
        dd = numpy.size(Nd)

        if self.debug==0:
            pass
        else:
            checker(x,Nd)

        if numpy.ndim(x) == dd:   
            if weight_flag == 1:
                x = x * st['sn']
            else:
                pass 

            Xk = self.emb_fftn(x, Kd,range(0,dd))

            Xk = numpy.reshape(Xk, (numpy.prod(Kd),),order='F')
            
        else:# otherwise, collapse all excess dimensions into just one
            xx = numpy.reshape(x, [numpy.prod(Nd), numpy.prod(dims[(dd):])],order='F')  # [*Nd *L]
            L = numpy.shape(xx)[1]
    #        print('L=',L)
    #        print('Lprod',Lprod)
            Xk = numpy.zeros( (numpy.prod(Kd), L),dtype=dtype) # [*Kd *L]
            for ll in xrange(0,L):
                xl = numpy.reshape(xx[:,ll], Nd,order='F') # l'th signal
                if weight_flag == 1:
                    xl = xl * st['sn'] # scaling factors
                else:
                    pass
                Xk[:,ll] = numpy.reshape(self.emb_fftn(xl, Kd,range(0,dd)),
                                         (numpy.prod(Kd),),order='F')
        if self.debug==0:
            pass
        else:                
            checker(Xk,numpy.prod(Kd))        
        return Xk
    
    def Kd2Nd(self,Xk_all,weight_flag):
        
        st=self.st

        Nd = st['Nd']
        Kd = st['Kd']
        dd = len(Nd)
        if self.debug==0:
            pass
        else:
            checker(Xk_all,numpy.prod(Kd)) # check X of correct shape

        dims = numpy.shape(Xk_all)
        Lprod= numpy.prod(dims[1:]) # how many channel * slices

        if numpy.size(dims) == 1:
            Lprod = 1
        else:
            Lprod = dims[1]  
            
                  
        x=numpy.zeros(Kd+(Lprod,),dtype=dtype)  # [*Kd *L]

#         if Lprod > 1:

        Xk = numpy.reshape(Xk_all, Kd+(Lprod,) , order='F')
        
        for ll in xrange(0,Lprod):  # ll = 0, 1,... Lprod-1
            x[...,ll] =  self.emb_ifftn(Xk[...,ll],Kd,range(0,dd))#.flatten(order='F'))

        x = x[crop_slice_ind(Nd)]


        if weight_flag == 0:
            pass
            
        else: #weight_flag =1 scaling factors
            snc = st['sn'].conj()
            for ll in xrange(0,Lprod):  # ll = 0, 1,... Lprod-1    
                x[...,ll] = x[...,ll]*snc  #% scaling factors

        if self.debug==0:
            pass # turn off checker
        else:
            checker(x,Nd) # checking size of x divisible by Nd
        return x
      
    def emb_fftn(self, input_x, output_dim, act_axes):
        '''
        The abstraction for fast Fourier transform
        '''
        output_x=numpy.zeros(output_dim, dtype=dtype)

        output_x[crop_slice_ind(input_x.shape)] = input_x



        output_x=numpy.fft.fftn(output_x, output_dim, act_axes)

        return output_x
  
    def emb_ifftn(self, input_x, output_dim, act_axes):
        '''
        The abstraction for inverse fast Fourier transform
        '''
        output_x=numpy.zeros(output_dim, dtype=dtype)

        output_x[crop_slice_ind(input_x.shape)] = input_x 
 
        output_x=numpy.fft.ifftn(output_x, output_dim, act_axes)

        return output_x 
def test_1D():
# import several modules
    import numpy 
    import matplotlib.pyplot

    import os
    path = os.path.abspath(__file__)
    dir_path = os.path.dirname(path)

#create 1D curve from 2D image
    image = numpy.loadtxt(dir_path+'/phantom_256_256.txt') 
    profile_1d = image[:,128]
#determine the location of samples
    om = numpy.loadtxt(dir_path+'/om1D.txt')[0:]
    om = numpy.reshape(om,(numpy.size(om),1),order='F')
# reconstruction parameters
    Nd =(256,) # image space size
    Kd =(512,) # k-space size
      
    Jd =(6,) # interpolation size
# initiation of the object
    NufftObj = pynufft(om, Nd,Kd,Jd,None)
# simulate "data"
    data= NufftObj.forward(profile_1d)
#adjoint(reverse) of the forward transform


    NufftObj.pipe_density() # recalculate the density compensation
    
    profile_distorted= NufftObj.adjoint(data*NufftObj.st['W'])[:,0]

 
#Showing histogram of sampling locations
#     matplotlib.pyplot.hist(om,20)
#     matplotlib.pyplot.title('histogram of the sampling locations')
#     matplotlib.pyplot.show()
#show reconstruction
    matplotlib.pyplot.subplot(1,2,1)
 
    matplotlib.pyplot.plot(profile_1d)
    matplotlib.pyplot.title('original') 
    matplotlib.pyplot.ylim([0,1]) 
            

             
    matplotlib.pyplot.subplot(1,2,2)
    matplotlib.pyplot.plot(numpy.abs(profile_distorted)) 
    matplotlib.pyplot.title('profile_distorted')
 
    matplotlib.pyplot.show()  
         
    
def test_2D():
 
    import numpy 
    import matplotlib.pyplot
#     import os
#     path = os.path.dirname(self.__file__)
    import os
    path = os.path.abspath(__file__)
    dir_path = os.path.dirname(path)
#     print(dir_path)
    cm = matplotlib.cm.gray
    # load example image    
 
    image = numpy.loadtxt(dir_path+'/phantom_256_256.txt') 
    image[128,128]= 1.0   
    Nd =(256,256) # image space size
    Kd =(512,512) # k-space size   
    Jd =(6,6) # interpolator size
     
    # load k-space points
    om = numpy.loadtxt(dir_path+'/om.txt')
     
    #create object
  
#      
    NufftObj = pynufft(om, Nd,Kd,Jd, None)   
     
#     NufftObj = pynufft.pynufft.pynufft(om, Nd,Kd,Jd, None)   

    data= NufftObj.forward(image )
    
    NufftObj.pipe_density() # recalculate the density compensation
    
    image_blur = NufftObj.adjoint(data*NufftObj.st['W'])
#     image_recon = Normalize(image_recon)
 
    matplotlib.pyplot.plot(om[:,0],om[:,1],'x')
    matplotlib.pyplot.show()
 
    norm=matplotlib.colors.Normalize(vmin=0.0, vmax=1.0) 
    norm2=matplotlib.colors.Normalize(vmin=0.0, vmax=5.0e-2)
    # display images
    matplotlib.pyplot.subplot(2,2,1)
    matplotlib.pyplot.imshow(image,
                             norm = norm,cmap =cm,interpolation = 'nearest')
    matplotlib.pyplot.title('true image')   
   
    matplotlib.pyplot.subplot(2,2,2)
    matplotlib.pyplot.imshow(image_blur[:,:,0].real,
                              norm = norm2,cmap= cm,interpolation = 'nearest')
    matplotlib.pyplot.title('blurred image') 
 
     
    matplotlib.pyplot.show()      

if __name__ == '__main__':
#     import cProfile
#     test_1D()
    test_2D()   
     