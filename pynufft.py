###########################################################################################
# The MIT License (MIT)
# 
# Copyright (c) 2013 - 2016 pynufft team
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
###########################################################################################
import numpy
import scipy.sparse
import numpy.fft
import scipy.signal
import scipy.linalg
import scipy.special
# from helper import *


def dirichlet(x):
    return numpy.sinc(x)
 
def outer_sum(xx, yy):
    return numpy.add.outer(xx,yy)
#     nx = numpy.size(xx)
#     ny = numpy.size(yy)
# 
#     arg1 = numpy.tile(xx, (ny, 1)).T
#     arg2 = numpy.tile(yy, (nx, 1))
#     return arg1 + arg2
 
 
def nufft_offset(om, J, K):
    """
    For every om points(outside regular grids), find the nearest
    central grid (from Kd dimension)
    """
    gam = 2.0 * numpy.pi / (K * 1.0)
    k0 = numpy.floor(1.0 * om / gam - 1.0 * J / 2.0)  # new way
    return k0
 
 
def nufft_alpha_kb_fit(N, J, K, dtype):
    """
    find out parameters alpha and beta
    of scaling factor st['sn']
    Note, when J = 1 , alpha is hardwired as [1,0,0...]
    (uniform scaling factor)
    """
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
    coef = numpy.linalg.lstsq(X, sn_kaiser)[0]  # (X \ sn_kaiser.H);
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
    """
    interpolation weight for given J/alpha/kb-m
    """
 
    u = u * (1.0 + 0.0j)
    import scipy.special
    z = numpy.sqrt((2 * numpy.pi * (J / 2) * u) ** 2.0 - alpha ** 2.0)
    nu = d / 2 + kb_m
    y = ((2 * numpy.pi) ** (d / 2)) * ((J / 2) ** d) * (alpha ** kb_m) / \
        scipy.special.iv(kb_m, alpha) * scipy.special.jv(nu, z) / (z ** nu)
    y = numpy.real(y)
    return y
 
 
def nufft_scale1(N, K, alpha, beta, Nmid):
    """
    calculate image space scaling factor
    """
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
        sn = numpy.dot(alpha, numpy.ones((N, 1)))
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
    """
     equation (29) and (26)Fessler's paper
     create the overlapping matrix CSSC (diagonal dominent matrix)
     of J points
     and then find out the pseudo-inverse of CSSC """
 
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
            cssc = cssc + alf1 * alf2 * tmp
        
    return mat_inv(cssc)
 
 
def iterate_sum(rr, alf, r1):
    rr = rr + alf * r1
    return rr
 
 
def iterate_l1(L, alpha, arg, beta, K, N, rr):
    oversample_ratio = (1.0 * K / N)
    import time
    t0=time.time()
    for l1 in range(-L, L + 1):
        alf = alpha[abs(l1)] * 1.0
#         if l1 < 0:
#             alf = numpy.conj(alf)
    #             r1 = numpy.sinc(1.0*(arg+1.0*l1*beta)/(1.0*K/N))
        input_array = (arg + 1.0 * l1 * beta) / oversample_ratio
        r1 = dirichlet(input_array)
        rr = iterate_sum(rr, alf, r1)
 
    return rr
 
 
def nufft_r(om, N, J, K, alpha, beta):
    """
    equation (30) of Fessler's paper
 
    """
 
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
    """
    multiply the amplitudes along different axes
    """
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
    """
    update the new index after adding a new axis
    """
    (J1, M) = x1.shape
    (J2, M) = x2.shape
    xx1 = x1.reshape((J1, 1, M), order='F')  # [J1 1 M] from [J1 M]
    xx1 = numpy.tile(xx1, (1, J2, 1))  # [J1 J2 M], emulating ndgrid
    xx2 = x2.reshape((1, J2, M), order='F')  # [1 J2 M] from [J2 M]
    xx2 = numpy.tile(xx2, (J1, 1, 1))  # [J1 J2 M], emulating ndgrid
    y = xx1 + xx2
    return y  # [J1 J2 M]
 
 
def crop_slice_ind(Nd):
    """
    Return the "slice" of Nd size.
    In Python language, "slice" means the index of a matrix.
    Slice can provide a smaller "view" of a larger matrix. 
    """
    return [slice(0, Nd[ss]) for ss in range(0, len(Nd))]


class NUFFT:
    """
    The class pynufft computes Non-Uniform Fast Fourier Transform (NUFFT).
    Using Fessler and Sutton's min-max interpolator algorithm.
    
    "Fessler JA, Sutton BP. Nonuniform fast Fourier transforms using min-max interpolation. IEEE Trans Signal Process 2003;51(2):560-574."

    Methods
    ----------
    __init__() : constructor
        Input: 
            None
        Return: 
            pynufft instance
        Example: MyNufft = pynufft.pynufft()
    plan(om, Nd, Kd, Jd) : to plan the pynufft object
        Input:
            om: M * ndims array: The locations of M non-uniform points in the ndims dimension. Normalized between [-pi, pi]
            Nd: tuple with ndims elements. Image matrix size. Example: (256,256)
            Kd: tuple with ndims elements. Oversampling k-space matrix size. Example: (512,512)
            Jd: tuple with ndims elements. The number of adjacent points in the interpolator. Example: (6,6)
        Return:
            None
    forward(x) : perform NUFFT
        Input:
            x: numpy.array. The input image on the regular grid. The size must be Nd. 
        Output:
            y: M array.The output M points array.
             
    adjoint(y): adjoint NUFFT (Hermitian transpose (a.k.a. conjugate transpose) of NUFFT)
                Note: adjoint is not the inverse of forward NUFFT,
                because Non-uniform coordinates cause uneven density,
                which must be compensated by "density compensation (DC)"
                See inverse_DC() method
        Input:
            y: M array.The input M points array.
        Output:
            x: numpy.array. The output image on the regular grid.
            
    inverse_DC(y): inverse NUFFT using Pipe's sampling density compensation (James Pipe, Magn. Res. Med., 1999)
                Note: adjoint is not the inverse of forward NUFFT,
                because Non-uniform coordinates cause uneven density,
                which must be compensated by "density compensation (DC)"
                Note: A more accurate inverse NUFFT requires iterative reconstruction,
                    such as conjugate gradient method (CG) or other optimization methods. 
                
        Input: 
            y: M array.The input M points array.
        Output:
            x: numpy.array. The output image on the regular grid.

    """
    def __init__(self):
        """
        Construct the pynufft instance
        """
        self.dtype = numpy.complex64

    def plan(self, om, Nd, Kd, Jd):
        """
        Plan pyNufft
        Start from here
        om: M * ndims array: The locations of M non-uniform points in the ndims dimension. Normalized between [-pi, pi]
        Nd: tuple with ndims elements. Image matrix size. Example: (256,256)
        Kd: tuple with ndims elements. Oversampling k-space matrix size. Example: (512,512)
        Jd: tuple with ndims elements. The number of adjacent points in the interpolator. Example: (6,6)
        """
        
        self.debug = 0  # debug


        if type(Nd) != tuple:
            raise TypeError('Nd must be tuple, e.g. (256, 256)')

        if type(Kd) != tuple:
            raise TypeError('Kd must be tuple, e.g. (512, 512)')

        if type(Jd) != tuple:
            raise TypeError('Jd must be tuple, e.g. (6, 6)')

        if (len(Nd) != len(Kd)) | (len(Nd) != len(Jd))  | len(Kd) != len(Jd):
            raise KeyError('Nd, Kd, Jd must be in the same length, e.g. Nd=(256,256),Kd=(512,512),Jd=(6,6)')

        # dimensionality of input space (usually 2 or 3)
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
                Nd[dimid], Jd[dimid], Kd[dimid],
                                             self.dtype)
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
#             if len(om.shape):
#                 (r, arg) = nufft_r(om[:, ], N, J,
#                                K, alpha, beta)  # large N approx [J? M]
#             else:
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
#             if len(om.shape) == 1:
#                 koff = nufft_offset(om[:, ], J, K)
#             else:
            koff = nufft_offset(om[:, dimid], J, K)
            # FORMULA 9, find the indexes on Kd grids, of each M point
            kd[dimid] = numpy.mod(
                outer_sum(
                    numpy.arange(
                        1,
                        J + 1) * 1.0,
                    koff),
                K)
            if dimid > 0:  # trick: pre-convert these indices into offsets!
                #            ('trick: pre-convert these indices into offsets!')
                kd[dimid] = kd[dimid] * numpy.prod(Kd[0:dimid]) - 1

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

        # row indices, from 1 to M, convert array to list
        rowindx = numpy.reshape(mm, (Jprod * M, ), order='F')

        # colume indices, from 1 to prod(Kd), convert array to list
        colindx = numpy.reshape(kk, (Jprod * M, ), order='F')

        # The shape of sparse matrix
        csrshape = (M, numpy.prod(Kd))

        # Build sparse matrix (interpolator)
        st['p'] = scipy.sparse.csr_matrix((csrdata, (rowindx, colindx)),
                                           shape=csrshape)#.tocsr()
        # Note: the sparse matrix requires the following linear phase,
        #       which moves the image to the center of the image

        self.st = st
        self.Nd = self.st['Nd']  # backup
        self.sn = self.st['sn']  # backup
        self.ndims=len(self.st['Nd']) # dimension
        self.linear_phase(n_shift)  
        
        # calculate the linear phase thing
        
        self.st['W'] = self.pipe_density()
 
    def pipe_density(self):
        """
        Create the density function by iterative solution
        Generate pHp matrix
        """
#         W = pipe_density(self.st['p'])
        # sampling density function
        
        W = numpy.ones((self.st['M'],),dtype=self.dtype)
        V1= self.st['p'].getH()
    #     VVH = V.dot(V.getH()) 
        
        for pp in range(0,1):
#             E = self.st['p'].dot(V1.dot(W))
            E = self.forward(self.adjoint(W))
            W = W/E
        
#         pHp = V1.dot(self.st['p'])
        # Precomputing the regridding matrix * interpolation matrix
#         W_diag = scipy.sparse.diags(W,0, dtype=dtype,format="csr")
#         pHWp = V1.dot(W_diag.dot(self.st['p']))
        
        return W
        # density of the k-space, reduced size

    def linear_phase(self, n_shift):
        """
        This method shifts the interpolation matrix self.st['p0']
        and create a shifted interpolation matrix self.st['p']
        
        Parameters
        ----------
        n_shift : tuple
            Input shifts. integers 
    
        Raises
        ------
        ValueError
            If `s` and `axes` have different length.
        IndexError
            If an element of `axes` is larger than than the number of axes of `a`.        
            
        """
        om = self.st['om']
        M = self.st['M']
        final_shifts = tuple(
            numpy.array(n_shift) +
            numpy.array(
                self.st['Nd']) /
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

        self.st['p'] = scipy.sparse.diags(phase, 0).dot(self.st['p'])
        return 0  # shifted sparse matrix


    def forward(self, x):
        """
        This method computes the Non-Uniform Fast Fourier Transform.
        input:
            x: Nd array
        output:
            y: (M,) array
        """
        y = self.k2y(self.xx2k(self.x2xx(x)))

        return y

    def adjoint(self, y):
        """
        adjoint method (not inverse, see inverse_DC() method) computes the adjoint transform
        (conjugate transpose, or Hermitian transpose) of forward
        Non-Uniform Fast Fourier Transform.
        input:
            y: non-uniform data, (M,) array
        output:
            x: adjoint image, Nd array
        """        
        x = self.xx2x(self.k2xx(self.y2k(y)))

        return x
    def inverse_cg(self, y):
        '''
        conjugate inverse iteration
        '''
        
        b = self.st['p'].getH().dot(y)  
        
        A =  (self.st['p'].getH().dot(self.st['p'])).tocsc()
        A.eliminate_zeros()        
#        from scipy.sparse import csc_matrix, linalg as sla
#        lu = sla.spilu(A)
        import scipy.sparse.linalg as splinalg        
#         b2 = splinalg.spsolve(A, b, permc_spec="MMD_AT_PLUS_A")
        b2 = splinalg.cg(A, b )[0]
        
        x = (self.k2xx(self.vec2k(b2)))/self.st['sn'] 
         
        return x
    def selfadjoint(self, x):

        x2 = self.adjoint(self.forward(x))
        
#         x2 = self.xx2x(self.k2xx(self.k2k(self.xx2k(self.x2xx(x)))))
        
        return x2
    def forward_modulate_adjoint(self, x):
        
        x2 = self.adjoint(self.st['W']*self.forward(x))
        
#         x2 = self.xx2x(self.k2xx(self.k2k(self.xx2k(self.x2xx(x)))))
        
        return x2
    def inverse_DC(self,y):
        """
        reconstruction of non-uniform data y into image
        using density compensation method
        input: 
            y: (M,) array
        output:
            x2: Nd array
        """
        x2 = self.adjoint(self.st['W']*y)
        return x2
    def x2xx(self, x):
        """
        
        scaling of the image, generate Nd array
        Scaling of image is equivalent to convolution in k-space.
        Thus, scaling improves the accuracy of k-space interpolation.
          
        input:
            x: 2D image
        output:
            xx: scaled 2D image
        """
        xx = x * self.st['sn']
        return xx
    def y2k_DC(self,y):
        """
        Density compensated, adjoint transform of the non-uniform data (y: (M,) array) to k-spectrum (Kd array)
                Note: adjoint is not the inverse of forward NUFFT,
                because Non-uniform coordinates cause uneven density,
                which must be compensated by "density compensation (DC)"
        k-spectrum requires another numpy.fft.fftshift to move the k-space center.
        
        input:
            y: (M,) array
        output
            k: Kd array
        """
        k = self.y2k(y*self.st['W'])
        return k
    def xx2k(self, xx):
        """
        fft of the image
        input:
            xx:    scaled 2D image
        output:
            k:    k-space grid
        """
        dd = numpy.size(self.st['Kd'])      
        output_x = numpy.zeros(self.st['Kd'], dtype=self.dtype)
        output_x[crop_slice_ind(xx.shape)] = xx
        k = numpy.fft.fftn(output_x, self.st['Kd'], range(0, dd))

        return k
    def k2vec(self,k):
        
        k_vec = numpy.reshape(k, (numpy.prod(self.st['Kd']), ), order='F')
   
        return k_vec
    
    def vec2y(self,k_vec):
        """
        gridding: 
        
        """
        y = self.st['p'].dot(k_vec)
        
        return y
    def k2y(self, k):
        """
        2D k-space grid to 1D array
        input:
            k:    k-space grid,
        output:
            y: non-Cartesian data
        """
        
        y = self.vec2y(self.k2vec(k)) #numpy.reshape(self.st['p'].dot(Xk), (self.st['M'], ), order='F')
        
        return y
    def y2vec(self, y):
        """
       regridding non-uniform data, (unsorted vector)
        """
        k_vec = self.st['p'].getH().dot(y)
        
        return k_vec
    def vec2k(self, k_vec):
        """
        Sorting the vector to k-spectrum Kd array
        """
        k = numpy.reshape(k_vec, self.st['Kd'], order='F')
        
        return k
    
    def y2k(self, y):
        """
        Conjugate transpose of min-max interpolator
        
        input:
            y:    non-uniform data, (M,) array
        output:
            k:    k-spectrum on the Kd grid (Kd array)
        """
        
        k_vec = self.y2vec(y)
        
        k = self.vec2k(k_vec)
        
        return k

    def k2xx(self, k):
        """
        Transform regular k-space (Kd array) to scaled image xx (Nd array)
        """
        dd = numpy.size(self.st['Kd'])
        
        xx = numpy.fft.ifftn(k, self.st['Kd'], range(0, dd))
        
        xx = xx[crop_slice_ind(self.st['Nd'])]
        return xx

    def xx2x(self, xx):
        """
        Rescaled image xx (Nd array) to x (Nd array)
        
        Thus, rescaling improves the accuracy of k-space interpolation.
        
        """
        x = self.x2xx(xx)
        return x
    def k2k(self, k):
        """
        gridding and regridding of k-spectrum 
        input 
            k: Kd array, 
        output 
            k: Kd array
        """
        Xk = numpy.reshape(k, (numpy.prod(self.st['Kd']), ), order='F')
#         y = numpy.reshape(self.st['p'].dot(Xk), (self.st['M'], ), order='F')        
        k = self.st['pHp'].dot(Xk)
        k = numpy.reshape(k, self.st['Kd'], order='F')
        return k

def test_installation():
    '''
    Test the installation
    '''
    import pkg_resources
    PYNUFFT_PATH = pkg_resources.resource_filename('pynufft', './')
    DATA_PATH = pkg_resources.resource_filename('pynufft', 'data/')
    import os.path
    
    
    print('Does pynufft.py exist? ',os.path.isfile(PYNUFFT_PATH+'pynufft.py'))
    print('Does om1D.npz exist?',os.path.isfile(DATA_PATH+'om1D.npz'))
    print('Does om2D.npz exist?',os.path.isfile(DATA_PATH+'om2D.npz'))
    print('Does om3D.npz exist?',os.path.isfile(DATA_PATH+'om3D.npz'))
    print('Does phantom_3D_128_128_128.npz exist?', os.path.isfile(DATA_PATH+'phantom_3D_128_128_128.npz'))
    print('Does phantom_256_256.npz exist?', os.path.isfile(DATA_PATH+'phantom_256_256.npz'))
    print('Does 1D_example.py exist?', os.path.isfile(PYNUFFT_PATH+'example/1D_example.py'))
    print('Does 2D_example.py exist?', os.path.isfile(PYNUFFT_PATH+'example/1D_example.py'))
    

def test_2D():
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

    
    
    Nd = (256, 256)  # image size
    print('setting image dimension Nd...', Nd)
    Kd = (512, 512)  # k-space size
    print('setting spectrum dimension Kd...', Kd)
    Jd = (6, 6)  # interpolation size
    print('setting interpolation size Jd...', Jd)
    # load k-space points
    # om = numpy.loadtxt(DATA_PATH+'om.txt')
    om = numpy.load(DATA_PATH+'om2D.npz')['arr_0']
    print('setting non-uniform coordinates...')
    matplotlib.pyplot.plot(om[::10,0],om[::10,1],'o')
    matplotlib.pyplot.title('non-uniform coordinates')
    matplotlib.pyplot.xlabel('axis 0')
    matplotlib.pyplot.ylabel('axis 1')
    matplotlib.pyplot.show()

    NufftObj = NUFFT()
    NufftObj.plan(om, Nd, Kd, Jd)
    
    y = NufftObj.forward(image)
    print('setting non-uniform data')
    print('y is an (M,) list',type(y), y.shape)
    
    kspectrum = NufftObj.y2k_DC(y)
    shifted_kspectrum = numpy.fft.fftshift(kspectrum, axes=(0,1))
    print('getting the k-space spectrum, shape =',kspectrum.shape)
    print('Showing the shifted k-space spectrum')
    
    matplotlib.pyplot.imshow( shifted_kspectrum.real, cmap = matplotlib.cm.gray, norm=matplotlib.colors.Normalize(vmin=-100, vmax=100))
    matplotlib.pyplot.title('shifted k-space spectrum')
    matplotlib.pyplot.show()
    
    image2 = NufftObj.adjoint(y * NufftObj.st['W'])


    matplotlib.pyplot.imshow(image2.real, cmap=matplotlib.cm.gray, norm=matplotlib.colors.Normalize(vmin=0.0, vmax=1))
    matplotlib.pyplot.show()

    image2 = NufftObj.inverse_cg(y)
    matplotlib.pyplot.imshow(image2.real, cmap=matplotlib.cm.gray, norm=matplotlib.colors.Normalize(vmin=0.0, vmax=1))
    matplotlib.pyplot.show()


if __name__ == '__main__':
    """
    Test the module pynufft
    """
    test_2D()
#     test_installation()