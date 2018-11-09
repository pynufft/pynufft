special_license='''
The license of the 3D Shepp-Logan phantom:
Copyright (c) 2006, Matthias Schabel 
All rights reserved.

Redistribution and use in source and binary forms, with or without 
modification, are permitted provided that the following conditions are 
met:

* Redistributions of source code must retain the above copyright 
notice, this list of conditions and the following disclaimer. 
* Redistributions in binary form must reproduce the above copyright 
notice, this list of conditions and the following disclaimer in 
the documentation and/or other materials provided with the distribution 
* Neither the name of the University of Utah Department of Radiology nor the names 
of its contributors may be used to endorse or promote products derived 
from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE 
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
POSSIBILITY OF SUCH DAMAGE.
'''
import numpy 
import matplotlib.pyplot as pyplot
from matplotlib import cm
gray = cm.gray
def indxmap_diff(Nd):
    """
    Preindixing for rapid image gradient ()
    
    Diff(x) = x.flat[d_indx[0]] - x.flat
    
    Diff_t(x) =  x.flat[dt_indx[0]] - x.flat
                            
    :param Nd: the dimension of the image
    :type Nd: tuple with integers
    :returns d_indx: iamge gradient
    :returns  dt_indx:  the transpose of the image gradient 
    :rtype: d_indx: lists with numpy ndarray
    :rtype: dt_indx: lists with numpy ndarray
    """    

    ndims = len(Nd)
    Ndprod = numpy.prod(Nd)
    mylist = numpy.arange(0, Ndprod).astype(numpy.int32)
    mylist = numpy.reshape(mylist, Nd)
    d_indx = []
    dt_indx = []
    for pp in range(0, ndims):
        d_indx = d_indx + [ numpy.reshape(   numpy.roll(  mylist, +1 , pp  ), (Ndprod,)  ,order='C').astype(numpy.int32) ,]
        dt_indx = dt_indx + [ numpy.reshape(   numpy.roll(  mylist, -1 , pp  ) , (Ndprod,) ,order='C').astype(numpy.int32) ,]

    return d_indx,  dt_indx  
import scipy.sparse
def gradient_class(Nd, axis):
    d_indx, dt_indx = indxmap_diff(Nd)
    I = scipy.sparse.eye(numpy.prod(Nd), numpy.prod(Nd))
    data = numpy.ones((numpy.prod(Nd),))
    
    row_ind = d_indx[axis]
    col_ind= numpy.arange(0, numpy.prod(Nd)).astype(numpy.int) 
    G = scipy.sparse.csr_matrix(( data, 
                                            (row_ind, col_ind)), shape = (numpy.prod(Nd),numpy.prod(Nd))
                                 )
    G = G- I

    G = G.tocsr()
    return G
def gradient_class2(Nd, axis):
    d_indx, dt_indx = indxmap_diff(Nd)
    I = scipy.sparse.eye(numpy.prod(Nd), numpy.prod(Nd))
    data = numpy.ones((numpy.prod(Nd),))
    
    row_ind = dt_indx[axis]
    col_ind= numpy.arange(0, numpy.prod(Nd)).astype(numpy.int) 
    G = scipy.sparse.csr_matrix(( data, 
                                            (row_ind, col_ind)), shape = (numpy.prod(Nd),numpy.prod(Nd))
                                 )
    G = G- I

    G = G.tocsr()
    return G
def GBPDNA_old(nufft, gy, maxiter):
    """
    GBPDNA: test 3D total variation
    """
    import pynufft.src._helper.helper as helper
    f = gy.get()
    def A(x):
        px = numpy.array(x.astype(numpy.complex64), order='C')
        y2 = nufft.forward(nufft.thr.to_device(numpy.reshape(px, nufft.st['Nd']))).get()
        return y2
    
    def AH(y):
        py = numpy.array(y.astype(numpy.complex64), order='C')
        x2 = nufft.adjoint(nufft.thr.to_device(py)).get().flatten()
        return x2    
    
    
    Nd = nufft.st['Nd']
    Gx = gradient_class(Nd, 0)
    Gy = gradient_class(Nd, 1)
    Gz = gradient_class(Nd, 2)
    Gx2 = gradient_class2(Nd, 0)
    Gy2 = gradient_class2(Nd, 1)
    Gz2 = gradient_class2(Nd, 2)
    
    M = nufft.st['M']
    v = numpy.ones(M,)

    for pp in range(0,20):
        w = A(AH((v)))
        lab = numpy.inner(w,numpy.conj(v))/numpy.inner(v,numpy.conj(v))
        tau_1 = 1/lab.real
    #     print(lab, tau_1)
        w = w/numpy.linalg.norm(w)
        v= w
    v= numpy.random.rand(numpy.prod(Nd),)

    for pp in range(0,20):
        w = Gx.getH().dot(Gx.dot(v))
        lab = numpy.inner(w,numpy.conj(v))/numpy.inner(v,numpy.conj(v))
        tau_2 = 1/(lab.real)
    
        w = w/numpy.linalg.norm(w)
        v= w
        
    print("tau_1 = ", tau_1)   
    print("tau_2 = ", tau_2)
#     tau_1 = 0.1*tau_1
    tau_2 = 0.01*tau_2
#     tau_2 *= 3
    delta = 1.0
    mu = 0.001*numpy.max(numpy.abs(AH(f))[...])
    print("mu=",mu)
    
    def P_lambda(w_i, mu, tau_1):
        w_abs = numpy.abs(w_i)
    #     print(w_abs.shape)
    #     print(w_iw_abs.shape)
        out = ((w_i+1e-10)/(w_abs+1e-10))*mu/tau_1
        
        indx= w_abs <= (mu/tau_1)
        out[indx] =w_i[indx]
        return out
    def Q_f_eps(v, f, eps):
        v_f = v-f
        v_f_abs = numpy.abs(v_f)
        out = f + eps* v_f/v_f_abs
        indx = (v_f_abs <= eps)
        out[indx] = v[indx]
        return out
    N = numpy.prod(Nd)
    u_bold_k = numpy.zeros(N,)
    v_k = numpy.zeros(M,)
    z_k = numpy.zeros(M,)
    w_kx = numpy.zeros(N,)
    w_ky = numpy.zeros(N,)
    w_kz = numpy.zeros(N,)
#     w_kx2 = numpy.zeros(N,)
#     w_ky2 = numpy.zeros(N,)
#     w_kz2 = numpy.zeros(N,)
    
    hx = numpy.zeros(N,)
    hy = numpy.zeros(N,)
    hz = numpy.zeros(N,)
    hx2 = numpy.zeros(N,)
    hy2 = numpy.zeros(N,)
    hz2 = numpy.zeros(N,)
    
    tmp_f=numpy.zeros(M,)
    
    eps = 1e-16
    for iter in range(0, maxiter):
        
        print(iter)
        tmp_u= u_bold_k - tau_1 * AH(v_k + tmp_f- z_k).flat[...]
        
        u_bar_kp1 = tmp_u   -    tau_1 *( Gx.getH().dot(w_kx) + Gy.getH().dot(w_ky) + Gz.getH().dot(w_kz) )
#                                           Gx2.getH().dot(w_kx2) + Gy2.getH().dot(w_ky2) + Gz2.getH().dot(w_kz2) ) 
        
    #     sx = Gx.dot(u_bar_kp1)
    #     sy = Gy.dot(u_bar_kp1)
    #     s = (sx**2 + sy**2)**0.5
          
        w_kp1x = P_lambda(w_kx + (tau_2/tau_1)*Gx.dot(u_bar_kp1), mu, tau_1)
        w_kp1y = P_lambda(w_ky+ (tau_2/tau_1)*Gy.dot(u_bar_kp1), mu, tau_1)
        w_kp1z = P_lambda(w_kz+ (tau_2/tau_1)*Gz.dot(u_bar_kp1), mu, tau_1)
#         w_kp1x2 = P_lambda(w_kx2+ (tau_2/tau_1)*Gx2.dot(u_bar_kp1), mu, tau_1)
#         w_kp1y2 = P_lambda(w_ky2+ (tau_2/tau_1)*Gy2.dot(u_bar_kp1), mu, tau_1)
#         w_kp1z2 = P_lambda(w_kz2+ (tau_2/tau_1)*Gz2.dot(u_bar_kp1), mu, tau_1)
        
    #     hx = (sx+eps)/(s+eps)*Gx.getH().dot(w_kp1)
    #     hy = (sy+eps)/(s+eps)*Gy.getH().dot(w_kp1)
        
        u_bold_kp1 = tmp_u   -   tau_1 *( Gx.getH().dot(w_kp1x) + Gy.getH().dot(w_kp1y) + Gz.getH().dot(w_kp1z))
#                                           Gx2.getH().dot(w_kp1x2) + Gy2.getH().dot(w_kp1y2) + Gz2.getH().dot(w_kp1z2)) 
        
        tmp_f=A(numpy.reshape( u_bold_kp1, Nd))
        z_kp1 = Q_f_eps(tmp_f + v_k, f, eps)
        v_kp1 = v_k + delta * (tmp_f    -   z_kp1)
        w_kx = w_kp1x
        w_ky = w_kp1y
        w_kz = w_kp1z
#         w_kx2 = w_kp1x2
#         w_ky2 = w_kp1y2
#         w_kz2 = w_kp1z2
        
        u_bold_k = u_bold_kp1
        v_k = v_kp1
        z_k = z_kp1
    return numpy.reshape(u_bar_kp1, Nd)

# def GBPDNA(nufft, gy, maxiter):
#     """
#     GBPDNA: test 3D total variation
#     """
#     Nd = nufft.st['Nd']
#     ndims = len(Nd)
# #     Gx = gradient_class(Nd, 0)
# #     Gy = gradient_class(Nd, 1)
# #     Gz = gradient_class(Nd, 2)
#     M = nufft.st['M']    
#     import pynufft.src._helper.helper as helper
#     d_indx, dt_indx = helper.indxmap_diff(Nd)
#     for jj in range(    0,  ndims): # n_dims + 1 for wavelets
#         d_indx[jj] = nufft.thr.to_device(d_indx[jj])
#         dt_indx[jj] = nufft.thr.to_device(dt_indx[jj])    
#     
#     Nd_cpu = numpy.zeros(Nd, dtype = numpy.complex64, order='C')
#     Nd_gpu = nufft.thr.to_device(Nd_cpu) # zero array on GPU
#     M_cpu = numpy.zeros((M,),   dtype = numpy.complex64,   order='C')
#     M_gpu = nufft.thr.to_device(M_cpu) # zero array on GPU
#     
#     print('I am here')
#     gf = gy
#     def A(input_x):
#         output_y = nufft.forward(input_x) 
#         return output_y
#     
#     def AH(input_y):
#         output_x = nufft.adjoint(input_y)
#         return output_x    
#     
#     def GH(input_x, pp):
#         output_x = nufft.thr.copy_array(Nd_gpu) 
#         nufft.cDiff(dt_indx[pp], input_x, output_x, local_size=None, global_size=int(nufft.Ndprod))
#         return output_x
#     
#     def G(input_x, pp):
#         output_x = nufft.thr.copy_array(Nd_gpu) 
#         nufft.cDiff(d_indx[pp], input_x, output_x, local_size=None, global_size=int(nufft.Ndprod))
#         return output_x
# 
#     def get_tau1():
# 
#         v_cpu = numpy.ones(M, dtype = numpy.complex64, order='C')
#         v_gpu = nufft.thr.to_device(v_cpu)
#     
#         for pp in range(0,10):
#             
#             w = A(AH((v_gpu)))
#             v_new = w.get()
#             v_old = v_gpu.get()
#             lab = numpy.inner(v_new,numpy.conj(v_old))/numpy.inner(v_old,numpy.conj(v_old))
#             tau_1 = 0.999/(lab.real)**0.5
#             print(pp,'tau_1', tau_1)
#         #     print(lab, tau_1)
#     #         w = w/numpy.linalg.norm(w)
#             v_new = v_new/numpy.linalg.norm(v_new)
#             v_gpu = nufft.thr.to_device(v_new)
#         return tau_1
#     def get_tau2():
#         v = numpy.random.randn(numpy.prod(Nd))
#         v = v.reshape(Nd, order='C').astype(numpy.complex64)
# #         v= numpy.ones(Nd, dtype = numpy.complex64, order='C')
#     #     v= numpy.random.rand
#         v_gpu = nufft.thr.to_device(v)
#         for pp in range(0,10):
#     #         w = Gx.getH().dot(Gx.dot(v))
#             w = GH(G(v_gpu, 0), 0)
#             v_old = v_gpu.get().ravel()
#             v_new = w.get().ravel()
#             lab = numpy.inner(v_new,numpy.conj(v_old))/numpy.inner(v_old,numpy.conj(v_old))
#             tau_2 = 0.999/(lab.real)**0.5
#             print(pp,'tau_2', tau_2)
#     #         w = w/numpy.linalg.norm(w)
#             v_new = v_new/numpy.linalg.norm(v_new)
#             v_gpu = nufft.thr.to_device(v_new)
#         return tau_2
#     tau_1 = get_tau1()
#     tau_2 = get_tau2()
# 
#     tau_2 /= 3**0.5
#     R21 = tau_2 / tau_1
#     print("tau_1 = ", tau_1)   
#     print("tau_2 = ", tau_2)
#     print("R21 = ", R21)
#     delta = 1.0
#     mu = 0.001*numpy.max(numpy.abs(AH(gf).get())[...])
#     print("mu=",mu)
#     print('End here')
#     def P_lambda(w_i, mu, tau_1):
#         w_i = numpy.nan_to_num(w_i.astype(numpy.complex128))
#         w_abs = numpy.abs(w_i)
#     #     print(w_abs.shape)
#     #     print(w_iw_abs.shape)
#         out = ((w_i+1e-7)/(w_abs+1e-7))
#         out *= mu/tau_1
#         
#         indx= w_abs <= (mu/tau_1)
#         out[indx] =w_i[indx]
#         return out
#     def Q_f_eps(v, f, eps):
#         v = numpy.nan_to_num(v.astype(numpy.complex128))
#         v_f = v-f
#         v_f_abs = numpy.abs(v_f)
#         out = f + eps* ( v_f + 1e-7)/(v_f_abs+1e-7)
#         indx = (v_f_abs <= eps)
#         out[indx] = v[indx]
#         return out
#     
# #     N = numpy.prod(Nd)
#     
# #     u_bold_k = numpy.zeros(N,)
# #     v_k = numpy.zeros(M,)
# #     z_k = numpy.zeros(M,)
# #     w_kx = numpy.zeros(N,)
# #     w_ky = numpy.zeros(N,)
# #     w_kz = numpy.zeros(N,)
# #     
# #     hx = numpy.zeros(N,)
# #     hy = numpy.zeros(N,)
# #     hz = numpy.zeros(N,)
# #     
# #     tmp_f=numpy.zeros(M,)
#     u_bold_k = nufft.thr.copy_array(Nd_gpu)
#     v_k = nufft.thr.copy_array(M_gpu)
#     z_k = nufft.thr.copy_array(M_gpu)
#     w_kx = nufft.thr.copy_array(Nd_gpu)
#     w_ky = nufft.thr.copy_array(Nd_gpu)
#     w_kz = nufft.thr.copy_array(Nd_gpu)
#     
#     hx = nufft.thr.copy_array(Nd_gpu)
#     hy = nufft.thr.copy_array(Nd_gpu)
#     hz = nufft.thr.copy_array(Nd_gpu)
#     
#     tmp_f = nufft.thr.copy_array(M_gpu)    
#     eps = 1e-16
#     for iter in range(0, maxiter):
#         
#         print(iter, 'R21', R21)
#         tmp_u= u_bold_k - tau_1 * AH(v_k +  tau_1 *tmp_f - z_k)
#         
#         u_bar_kp1 = tmp_u.get()   -    tau_1 *( GH(w_kx, 0) + GH(w_ky, 1) + GH(w_kz, 2)).get()
#         u_bar_kp1 = nufft.thr.to_device(u_bar_kp1.astype(numpy.complex64))
#     #     sx = Gx.dot(u_bar_kp1)
#     #     sy = Gy.dot(u_bar_kp1)
#     #     s = (sx**2 + sy**2)**0.5
#         
#         w_kp1x = P_lambda(w_kx.get() + R21*G(u_bar_kp1, 0).get(), mu, tau_1).astype(numpy.complex64)
#         w_kp1y = P_lambda(w_ky.get() + R21*G(u_bar_kp1, 1).get(), mu, tau_1).astype(numpy.complex64)
#         w_kp1z = P_lambda(w_kz.get() + R21*G(u_bar_kp1, 2).get(), mu, tau_1).astype(numpy.complex64)
#         
#         w_kp1x = nufft.thr.to_device(w_kp1x)
#         w_kp1y = nufft.thr.to_device(w_kp1y)
#         w_kp1z = nufft.thr.to_device(w_kp1z)
#     #     hx = (sx+eps)/(s+eps)*Gx.getH().dot(w_kp1)
#     #     hy = (sy+eps)/(s+eps)*Gy.getH().dot(w_kp1)
#         
#         u_bold_kp1 = tmp_u.get()   -   tau_1 *( GH(w_kp1x, 0) +  GH(w_kp1y, 1) +  GH(w_kp1z, 2)).get() 
# #         u_bold_kp1 = ( GH(w_kp1x, 0) +  GH(w_kp1y, 1) +  GH(w_kp1z, 2)).get()
# #         u_bold_kp1 = tmp_u  - tau_1 *u_bold_kp1
#         u_bold_kp1 = nufft.thr.to_device(u_bold_kp1.astype(numpy.complex64))
#         print(u_bold_kp1.get().shape,u_bold_kp1.get().dtype)
#         
#         tmp_f=A( u_bold_kp1)
#         z_kp1 = Q_f_eps( (tmp_f.get() + v_k.get()), gf.get(), eps).astype(numpy.complex64)
# #         z_kp1 = nufft.thr.to_device(z_kp1)
#         v_kp1 = v_k.get() + delta * (tmp_f.get()    -   z_kp1)
#         v_kp1 = nufft.thr.to_device(v_kp1.astype(numpy.complex64))
#         w_kx = w_kp1x
#         w_ky = w_kp1y
#         w_kz = w_kp1z
#         u_bold_k = u_bold_kp1
#         v_k = v_kp1
#         z_k = nufft.thr.to_device(z_kp1.astype(numpy.complex))
#     return u_bar_kp1.get()
        
        
import pkg_resources
DATA_PATH = pkg_resources.resource_filename('pynufft', './src/data/')   
image = numpy.load(DATA_PATH +'phantom_3D_128_128_128.npz')['arr_0']#[0::2, 0::2, 0::2]
image = numpy.array(image, order='C')

image = numpy.load('/home/sram/UCL/DATA/G/2 McwBra DICOM/CScontNoECG_DICOM/3D_volume.npz')['arr_0']
image = image/numpy.max(abs(image.ravel()))
image = image[32:32+128, 32:32+128,12:12+64]
image = numpy.abs(image)
# print(special_license)

# pyplot.imshow(numpy.abs(image[:,:,64]), label='original signal',cmap=gray)
# pyplot.show()

Nd = (128,128,64) # time grid, tuple
Kd = (256,256,128) # frequency grid, tuple
Jd = (6,6,6) # interpolator 
mid_slice = int(Nd[2]/2)
#     om=       numpy.load(DATA_PATH+'om3D.npz')['arr_0']
numpy.random.seed(0)
om = numpy.random.randn(int(5e+6),3)*0.7
print(om.shape)
from pynufft import NUFFT_cpu, NUFFT_memsave, NUFFT_hsa
NufftObj = NUFFT_memsave()

NufftObj.plan(om, Nd, Kd, Jd)

NufftObj.offload(API = 'ocl',   platform_number = 1, device_number = 0)
# NufftObj.offload(API = 'cuda',   platform_number = 0, device_number = 0)
gx = NufftObj.thr.to_device(image.astype(numpy.complex64))
gy =NufftObj.forward(gx) 
import time
t0 = time.time()
restore_x2 = GBPDNA_old(NufftObj, gy, maxiter=200)
t1 = time.time()
# restore_x = NufftObj.solve(gy,'cg', maxiter=50)
t2 = time.time()
print("GBPDNA time = ", t1 - t0)
print("CG time = ", t2 - t1)

#restore_image1 = NufftObj.solve(kspace,'L1TVLAD', maxiter=300,rho=0.1)
# 
# restore_x2 = NufftObj.solve(gy,'L1TVOLS', maxiter=100,rho=0.2)
# tau_1 = 1
# tau_2 = 0.1


pyplot.subplot(1,2,1)
pyplot.imshow(numpy.real(gx.get()[:,:,mid_slice]), label='original signal',cmap=gray)
pyplot.title('original')    
#pyplot.subplot(2,2,2)
#pyplot.imshow(numpy.real(restore_image1[:,:,32]), label='L1TVLAD',cmap=gray)
#pyplot.title('L1TVLAD')

pyplot.subplot(1,2,2)
pyplot.imshow(numpy.abs(restore_x2[:,:,mid_slice]), label='L1TVOLS',cmap=gray)
pyplot.title('GBPDNA (500 iterations)')
    


# pyplot.subplot(2,2,4)
# pyplot.imshow(numpy.real(restore_x.get()[:,:,mid_slice]), label='CG',cmap=gray)
# pyplot.title('CG')?
#     pyplot.legend([im1, im im4])


pyplot.show()


