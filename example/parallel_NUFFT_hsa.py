"""
An example of multiprocessing with NUFFT_hsa using PyCUDA backend, 
wrapped inside the atomic_NUFFT wrapper class.  
The two processes are running on two CPU cores.
Each process creates one NUFFT_hsa and offloads the computations to GPU.
nvidia-smi confirms that two python programes are using the GPU.
"""

import numpy
from pynufft import NUFFT_hsa, NUFFT_cpu
import scipy.misc
import matplotlib.pyplot
import multiprocessing
import os 
    


class atomic_NUFFT:
    def __init__(self, om, Nd, Kd, Jd, API, device_number):
        """
        This only caches the parameters.
        Any other GPU related stuffs are carried out in run()
        """
        self.om = om
        self.Nd = Nd
        self.Kd = Kd
        self.Jd = Jd
        self.API = API
        self.device_number = device_number
        
    def run(self, x, cpu_core):
        """
        In this method, the NUFFT_hsa are created and executed on a fixed CPU core.
        """
        pid= os.getpid()
        print('pid=', pid)
        os.system("taskset -p -c %d %d" % (cpu_core, pid))
        """
        Control the CPU affinity. Otherwise the process on one core can be switched to another core.
        """

        # create NUFFT
        NUFFT = NUFFT_hsa(self.API, self.device_number,0)
        
        # plan the NUFFT
        NUFFT.plan(self.om, self.Nd, self.Kd, self.Jd)

        # send the image to device
        gx = NUFFT.to_device(x)
        
        # carry out 10000 forward transform
        for pp in range(0, 100):
            gy = NUFFT.forward(gx)

        # return the object
        return gy.get()

Nd = (256,256)
Kd = (512,512)
Jd = (6,6)
om = numpy.random.randn(35636, 2) 
x = scipy.misc.imresize(scipy.misc.ascent(), Nd)
om1 = om[om[:,0]>0, :]
om2 = om[om[:,0]<=0, :]

# create pool
pool = multiprocessing.Pool(2)

# create the list to receive the return values
results = []

# Now enter the first process
# This is the standard multiprocessing Pool
D = atomic_NUFFT(om1, Nd, Kd, Jd, 'cuda',0)
# async won't obstruct the next line of code
result = pool.apply_async(D.run, args = (x, 0))
# the result is appended
results.append(result)

# Now enter the second process
# This is the standard multiprocessing Pool
D = atomic_NUFFT(om2, Nd, Kd, Jd, 'cuda',   0)
# Non-obstructive
result = pool.apply_async(D.run, args = (x, 1))
results.append(result)

# closing the pool 
pool.close()
pool.join()

# results are appended
# Now print the outputs
result1 = results[0].get()
result2 = results[1].get()

# check CPU results

NUFFT_cpu1 = NUFFT_cpu()
NUFFT_cpu1.plan(om1, Nd, Kd, Jd)
y1 = NUFFT_cpu1.forward(x)
print('norm = ', numpy.linalg.norm(y1 - result1) / numpy.linalg.norm(y1))

NUFFT_cpu2 = NUFFT_cpu()
NUFFT_cpu2.plan(om2, Nd, Kd, Jd)
y2 = NUFFT_cpu2.forward(x)
print('norm = ', numpy.linalg.norm(y2 - result2) / numpy.linalg.norm(y2))

