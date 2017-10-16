
def test_pkg(pkgname):
    """
    Test Reikna package
    """
    try:
        __import__(pkgname)
        print(pkgname+'  has been installed.')
        return 0
    except:
        print(pkgname+' cannot be imported, check installation!')
        print('       Install '+ pkgname +' by the command \'pip install '+ pkgname+' --user \'')
        return 1
    
def test_installation():
    '''
    Test the installation
    '''
    import pkg_resources
    PYNUFFT_PATH = pkg_resources.resource_filename('pynufft', './')
    DATA_PATH = pkg_resources.resource_filename('pynufft', 'src/data/')
    import os.path
    
    
    print('Does pynufft.py exist? ',os.path.isfile(PYNUFFT_PATH+'pynufft.py'))
    print('Does om1D.npz exist?',os.path.isfile(DATA_PATH+'om1D.npz'))
    print('Does om2D.npz exist?',os.path.isfile(DATA_PATH+'om2D.npz'))
    print('Does om3D.npz exist?',os.path.isfile(DATA_PATH+'om3D.npz'))
    print('Does phantom_3D_128_128_128.npz exist?', os.path.isfile(DATA_PATH+'phantom_3D_128_128_128.npz'))
    print('Does phantom_256_256.npz exist?', os.path.isfile(DATA_PATH+'phantom_256_256.npz'))
    print('Does example_1D.py exist?', os.path.isfile(PYNUFFT_PATH+'./tests/example_1D.py'))
    print('Does example_2D.py exist?', os.path.isfile(PYNUFFT_PATH+'./tests/example_2D.py'))
    
    
    for pkgname in ('reikna', 'pyopencl', 'pycuda'):
        error_code = test_pkg(pkgname)
        if 1 == error_code:
            break
        
    
if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
# from .. import *    
    test_installation()