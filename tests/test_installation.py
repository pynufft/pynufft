
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
    
    
#     print('Does nufft.py exist? ',os.path.isfile(PYNUFFT_PATH+'nufft.py'))
    print('The om1D.npz exists.',os.path.isfile(DATA_PATH+'om1D.npz'))
    print('The om2D.npz exists.',os.path.isfile(DATA_PATH+'om2D.npz'))
    print('The om3D.npz exist.',os.path.isfile(DATA_PATH+'om3D.npz'))
    print('The phantom_3D_128_128_128.npz exist.', os.path.isfile(DATA_PATH+'phantom_3D_128_128_128.npz'))
    print('The phantom_256_256.npz exists.', os.path.isfile(DATA_PATH+'phantom_256_256.npz'))
    print('The example_1D.py exists.', os.path.isfile(PYNUFFT_PATH+'./example/script_1D.py'))
    print('The example_2D.py exist.', os.path.isfile(PYNUFFT_PATH+'./example/script_2D.py'))
    
    
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
