import os
import sys
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
def run_all():
    from .test_installation import test_installation
    test_installation()
    
    from .example_1D import example_1D
    example_1D()    
    
    from .test_2D import test_2D
    test_2D()
    from .example_2D import example_2D
    example_2D()    
    from .test_init import test_init
    test_init()