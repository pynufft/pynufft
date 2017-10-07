import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../pynufft')))

import pynufft_hsa
pynufft_hsa.test_init()