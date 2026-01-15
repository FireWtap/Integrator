import sys
import os

print("Initial modules:", "numpy" in sys.modules)

import src.utils.env_setup

print("After env_setup modules:", "numpy" in sys.modules)
print("OPENBLAS_NUM_THREADS:", os.environ.get("OPENBLAS_NUM_THREADS"))

import numpy
print("Numpy config:")
try:
    numpy.show_config()
except:
    pass
