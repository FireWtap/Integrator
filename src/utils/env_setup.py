import os
import sys
import warnings

# Check if numpy is already loaded
if 'numpy' in sys.modules:
    warnings.warn(
        "WARNING: 'numpy' is already imported! "
        "Setting OPENBLAS_NUM_THREADS now may have NO EFFECT. "
        "Ensure 'src.utils.env_setup' is imported BEFORE any other package (scanpy, numpy, pandas, etc.).",
        RuntimeWarning
    )

# Set environment variables to limit threads before any other imports
# This is critical to prevent segmentation faults with OpenBLAS/Harmony
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_CORETYPE"] = "ARMV8"

print("--- Environment Setup ---")
for var in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    print(f"{var}: {os.environ.get(var)}")
print("-------------------------")
