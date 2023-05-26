import os, sys

if sys.platform == "win32":
    os.system("pip install tensorflow==2.12.0")

from models.matmul_dynamic import *
from models.matmul import *
