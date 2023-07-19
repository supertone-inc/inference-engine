import os, sys

if sys.platform != "darwin":
    os.system("pip install tensorflow==2.12.0")

from models.matmul_dynamic import *
from models.matmul import *
