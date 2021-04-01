import numpy
import sys
import os
from distutils.core import setup, Extension
from distutils.sysconfig import get_python_inc
from distutils.extension import Extension
#from Cython.Build import cythonize      # cythonize compiles a pyx
from Cython.Distutils import build_ext   # Extension for a c-file, build_ext for cython file

#modules = ["kfARlibMPmv_ram2", "kfARlib1c", "kdist", "ARcfSmplNoMCMC_ram"]
modules = ["kfARlibMPmv_ram2", "kfARlibMPmv_ram3", "kfARlibMPmv_ram5", "kdist", "ARcfSmplNoMCMC_ram", "inv_cov"]
#modules = ["inv_cov"]

###  import LogitWrapper 
###  LogitWrapper
#  -undefined dynamic_lookup
#  -lgsl
#  -fpic   #  build a shared
#  -bundle

#  OMP_THREAD_NUM

#  use --user  to install in
#  to specify compiler, maybe set CC environment variable
#  or python setup.py build --compiler=g++
incdir = [get_python_inc(plat_specific=1), numpy.get_include(), "pyPG/include/RNG"]
libdir = ['/usr/local/lib']

###  ARE WE RUNNING ON COLAB?
try:
  import google.colab
  RUN_ON_COLAB = True
except:
  RUN_ON_COLAB = False

if RUN_ON_COLAB:
    os.environ["CC"]  = "/usr/bin/gcc"   
else:
    os.environ["CC"]  = "/usr/local/bin/gcc-6"

##  Handle OPENMP switch here

#  http://stackoverflow.com/questions/677577/distutils-how-to-pass-a-user-defined-parameter-to-setup-py
USE_OPENMP = False
#  -fPIC meaningless in osx
#extra_compile_args = ["-fPIC", "-bundle", "-undefined dynamic_lookup", "-shared"]
#extra_compile_args = ["-undefined dynamic_lookup", "-shared"]
extra_compile_args = []
#extra_link_args    = ["-lblas", "-llapack", "-lgsl"]
#  didn't need -llapack on Ubuntu
#extra_link_args    = ["-lblas", "-lgsl", "-fopenmp"]
#extra_link_args    = ["-fopenmp"]

#if "--use_openmp" in sys.argv:
USE_OPENMP = True
extra_compile_args.extend(["-fopenmp", "-DUSE_OPEN_MP"])
#extra_link_args.append("-fopenmp")
#iop = sys.argv.index("--use_openmp")
#sys.argv.pop(iop)

#  may also need to set $LD_LIBRARY_PATH in order to use shared libgsl

cmdclass = {'build_ext' : build_ext}
#  Output to be named _LogitWrapper.so

for module in modules:
    ext_modules = Extension(module,
                            ["%s.pyx" % module],
                            #libraries = ['gsl', 'gslcblas'],
                            include_dirs=incdir,   #  include_dirs for Mac
                            extra_compile_args=extra_compile_args,
                            library_dirs=libdir)

    setup(
        name=module,
        cmdclass = cmdclass,
        #ext_modules = 
        ext_modules=[ext_modules],
    )
