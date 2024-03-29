from distutils.core import setup, Extension
import numpy

pykmeans = Extension(
    'pykmeans',
    sources=['pykmeans.c'],
    library_dirs=['/mnt/software/intel/composer_xe_2013.4.183/mkl/lib/intel64'],
    libraries=['mkl_rt', 'mkl_intel_ilp64', 'mkl_gnu_thread', 'mkl_core', 'dl', 'pthread', 'm', 'gomp'],
    extra_compile_args=['-fopenmp', '-g', '-DMKL_ILP64', '-m64', '-O3'],
    include_dirs=[numpy.get_include(), '.', '/mnt/software/intel/composer_xe_2013.4.183/mkl/include'],
)

setup(
    name='pykmeans',
    version='0.1',
    description='A fast and simple main-memory kmeans implementation using OpenMP and blas.',
    ext_modules=[pykmeans],
)
