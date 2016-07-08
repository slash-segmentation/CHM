#!/usr/bin/env python

from __future__ import division
#from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import

try: from setuptools import setup
except ImportError: from distutils.core import setup
from distutils.extension import Extension
from distutils.ccompiler import get_default_compiler
from Cython.Build import cythonize
from os.path import abspath, join, sep
import numpy
import pysegtools.general.cython
import chm

compiler_name = get_default_compiler() # TODO: this isn't the compiler that will necessarily be used, but is a good guess...
compiler_opt = {
        'msvc'    : ['/D_SCL_SECURE_NO_WARNINGS','/EHsc','/O2','/DNPY_NO_DEPRECATED_API=7','/bigobj','/openmp'],
        'unix'    : ['-std=c++11','-O3','-march=native','-DNPY_NO_DEPRECATED_API=7','-fopenmp'], # gcc/clang (whatever is system default)
        'mingw32' : ['-std=c++11','-O3','-march=native','-DNPY_NO_DEPRECATED_API=7','-fopenmp'],
        'cygwin'  : ['-std=c++11','-O3','-march=native','-DNPY_NO_DEPRECATED_API=7','-fopenmp'],
    }
linker_opt = {
        'msvc'    : [],
        'unix'    : ['-fopenmp'], # gcc/clang (whatever is system default)
        'mingw32' : ['-fopenmp'],
        'cygwin'  : ['-fopenmp'],
    }

cy_inc = pysegtools.general.cython.get_include()
np_inc = numpy.get_include()
np_rand = abspath(join(np_inc, '..', '..', 'random'))

def create_ext(name, dep=[], src=[], inc=[], lib=[]):
    return Extension(
        name=name,
        depends=dep,
        sources=[name.replace('.',sep)+'.pyx']+src,
        define_macros=[('NPY_NO_DEPRECATED_API','7'),],
        include_dirs=[np_inc,cy_inc]+inc,
        library_dirs=lib,
        extra_compile_args=compiler_opt.get(compiler_name, []),
        extra_link_args=linker_opt.get(compiler_name, []),
        language='c++',
    )

if __name__ == '__main__':
    setup(name='chm',
          version='%s'%chm.__version__,
          description='Cascaded Hierarchical Model - an image segmentation framework',
          author='Jeffrey Bush',
          author_email='jeff@coderforlife.com',
          url='https://www.sci.utah.edu/software/chm.html',
          packages=['chm'],
          install_requires=['numpy>=1.7','scipy>=0.16','cython>=0.22','pysegtools[MATLAB]>=0.1'],
          extras_require={ 'OPT': ['pyfftw>=0.10'], },
          use_2to3=True, # the code *should* support Python 3 once run through 2to3 but this isn't tested
          zip_safe=False, # I don't think this code would work when running from inside a zip file due to the dynamic-load and dynamic-cython systems
          ext_modules = cythonize([
              create_ext('chm._utils'),
              create_ext('chm._train', inc=[np_rand], lib=[np_rand]), #libraries=[':mtrand.so']
              create_ext('chm.__imresize'),
              create_ext('chm.filters._correlate'),
              create_ext('chm.filters._haar'),
              create_ext('chm.filters._hog', dep=[join('chm','filters','HOG.h')], src=[join('chm','filters','HOG.cpp')]),
              create_ext('chm.filters._sift'),
              create_ext('chm.filters._frangi'),
              create_ext('chm.filters._intensity'),
          ], include_path=[cy_inc]))
