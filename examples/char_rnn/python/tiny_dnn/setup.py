#!/usr/bin/env python

"""
setup.py file for SWIG example
"""
import os
from distutils.core import setup, Extension

tiny_char_rnn = Extension('_tiny_char_rnn',
                          include_dirs=[os.path.abspath('../../../../')],
                          extra_compile_args=['-std=c++14', '-O3', '-DNDEBUG', '-O3', '-msse3', '-mavx', '-Wall',
                                              '-Wpedantic', '-Wno-narrowing', '-Wno-deprecated', '-msse3', '-mavx',
                                              '-Wall', '-Wpedantic', '-Wno-narrowing', '-Wno-deprecated'],
                          sources=['tiny_char_rnn_wrap.cxx'],
                          )

setup(name='tiny_char_rnn',
      version='0.1',
      author="prlz77",
      description="""Simple swig example from docs""",
      ext_modules=[tiny_char_rnn],
      py_modules=["tiny_char_rnn"],
      )
