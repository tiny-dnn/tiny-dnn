
VERSION="0.0.1"
APPNAME='tiny-cnn'

srcdir = '.'
blddir = 'build'

def options(opt):
    opt.load('compiler_cxx')

def configure(conf):
    conf.env.CXX = 'g++'
    conf.load('g++')

def build(bld):
    bld.recurse('src')

