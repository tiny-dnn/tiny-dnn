
VERSION="0.0.1"
APPNAME='tiny-cnn'

srcdir = '.'
blddir = 'build'

def options(opt):
    opt.load('compiler_cxx')
    opt.add_option('--TBB',
                   action = 'store_true',
                   default = False,
                   help='enable TBB parallelization [default:False]')
    opt.add_option('--AVX',
                   action = 'store_true',
                   default = False,
                   help='enable AVX vectorization [default:False]')
    opt.add_option('--SSE',
                   action = 'store_true',
                   default = False,
                   help='enable AVX vectorization [default:False]')
    opt.add_option('--BOOST_ROOT',
                   action = 'store',
                   default = 'C:/Program Files/boost/boost_1_51_0',
                   help='root directory of boost [default:\'C:/Program Files/boost/boost_1_51_0\']')
    opt.add_option('--TBB_ROOT',
                   action = 'store',
                   default = 'C:/Program Files/tbb41_20121003oss',
                   help='root directory of TBB [default:\'C:/Program Files/tbb41_20121003oss\']')

def configure(conf):
    conf.env.CXX = 'g++'
    conf.load('g++')
    if conf.options.TBB:
       conf.define("CNN_USE_TBB", "")
    if conf.options.SSE:
       conf.define("CNN_USE_AVX", "")
    if conf.options.SSE:
       conf.define("CNN_USE_SSE", "")
    conf.env.TBB_ROOT = conf.options.TBB_ROOT
    conf.env.BOOST_ROOT = conf.options.BOOST_ROOT

def build(bld):
    bld.recurse('src')

