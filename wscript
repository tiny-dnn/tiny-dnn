
VERSION="0.0.1"
APPNAME='tiny-cnn'

srcdir = '.'
blddir = 'build'

import sys
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
                   help='enable SSE vectorization [default:False]')

    if sys.platform.startswith('win'):
        BOOST_ROOT_DEFAULT = 'C:/Program Files/boost/boost_1_51_0/'
    else:
        BOOST_ROOT_DEFAULT = '/usr/local/include/boost/'
    opt.add_option('--BOOST_ROOT',
                   action = 'store',
                   default = BOOST_ROOT_DEFAULT,
                   help='root directory of boost [default:%s]' % BOOST_ROOT_DEFAULT)

    if sys.platform.startswith('win'):
        TBB_ROOT_DEFAULT = 'C:/Program Files/tbb/include'
    else:
        TBB_ROOT_DEFAULT = '/usr/local/include/tbb/'
    opt.add_option('--TBB_ROOT',
                   action = 'store',
                   default = TBB_ROOT_DEFAULT,
                   help='root directory of TBB [default:%s]' % TBB_ROOT_DEFAULT)

def configure(conf):
    conf.load('compiler_cxx')
    if conf.options.TBB:
       conf.define("CNN_USE_TBB", "")
       print "enable option:TBB(root=%s)" % conf.options.TBB_ROOT
    if conf.options.AVX:
       conf.define("CNN_USE_AVX", "")
       print("enable option:AVX")
    if conf.options.SSE:
       conf.define("CNN_USE_SSE", "")
       print("enable option:SSE")
    conf.env.TBB_ROOT = conf.options.TBB_ROOT
    conf.env.BOOST_ROOT = conf.options.BOOST_ROOT

def build(bld):
    bld.recurse('examples')

