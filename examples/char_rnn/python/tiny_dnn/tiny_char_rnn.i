%module(package="tiny_dnn") tiny_char_rnn

%{
#define SWIG_FILE_WITH_INIT
#include "tiny_char_rnn.h"
%}

%include "tiny_char_rnn.h"
