"""Generates the meta gemm/gemv library header."""

import cc_emitter
import gemm_MxNxK
import gemv_1xMxK

_HEADER_COPYRIGHT = """// Copyright 2015 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// single_thread_gemm.h: programatically generated GEMM library header.
"""


def GenerateHeader(emitter):
  """Generate the common front part of the header file."""
  emitter.EmitCodeNoSemicolon(_HEADER_COPYRIGHT)
  emitter.EmitHeaderBegin('gemmlowp_meta_single_thread_gemm')

  emitter.EmitPreprocessor1(
      'if', 'defined(GEMMLOWP_NEON_32) || defined(GEMMLOWP_NEON_64)')
  emitter.EmitNewline()

  emitter.EmitInclude('<cassert>')
  emitter.EmitNewline()

  emitter.EmitPreprocessor1('if', 'defined(GEMMLOWP_NEON_32)')
  emitter.EmitInclude('"single_thread_gemm_arm32.h"')
  emitter.EmitPreprocessor1('elif', 'defined(GEMMLOWP_NEON_64)')
  emitter.EmitInclude('"single_thread_gemm_arm64.h"')
  emitter.EmitPreprocessor('endif')
  emitter.EmitNewline()


def GenerateInternalFunctions(emitter):
  """Generate all the functions hidden in the internal namespace."""
  gemm_MxNxK.GenerateInternalFunctions(emitter)
  emitter.EmitNewline()

  gemv_1xMxK.GenerateInternalFunctions(emitter)
  emitter.EmitNewline()


def GeneratePublicFunctions(emitter):
  gemm_MxNxK.GeneratePublicFunctions(emitter)
  emitter.EmitNewline()

  gemv_1xMxK.GeneratePublicFunctions(emitter)
  emitter.EmitNewline()


def GenerateFooter(emitter):
  emitter.EmitPreprocessor('else')
  emitter.EmitPreprocessor1(
      'warning', '"Meta gemm fast-path requires GEMMLOWP_NEON_(32|64)!"')
  emitter.EmitPreprocessor('endif')
  emitter.EmitNewline()
  emitter.EmitHeaderEnd()


def Main():
  """Generate the single threaded meta gemm library."""
  emitter = cc_emitter.CCEmitter()
  GenerateHeader(emitter)

  emitter.EmitNamespaceBegin('gemmlowp')
  emitter.EmitNamespaceBegin('meta')
  emitter.EmitNamespaceBegin('internal')
  emitter.EmitNewline()

  GenerateInternalFunctions(emitter)

  emitter.EmitNamespaceEnd()
  emitter.EmitNewline()

  GeneratePublicFunctions(emitter)

  emitter.EmitNamespaceEnd()
  emitter.EmitNamespaceEnd()
  emitter.EmitNewline()

  GenerateFooter(emitter)


if __name__ == '__main__':
  Main()
