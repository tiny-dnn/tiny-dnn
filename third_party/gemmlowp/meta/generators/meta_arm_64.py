"""Generates the arm32 headers used by the gemm/gemv lib."""

import cc_emitter
import meta_arm_common
import neon_emitter_64


def Main():
  """Generate the single threaded meta gemm library."""
  cc = cc_emitter.CCEmitter()
  meta_arm_common.GenerateHeader(cc, 'gemmlowp_meta_single_thread_gemm_arm64',
                                 'GEMMLOWP_NEON_64')

  cc.EmitNamespaceBegin('gemmlowp')
  cc.EmitNamespaceBegin('meta')
  cc.EmitNamespaceBegin('internal')
  cc.EmitNewline()

  meta_arm_common.GenerateInternalFunctions(cc, neon_emitter_64.NeonEmitter64())

  cc.EmitNamespaceEnd()
  cc.EmitNamespaceEnd()
  cc.EmitNamespaceEnd()
  cc.EmitNewline()

  meta_arm_common.GenerateFooter(
      cc, 'Meta gemm for arm64 requires: GEMMLOWP_NEON_64!')


if __name__ == '__main__':
  Main()
