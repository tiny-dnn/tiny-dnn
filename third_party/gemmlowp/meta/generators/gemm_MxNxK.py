"""Generates the specialized gemm functions."""

import mul_Nx8_Mx8_neon
import qnt_Nx8_neon
import zip_Nx8_neon

_QUANTIZED_8BIT = 'quantized_8bit'
_FULL_32BIT = 'full_32bit'
_FULL_FLOAT = 'full_float'


class Error(Exception):
  """Module level error."""


class ConfigurationError(Error):
  """Runtime configuration error."""


def GenerateCommonTempsCountersAndConsts(emitter, rows):
  """Generates all the boilerplate variables for each of the gemm functions."""
  emitter.EmitDeclare('const std::int32_t', 'row_chunks', 'm / 3')
  emitter.EmitDeclare('const std::int32_t', 'col_chunks', 'n / 3')
  emitter.EmitDeclare('const std::int32_t', 'padded_k', '((k + 7) / 8) * 8')
  emitter.EmitDeclare('const std::int32_t', 'chunk_size', 'k * 3')
  emitter.EmitDeclare('const std::int32_t', 'zipped_chunk_size',
                      '(padded_k + 16) * 3')
  emitter.EmitDeclare('const std::int32_t', 'zipped_rhs_size',
                      '(padded_k + 16) * n')
  emitter.EmitDeclare('const std::uint8_t*', 'lhs_chunk', 'lhs')
  emitter.EmitDeclare('const std::uint8_t*', 'rhs_chunk', 'rhs')
  emitter.EmitDeclare('std::uint8_t*', 'zipped_lhs', 'scratch')
  emitter.EmitDeclare(
      'std::int32_t*', 'zipped_lhs_3_offsets',
      'reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * 3)')
  if rows is not 0:
    emitter.EmitDeclare(
        'std::int32_t*', 'zipped_lhs_%d_offsets' % rows,
        'reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k * %d)' % rows)
  emitter.EmitDeclare('std::uint8_t*', 'zipped_rhs',
                      'scratch + ((zipped_chunk_size + 15) / 16) * 16')
  emitter.EmitDeclare('std::uint8_t*', 'zipped_rhs_chunk', 'zipped_rhs')
  emitter.EmitDeclare('const std::int32_t', 'result_chunk_stride',
                      'result_stride * 3')
  emitter.EmitNewline()


def GenerateQuantized8BitTempsCountersAndConsts(emitter, rows):
  """Generates all the boilerplate variables for the q8 gemm function."""
  GenerateCommonTempsCountersAndConsts(emitter, rows)
  emitter.EmitDeclare('const std::int32_t', 'const_offset',
                      'lhs_offset * rhs_offset * k + result_offset')
  emitter.EmitDeclare('const std::int32_t', 'rounding_offset',
                      '(1 << (shift - 1))')
  emitter.EmitDeclare('std::int32_t*', 'temp_result',
                      'reinterpret_cast<std::int32_t*>('
                      'zipped_rhs + ((zipped_rhs_size + 15) / 16) * 16)')
  emitter.EmitDeclare('std::uint8_t*', 'result_chunk', 'result')
  emitter.EmitDeclare('std::int32_t*', 'mul_result_chunk', 'temp_result')
  emitter.EmitDeclare('const std::int32_t', 'mul_result_chunk_stride_bytes',
                      '((n * 4 + 31) / 32) * 32')
  emitter.EmitNewline()


def GenerateFullTempsCountersAndConsts(emitter, result_type, rows):
  """Generates all the boilerplate variables for the int32 and float gemms."""
  GenerateCommonTempsCountersAndConsts(emitter, rows)
  emitter.EmitDeclare('const std::int32_t', 'const_offset',
                      'lhs_offset * rhs_offset * k')
  emitter.EmitDeclare(result_type, 'result_chunk', 'result')
  emitter.EmitDeclare(result_type, 'mul_result_chunk', 'result')
  emitter.EmitDeclare('const std::int32_t', 'mul_result_chunk_stride_bytes',
                      'result_stride * 4')
  emitter.EmitNewline()


def ZipName(rows, leftovers, aligned):
  return zip_Nx8_neon.BuildName(rows, leftovers, aligned)


def GenerateZipRhs(emitter, aligned, cols, leftovers):
  """Emits the code responsible for zipping the rhs matrix."""
  emitter.EmitOpenBracket('for (int i = 0; i < col_chunks; ++i)')
  emitter.EmitCall(
      ZipName(3, leftovers, aligned),
      ['rhs_chunk', 'k', 'k', 'zipped_rhs_chunk', 'lhs_offset', 0])
  emitter.EmitAssignIncrement('rhs_chunk', 'chunk_size')
  emitter.EmitAssignIncrement('zipped_rhs_chunk', 'zipped_chunk_size')
  emitter.EmitCloseBracket()

  if cols is not 0:
    emitter.EmitCall(
        ZipName(cols, leftovers, aligned),
        ['rhs_chunk', 'k', 'k', 'zipped_rhs_chunk', 'lhs_offset', 0])
  emitter.EmitNewline()


def MulName(result_type, lhs_add, rhs_add, rows, cols):
  return mul_Nx8_Mx8_neon.BuildName(result_type, lhs_add, rhs_add, rows, cols)


def GetMulParams(result_type):
  params = ['zipped_lhs', 'zipped_rhs_chunk', 'padded_k', 'mul_result_chunk',
            'mul_result_chunk_stride_bytes']
  if result_type is 'float':
    params.append('result_scale')
  return params


def GenerateMulRows(emitter, result, result_type, lhs_add, rhs_add, aligned,
                    rows, cols, leftovers):
  """Emits code responsible for multiplication of one horizontal lhs strip."""
  emitter.EmitCall(
      ZipName(rows, leftovers, aligned),
      ['lhs_chunk', 'k', 'k', 'zipped_lhs', 'rhs_offset', 'const_offset'])
  emitter.EmitAssign('zipped_rhs_chunk', 'zipped_rhs')
  emitter.EmitAssign('mul_result_chunk', result)

  emitter.EmitOpenBracket('for (int j = 0; j < col_chunks; ++j)')

  emitter.EmitCall(
      MulName(result_type, lhs_add, rhs_add, rows, 3),
      GetMulParams(result_type))
  emitter.EmitAssignIncrement('zipped_rhs_chunk', 'zipped_chunk_size')
  emitter.EmitAssignIncrement('mul_result_chunk', 3)

  emitter.EmitCloseBracket()

  if cols is not 0:
    emitter.EmitCall(
        MulName(result_type, lhs_add, rhs_add, rows, cols),
        GetMulParams(result_type))


def GenerateQuantized8BitMul(emitter, aligned, rows, cols, leftovers):
  """Emits code for all lhs strips & leftover rows. Quantize after mul code."""
  emitter.EmitOpenBracket('for (int i = 0; i < row_chunks; ++i)')
  GenerateMulRows(emitter, 'temp_result', 'int32', False, True, aligned, 3,
                  cols, leftovers)
  emitter.EmitCall(
      qnt_Nx8_neon.BuildMultiQuantizeName(aligned, 3),
      ['temp_result', 'n', 'mul_result_chunk_stride_bytes',
       'zipped_lhs_3_offsets', 'result_chunk', 'result_stride',
       'multiplicative_offset', 'rounding_offset', '-shift'])
  emitter.EmitAssignIncrement('lhs_chunk', 'chunk_size')
  emitter.EmitAssignIncrement('result_chunk', 'result_chunk_stride')
  emitter.EmitCloseBracket()
  emitter.EmitNewline()

  if rows is not 0:
    GenerateMulRows(emitter, 'temp_result', 'int32', False, True, aligned, rows,
                    cols, leftovers)
    emitter.EmitCall(
        qnt_Nx8_neon.BuildMultiQuantizeName(aligned, rows),
        ['temp_result', 'n', 'mul_result_chunk_stride_bytes',
         'zipped_lhs_%d_offsets' % rows, 'result_chunk', 'result_stride',
         'multiplicative_offset', 'rounding_offset', '-shift'])


def GenerateFullMul(emitter, result_type, aligned, rows, cols, leftovers):
  emitter.EmitOpenBracket('for (int i = 0; i < row_chunks; ++i)')
  GenerateMulRows(emitter, 'result_chunk', result_type, True, True, aligned, 3,
                  cols, leftovers)
  emitter.EmitAssignIncrement('lhs_chunk', 'chunk_size')
  emitter.EmitAssignIncrement('result_chunk', 'result_chunk_stride')
  emitter.EmitCloseBracket()
  emitter.EmitNewline()

  if rows is not 0:
    GenerateMulRows(emitter, 'result_chunk', result_type, True, True, aligned,
                    rows, cols, leftovers)


def BuildName(output_type, aligned, rows, cols, leftover):
  name = BuildMainGemmName(output_type) + '_%d_%d_%d' % (rows, cols, leftover)
  if aligned:
    name += '_aligned'
  return name


def GetCommonGemmParameters():
  return [['std::uint8_t*', 'scratch'], ['const std::uint8_t*', 'lhs'],
          ['const std::uint8_t*', 'rhs'], ['std::int32_t', 'm'],
          ['std::int32_t', 'n'], ['std::int32_t', 'k'],
          ['std::int32_t', 'lhs_offset'], ['std::int32_t', 'rhs_offset']]


def GetGemmParameters(output_type, extra_params=None):
  """Prepares a (type, parameter) array for the gemm functions."""
  if extra_params is None:
    extra_params = []
  params = GetCommonGemmParameters()
  if output_type is _QUANTIZED_8BIT:
    params += [['std::int32_t', 'result_offset'],
               ['std::int32_t', 'multiplicative_offset'],
               ['std::int32_t', 'shift'], ['std::uint8_t*', 'result']]
  elif output_type is _FULL_32BIT:
    params += [['std::int32_t*', 'result']]
  elif output_type is _FULL_FLOAT:
    params += [['float', 'result_scale'], ['float*', 'result']]
  else:
    raise ConfigurationError('Unsupported output type: %s' % output_type)
  return params + extra_params


def GetStridedGemmParameters(output_type):
  return GetGemmParameters(output_type, [['std::int32_t', 'result_stride']])


def GenerateGemm(emitter, output_type, aligned, rows, cols, leftovers):
  """Build one gemm function for given row, col, and depth leftovers."""
  emitter.EmitFunctionBeginA(
      BuildName(output_type, aligned, rows, cols, leftovers),
      GetStridedGemmParameters(output_type), 'void')

  emitter.EmitAssert('m %% 3 == %d' % rows)
  emitter.EmitAssert('n %% 3 == %d' % cols)
  emitter.EmitAssert('k %% 8 == %d' % leftovers)

  if output_type is _QUANTIZED_8BIT:
    GenerateQuantized8BitTempsCountersAndConsts(emitter, rows)
    GenerateZipRhs(emitter, aligned, cols, leftovers)
    GenerateQuantized8BitMul(emitter, aligned, rows, cols, leftovers)
  elif output_type is _FULL_32BIT:
    GenerateFullTempsCountersAndConsts(emitter, 'std::int32_t*', rows)
    GenerateZipRhs(emitter, aligned, cols, leftovers)
    GenerateFullMul(emitter, 'int32', aligned, rows, cols, leftovers)
  elif output_type is _FULL_FLOAT:
    GenerateFullTempsCountersAndConsts(emitter, 'float*', rows)
    GenerateZipRhs(emitter, aligned, cols, leftovers)
    GenerateFullMul(emitter, 'float', aligned, rows, cols, leftovers)
  else:
    raise ConfigurationError('Unknown output type: %s' % output_type)

  emitter.EmitFunctionEnd()


def GenerateGemmCall(emitter, output_type, aligned, m_mod, n_mod, leftovers):
  emitter.EmitCall(
      emitter.Scope('internal',
                    BuildName(output_type, aligned, m_mod, n_mod, leftovers)),
      [p for (unused_t, p) in GetStridedGemmParameters(output_type)])


def GenerateGemmSwitch3(emitter, output_type, aligned, m_mod, n_mod):
  """Third level of main switch, choose optimized version on depth leftover."""
  emitter.EmitSwitch('k % 8')

  for leftovers in range(0, 8):
    emitter.EmitCase(leftovers)
    emitter.PushIndent()
    GenerateGemmCall(emitter, output_type, aligned, m_mod, n_mod, leftovers)
    emitter.EmitBreak()
    emitter.PopIndent()

  emitter.EmitSwitchEnd()


def GenerateGemmSwitch2(emitter, output_type, aligned, m_mod):
  """Second level of main switch, choose optimized version on cols leftover."""
  emitter.EmitSwitch('n % 3')

  for n_mod in range(0, 3):
    emitter.EmitCase(n_mod)
    emitter.PushIndent()
    GenerateGemmSwitch3(emitter, output_type, aligned, m_mod, n_mod)
    emitter.EmitBreak()
    emitter.PopIndent()

  emitter.EmitSwitchEnd()


def GenerateGemmSwitch1(emitter, output_type, aligned):
  """First level of main switch, choose optimized version on rows leftover."""
  emitter.EmitSwitch('m % 3')

  for m_mod in range(0, 3):
    emitter.EmitCase(m_mod)
    emitter.PushIndent()
    GenerateGemmSwitch2(emitter, output_type, aligned, m_mod)
    emitter.EmitBreak()
    emitter.PopIndent()

  emitter.EmitSwitchEnd()


def BuildMainGemmName(output_type):
  if output_type is _QUANTIZED_8BIT:
    return 'gemm_q8'
  elif output_type is _FULL_32BIT:
    return 'gemm_i32'
  elif output_type is _FULL_FLOAT:
    return 'gemm_f'
  else:
    raise ConfigurationError('Unsupported output type: %s' % output_type)


def BuildStridedMainGemmName(output_type):
  return BuildMainGemmName(output_type) + '_strided'


def GenerateMainGemmFunction(emitter, output_type):
  """Emit high level gemm function that switches between optimized versions."""
  emitter.EmitFunctionBeginA(
      BuildStridedMainGemmName(output_type),
      GetStridedGemmParameters(output_type), 'void')

  emitter.EmitDeclare('const bool', 'lhs_aligned',
                      '((reinterpret_cast<std::uintptr_t>(lhs) % 8) == 0)')
  emitter.EmitDeclare('const bool', 'rhs_aligned',
                      '((reinterpret_cast<std::uintptr_t>(rhs) % 8) == 0)')
  emitter.EmitDeclare('const bool', 'k_aligned', '((k % 8) == 0)')

  if output_type is _QUANTIZED_8BIT:
    emitter.EmitDeclare('const bool', 'result_aligned',
                        '((reinterpret_cast<std::uintptr_t>(result) % 8) == 0)')
    emitter.EmitDeclare('const bool', 'result_stride_aligned',
                        '((result_stride % 8) == 0)')
    emitter.EmitDeclare('const bool', 'aligned',
                        'lhs_aligned && rhs_aligned && result_aligned '
                        '&& k_aligned && result_stride_aligned')
  else:
    emitter.EmitDeclare('const bool', 'aligned',
                        'lhs_aligned && rhs_aligned && k_aligned')

  emitter.EmitIf('aligned')
  GenerateGemmSwitch1(emitter, output_type, True)
  emitter.EmitElse()
  GenerateGemmSwitch1(emitter, output_type, False)
  emitter.EmitEndif()
  emitter.EmitFunctionEnd()


def GenerateWrapperGemmFunction(emitter, output_type):
  emitter.EmitFunctionBeginA(
      BuildMainGemmName(output_type), GetGemmParameters(output_type), 'void')
  emitter.EmitCall(
      BuildStridedMainGemmName(output_type),
      [p for (unused_t, p) in GetGemmParameters(output_type)] + ['n'])
  emitter.EmitFunctionEnd()


def GenerateInternalFunctions(emitter):
  """Generate all the functions hidden in the internal namespace."""
  for output_type in [_QUANTIZED_8BIT, _FULL_32BIT, _FULL_FLOAT]:
    for aligned in [True, False]:
      for rows in range(0, 3):
        for cols in range(0, 3):
          for leftover in range(0, 8):
            GenerateGemm(emitter, output_type, aligned, rows, cols, leftover)
            emitter.EmitNewline()


def GeneratePublicFunctions(emitter):
  for output_type in [_QUANTIZED_8BIT, _FULL_32BIT, _FULL_FLOAT]:
    GenerateMainGemmFunction(emitter, output_type)
    emitter.EmitNewline()

    GenerateWrapperGemmFunction(emitter, output_type)
    emitter.EmitNewline()
