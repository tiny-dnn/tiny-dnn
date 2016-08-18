"""Generates the specialized gemv functions."""

import mul_1x8_Mx8_neon
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


def GenerateCommonTempsCountersAndConsts(emitter):
  """Generates common gemv boilerplate variables."""
  emitter.EmitDeclare('const std::int32_t', 'col_chunks', 'n / 8')
  emitter.EmitDeclare('const std::int32_t', 'padded_k', '((k + 7) / 8) * 8')
  emitter.EmitDeclare('const std::int32_t', 'chunk_size', 'k * 4')
  emitter.EmitDeclare('const std::int32_t', 'zipped_chunk_size',
                      '(padded_k + 16) * 4')
  emitter.EmitDeclare('const std::uint8_t*', 'rhs_chunk', 'rhs')
  emitter.EmitDeclare('std::uint8_t*', 'zipped_lhs', 'scratch')
  emitter.EmitDeclare('std::int32_t*', 'zipped_lhs_offsets',
                      'reinterpret_cast<std::int32_t*>(zipped_lhs + padded_k)')
  emitter.EmitDeclare('std::uint8_t*', 'zipped_rhs_1',
                      'scratch + ((padded_k + 31) / 16) * 16')
  emitter.EmitDeclare('std::uint8_t*', 'zipped_rhs_2',
                      'zipped_rhs_1 + zipped_chunk_size')
  emitter.EmitNewline()


def GenerateQuantized8BitTempsCountersAndConsts(emitter):
  """Generates all the boilerplate variables for the q8 gemm function."""
  GenerateCommonTempsCountersAndConsts(emitter)
  emitter.EmitDeclare('const std::int32_t', 'const_offset',
                      'lhs_offset * rhs_offset * k + result_offset')
  emitter.EmitDeclare('const std::int32_t', 'rounding_offset',
                      '(1 << (shift - 1))')
  emitter.EmitDeclare('std::int32_t*', 'temp_result',
                      'reinterpret_cast<std::int32_t*>('
                      'zipped_rhs_2 + zipped_chunk_size)')
  emitter.EmitDeclare('std::int32_t*', 'mul_result_chunk', 'temp_result')
  emitter.EmitNewline()


def GenerateFullTempsCountersAndConsts(emitter, result_type):
  """Generates all the boilerplate variables for the int32 and float gemms."""
  GenerateCommonTempsCountersAndConsts(emitter)
  emitter.EmitDeclare('const std::int32_t', 'const_offset',
                      'lhs_offset * rhs_offset * k')
  emitter.EmitDeclare(result_type, 'mul_result_chunk', 'result')
  emitter.EmitNewline()


def GenerateZipVector(emitter, aligned, leftovers):
  emitter.EmitCall(
      zip_Nx8_neon.BuildName(1, leftovers, aligned),
      ['lhs', 'k', 'k', 'zipped_lhs', 'rhs_offset', 0])


def GetMul2Params(result_type):
  params = ['zipped_lhs', 'zipped_rhs_1', 'zipped_rhs_2', 'padded_k',
            'mul_result_chunk']
  if result_type is 'float':
    params.append('result_scale')
  return params


def GetMulParams(result_type):
  params = ['zipped_lhs', 'zipped_rhs_1', 'padded_k', 'mul_result_chunk', 0]
  if result_type is 'float':
    params.append('result_scale')
  return params


def GenerateMulCols(emitter, result_type, lhs_add, rhs_add, aligned, cols,
                    leftovers):
  """Emits code responsible for multiplication of one horizontal lhs strip."""
  emitter.EmitOpenBracket('for (int i = 0; i < col_chunks; ++i)')
  emitter.EmitCall(
      zip_Nx8_neon.BuildName(4, leftovers, aligned),
      ['rhs_chunk', 'k', 'k', 'zipped_rhs_1', 'lhs_offset', 'const_offset'])
  emitter.EmitAssignIncrement('rhs_chunk', 'chunk_size')

  emitter.EmitCall(
      zip_Nx8_neon.BuildName(4, leftovers, aligned),
      ['rhs_chunk', 'k', 'k', 'zipped_rhs_2', 'lhs_offset', 'const_offset'])
  emitter.EmitAssignIncrement('rhs_chunk', 'chunk_size')

  emitter.EmitCall(
      mul_1x8_Mx8_neon.BuildName(result_type, lhs_add, rhs_add, 8),
      GetMul2Params(result_type))

  emitter.EmitAssignIncrement('mul_result_chunk', 8)
  emitter.EmitCloseBracket()

  if cols > 4:
    emitter.EmitCall(
        zip_Nx8_neon.BuildName(4, leftovers, aligned),
        ['rhs_chunk', 'k', 'k', 'zipped_rhs_1', 'lhs_offset', 'const_offset'])
    emitter.EmitAssignIncrement('rhs_chunk', 'chunk_size')

    emitter.EmitCall(
        zip_Nx8_neon.BuildName(cols - 4, leftovers, aligned),
        ['rhs_chunk', 'k', 'k', 'zipped_rhs_2', 'lhs_offset', 'const_offset'])

    emitter.EmitCall(
        mul_1x8_Mx8_neon.BuildName(result_type, lhs_add, rhs_add, cols),
        GetMul2Params(result_type))
  elif cols > 0:
    emitter.EmitCall(
        zip_Nx8_neon.BuildName(cols, leftovers, aligned),
        ['rhs_chunk', 'k', 'k', 'zipped_rhs_1', 'lhs_offset', 'const_offset'])

    emitter.EmitCall(
        mul_Nx8_Mx8_neon.BuildName(result_type, lhs_add, rhs_add, 1, cols),
        GetMulParams(result_type))


def GenerateQuantized8BitMul(emitter, aligned, cols, leftovers):
  """Emits code for all lhs strips & leftover rows. Quantize after mul code."""
  GenerateMulCols(emitter, 'int32', False, True, aligned, cols, leftovers)
  emitter.EmitCall(
      qnt_Nx8_neon.BuildName(1, cols, aligned),
      ['temp_result', 'n', 0, 'zipped_lhs_offsets', 'result', 0,
       'multiplicative_offset', 'rounding_offset', '-shift'])


def GenerateFullMul(emitter, result_type, aligned, cols, leftovers):
  GenerateMulCols(emitter, result_type, True, True, aligned, cols, leftovers)


def BuildName(output_type, aligned, cols, leftover):
  name = BuildMainGemvName(output_type) + '_%d_%d' % (cols, leftover)
  if aligned:
    name += '_aligned'
  return name


def GetCommonGemvParameters():
  return [['std::uint8_t*', 'scratch'], ['const std::uint8_t*', 'lhs'],
          ['const std::uint8_t*', 'rhs'], ['std::int32_t', 'n'],
          ['std::int32_t', 'k'], ['std::int32_t', 'lhs_offset'],
          ['std::int32_t', 'rhs_offset']]


def GetGemvParameters(output_type):
  """Prepares a (type, parameter) array for the gemm functions."""
  params = GetCommonGemvParameters()
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
  return params


def GenerateGemv(emitter, output_type, aligned, cols, leftovers):
  """Build one gemm function for given col, and depth leftovers."""
  emitter.EmitFunctionBeginA(
      BuildName(output_type, aligned, cols, leftovers),
      GetGemvParameters(output_type), 'void')

  emitter.EmitAssert('n %% 8 == %d' % cols)
  emitter.EmitAssert('k %% 8 == %d' % leftovers)

  if output_type is _QUANTIZED_8BIT:
    GenerateQuantized8BitTempsCountersAndConsts(emitter)
    GenerateZipVector(emitter, aligned, leftovers)
    GenerateQuantized8BitMul(emitter, aligned, cols, leftovers)
  elif output_type is _FULL_32BIT:
    GenerateFullTempsCountersAndConsts(emitter, 'std::int32_t*')
    GenerateZipVector(emitter, aligned, leftovers)
    GenerateFullMul(emitter, 'int32', aligned, cols, leftovers)
  elif output_type is _FULL_FLOAT:
    GenerateFullTempsCountersAndConsts(emitter, 'float*')
    GenerateZipVector(emitter, aligned, leftovers)
    GenerateFullMul(emitter, 'float', aligned, cols, leftovers)
  else:
    raise ConfigurationError('Unknown output type: %s' % output_type)

  emitter.EmitFunctionEnd()


def GenerateGemvCall(emitter, output_type, aligned, m_mod, leftovers):
  emitter.EmitCall(
      emitter.Scope('internal',
                    BuildName(output_type, aligned, m_mod, leftovers)),
      [p for (unused_t, p) in GetGemvParameters(output_type)])


def GenerateGemvSwitch2(emitter, output_type, aligned, n_mod):
  """Second level of main switch, choose optimized version on depth leftover."""
  emitter.EmitSwitch('k % 8')

  for leftovers in range(0, 8):
    emitter.EmitCase(leftovers)
    emitter.PushIndent()
    GenerateGemvCall(emitter, output_type, aligned, n_mod, leftovers)
    emitter.EmitBreak()
    emitter.PopIndent()

  emitter.EmitSwitchEnd()


def GenerateGemvSwitch1(emitter, output_type, aligned):
  """First level of main switch, choose optimized version on cols leftover."""
  emitter.EmitSwitch('n % 8')

  for n_mod in range(0, 8):
    emitter.EmitCase(n_mod)
    emitter.PushIndent()
    GenerateGemvSwitch2(emitter, output_type, aligned, n_mod)
    emitter.EmitBreak()
    emitter.PopIndent()

  emitter.EmitSwitchEnd()


def BuildMainGemvName(output_type):
  if output_type is _QUANTIZED_8BIT:
    return 'gemv_q8'
  elif output_type is _FULL_32BIT:
    return 'gemv_i32'
  elif output_type is _FULL_FLOAT:
    return 'gemv_f'
  else:
    raise ConfigurationError('Unsupported output type: %s' % output_type)


def GenerateMainGemvFunction(emitter, output_type):
  """Emit high level gemv function that switches between optimized versions."""
  emitter.EmitFunctionBeginA(
      BuildMainGemvName(output_type), GetGemvParameters(output_type), 'void')

  emitter.EmitDeclare('const bool', 'lhs_aligned',
                      '((reinterpret_cast<std::uintptr_t>(lhs) % 8) == 0)')
  emitter.EmitDeclare('const bool', 'rhs_aligned',
                      '((reinterpret_cast<std::uintptr_t>(rhs) % 8) == 0)')
  emitter.EmitDeclare('const bool', 'k_aligned', '((k % 8) == 0)')

  if output_type is _QUANTIZED_8BIT:
    emitter.EmitDeclare('const bool', 'result_aligned',
                        '((reinterpret_cast<std::uintptr_t>(result) % 8) == 0)')
    emitter.EmitDeclare('const bool', 'aligned',
                        'lhs_aligned && rhs_aligned && result_aligned '
                        '&& k_aligned')
  else:
    emitter.EmitDeclare('const bool', 'aligned',
                        'lhs_aligned && rhs_aligned && k_aligned')

  emitter.EmitIf('aligned')
  GenerateGemvSwitch1(emitter, output_type, True)
  emitter.EmitElse()
  GenerateGemvSwitch1(emitter, output_type, False)
  emitter.EmitEndif()
  emitter.EmitFunctionEnd()


def GenerateInternalFunctions(emitter):
  """Generate all the functions hidden in the internal namespace."""
  for output_type in [_QUANTIZED_8BIT, _FULL_32BIT, _FULL_FLOAT]:
    for aligned in [True, False]:
      for cols in range(0, 8):
        for leftover in range(0, 8):
          GenerateGemv(emitter, output_type, aligned, cols, leftover)
          emitter.EmitNewline()


def GeneratePublicFunctions(emitter):
  for output_type in [_QUANTIZED_8BIT, _FULL_32BIT, _FULL_FLOAT]:
    GenerateMainGemvFunction(emitter, output_type)
    emitter.EmitNewline()
