"""Mul primitive used by the GEMM function.

The Mul primitive takes 1-3 zipped rows and 1-3 zipped columns and performs
matrix multiplication on those resulting in a small 1x1 to 3x3 block of results.
"""

import neon_emitter


class Error(Exception):
  """Module level error."""


class ConfigurationError(Error):
  """Unsupported configuration."""


def RegisterForCols(registers, cols, min_register=0):
  if cols == 1 or cols == 2:
    return registers.DoubleRegister(min_register * 2)
  elif cols == 3 or cols == 4:
    return registers.QuadRegister(min_register)
  else:
    raise ConfigurationError('Wrong no of cols for register: %d' % cols)


def GenerateAndClearAggregators(emitter, registers, count):
  """Prepare aggregators and emit aggregator clear code."""
  emitter.EmitComment('Clear aggregators.')
  aggregators = [registers.QuadRegister() for unused_i in range(count)]
  for i in range(count):
    if i < 3:
      emitter.EmitVMov('i32', aggregators[i], emitter.ImmediateConstant(0))
    else:
      emitter.EmitVMov('i32', aggregators[i], aggregators[i - 3])
  emitter.EmitNewline()
  return aggregators


def GenerateNxMLoadMultiplyAggregate(emitter, registers, left, right,
                                     aggregators, lhs, rhs, count):
  """Emit inner loop for N rows x M cols multiplication."""
  emitter.EmitComment('General NxM lanes loop.')
  emitter.EmitNumericalLabel(1)
  emitter.EmitNewline()
  emitter.EmitComment('Subtract counter.')
  emitter.EmitSubs(count, count, emitter.ImmediateConstant(8))
  emitter.EmitNewline()

  left_load = [registers.DoubleRegister() for unused_i in range(left)]
  right_load = [registers.DoubleRegister() for unused_i in range(right)]

  emitter.EmitVLoadA(1, 8, left_load, emitter.DereferenceIncrement(lhs, 64))
  emitter.EmitVLoadA(1, 8, right_load, emitter.DereferenceIncrement(rhs, 64))

  emitter.EmitPldOffset(lhs, emitter.ImmediateConstant(64))
  emitter.EmitPldOffset(rhs, emitter.ImmediateConstant(64))

  multiply_results = [registers.QuadRegister()
                      for unused_i in range(left * right)]

  for row in range(left):
    for col in range(right):
      index = row * right + col
      emitter.EmitVMull('u8', multiply_results[index], right_load[col],
                        left_load[row])

  for i in range(left * right):
    emitter.EmitVPadal('u16', aggregators[i], multiply_results[i])

  emitter.EmitNewline()
  emitter.EmitComment('Loop break.')
  emitter.EmitBneBack(1)
  emitter.EmitNewline()

  registers.FreeRegisters(left_load + right_load + multiply_results)


def Generate3x3LoadMultiplyAggregate(emitter, registers, aggregators, lhs, rhs,
                                     count):
  """Emit inner loop for 3 rows x 3 cols multiplication (register trick)."""
  emitter.EmitComment('3x3 lanes loop.')
  emitter.EmitNumericalLabel(1)
  emitter.EmitNewline()

  left_load = [registers.DoubleRegister() for unused_i in range(3)]
  right_load = [registers.DoubleRegister() for unused_i in range(3)]

  emitter.EmitVLoadA(1, 8, right_load, emitter.DereferenceIncrement(rhs, 64))
  emitter.EmitVLoad(1, 8, left_load[0], emitter.DereferenceIncrement(lhs, 64))

  temp = [registers.QuadRegister() for unused_i in range(4)]

  emitter.EmitVMull('u8', temp[0], left_load[0], right_load[0])
  emitter.EmitVLoad(1, 8, left_load[1], emitter.DereferenceIncrement(lhs, 64))

  emitter.EmitVMull('u8', temp[1], left_load[0], right_load[1])
  emitter.EmitVLoad(1, 8, left_load[2], emitter.DereferenceIncrement(lhs, 64))

  emitter.EmitVMull('u8', temp[2], left_load[0], right_load[2])
  emitter.EmitPldOffset(lhs, emitter.ImmediateConstant(64))

  emitter.EmitVMull('u8', temp[3], left_load[1], right_load[0])
  emitter.EmitPldOffset(rhs, emitter.ImmediateConstant(64))

  emitter.EmitVPadal('u16', aggregators[0], temp[0])
  emitter.EmitVPadal('u16', aggregators[1], temp[1])
  emitter.EmitVPadal('u16', aggregators[2], temp[2])
  emitter.EmitVPadal('u16', aggregators[3], temp[3])

  emitter.EmitVMull('u8', temp[0], left_load[1], right_load[1])
  emitter.EmitVMull('u8', temp[1], left_load[1], right_load[2])

  registers.FreeRegisters([left_load[0], left_load[1]])
  temp.append(registers.QuadRegister())

  emitter.EmitVMull('u8', temp[2], left_load[2], right_load[0])
  emitter.EmitVMull('u8', temp[3], left_load[2], right_load[1])

  emitter.EmitNewline()
  emitter.EmitComment('Subtract counter.')
  emitter.EmitSubs(count, count, emitter.ImmediateConstant(8))
  emitter.EmitNewline()

  emitter.EmitVMull('u8', temp[4], left_load[2], right_load[2])

  emitter.EmitVPadal('u16', aggregators[4], temp[0])
  emitter.EmitVPadal('u16', aggregators[5], temp[1])
  emitter.EmitVPadal('u16', aggregators[6], temp[2])
  emitter.EmitVPadal('u16', aggregators[7], temp[3])
  emitter.EmitVPadal('u16', aggregators[8], temp[4])

  emitter.EmitNewline()
  emitter.EmitComment('Loop break.')
  emitter.EmitBneBack(1)
  emitter.EmitNewline()

  registers.FreeRegisters(temp + right_load)
  registers.FreeRegister(left_load[2])


def ReadParams(emitter, registers, input_address, elements, min_register):
  register = RegisterForCols(registers, elements, min_register)
  emitter.EmitVLoad(1, 32, register, emitter.Dereference(input_address, 64))
  return register


def Duplicate(emitter, registers, rows, cols, min_register, values):
  """Populate a grid of registers duplicating provided values."""
  duplicated = [RegisterForCols(registers, cols, min_register)
                for unused_i in range(rows)]

  for i in range(rows):
    emitter.EmitVDup('32', duplicated[i], emitter.Lane(32, values, i))

  return duplicated


def DuplicateGeneralRegister(emitter, registers, cols, general_register,
                             min_register):
  duplicated = RegisterForCols(registers, cols, min_register)
  emitter.EmitVDup('32', duplicated, general_register)
  return duplicated


def ReduceAggregators(emitter, registers, aggregators):
  if len(aggregators) == 1 or len(aggregators) == 2:
    register = registers.DoubleRegister()
  elif len(aggregators) == 3 or len(aggregators) == 4:
    register = aggregators[0]
  else:
    raise ConfigurationError('Unsupported columns no: %d' % len(aggregators))
  emitter.EmitVSumReduce('u32', len(aggregators), 4, [register], aggregators)
  return register


def GenerateAggregatorReduceStore(emitter, registers, aggregators, result_type,
                                  lhs_add, rhs_add, left, right, lhs, rhs,
                                  results, results_stride):
  """Emit code that reduces 4 lane aggregators to 1 value, and stores them."""
  if lhs_add:
    left_offset = ReadParams(emitter, registers, lhs, left, 4)
    left_offsets = Duplicate(emitter, registers, left, right, 4, left_offset)
  else:
    left_offsets = None

  if rhs_add:
    right_offset = ReadParams(emitter, registers, rhs, right, 4)
  else:
    right_offset = None

  if result_type is 'float':
    result_scale = DuplicateGeneralRegister(
        emitter, registers, right, registers.MapParameter('result_scale'), 4)
  else:
    result_scale = None

  emitter.EmitNewline()
  emitter.EmitComment('Reduce rows.')

  row_temps = []
  for i in range(left):
    row_temps.append(ReduceAggregators(emitter, registers, aggregators[
        i * right:(i + 1) * right]))

  if lhs_add:
    emitter.EmitNewline()
    emitter.EmitComment('Add lhs offsets to aggregated rows.')
    for (row_temp, left_offset) in zip(row_temps, left_offsets):
      emitter.EmitVAdd('s32', row_temp, row_temp, left_offset)

  if rhs_add:
    emitter.EmitNewline()
    emitter.EmitComment('Add rhs offset to aggregated rows.')
    for row_temp in row_temps:
      emitter.EmitVAdd('s32', row_temp, row_temp, right_offset)

  if result_type is 'float':
    emitter.EmitNewline()
    emitter.EmitComment('Convert to float. Multiply by result scale.')
    for row_temp in row_temps:
      emitter.EmitVCvt('f32', 's32', row_temp, row_temp)
    for row_temp in row_temps:
      emitter.EmitVMul('f32', row_temp, row_temp, result_scale)

  emitter.EmitNewline()
  emitter.EmitComment('Store reduced rows.')
  for row_temp in row_temps:
    emitter.EmitVStoreOffsetE(32, right, row_temp, results, results_stride)


def BuildName(result_type, lhs_add, rhs_add, left, right):
  name = 'mul_%dx8_%dx8_%s' % (left, right, result_type)
  if lhs_add:
    name += '_lhsadd'
  if rhs_add:
    name += '_rhsadd'
  return name


def CppResultType(result_type):
  if result_type is 'int32':
    return 'std::int32_t*'
  elif result_type is 'float':
    return 'float*'
  else:
    raise ConfigurationError('Unsupported result type: %s' % result_type)


def GetParameters(result_type):
  params = [['const std::uint8_t*', 'lhs'], ['const std::uint8_t*', 'rhs'],
            ['std::int32_t', 'count'], [CppResultType(result_type), 'result'],
            ['std::int32_t', 'result_stride']]
  if result_type is 'float':
    params.append(['float', 'result_scale'])
  return params


def GenerateMulNx8Mx8(emitter, result_type, lhs_add, rhs_add, left, right):
  """Emit the multiply code for given rows and cols counts."""
  if left < 1 or left > 4:
    raise ConfigurationError('Left should be: 1, 2, 3 or 4.')
  if right < 1 or right > 4:
    raise ConfigurationError('Right should be: 1, 2, 3 or 4.')

  emitter.EmitFunctionBeginA(
      BuildName(result_type, lhs_add, rhs_add, left, right),
      GetParameters(result_type), 'inline void')
  emitter.EmitAssert('count % 8 == 0')
  emitter.EmitAssert('count >= 8')
  emitter.EmitAsmBegin()

  registers = emitter.CreateRegisters()

  count = registers.MapParameter('count')

  size = left * right

  lhs = registers.MapParameter('lhs')
  rhs = registers.MapParameter('rhs')

  emitter.EmitPld(lhs)
  emitter.EmitPld(rhs)

  aggregators = GenerateAndClearAggregators(emitter, registers, size)

  if size < 9:
    GenerateNxMLoadMultiplyAggregate(emitter, registers, left, right,
                                     aggregators, lhs, rhs, count)
  else:  # left == 3 and right == 3
    Generate3x3LoadMultiplyAggregate(emitter, registers, aggregators, lhs, rhs,
                                     count)

  GenerateAggregatorReduceStore(emitter, registers, aggregators, result_type,
                                lhs_add, rhs_add, left, right, lhs, rhs,
                                registers.MapParameter('result'),
                                registers.MapParameter('result_stride'))

  emitter.EmitAsmEnd(registers.MappedParameters(), [],
                     registers.Clobbers() + ['cc', 'memory'])
  emitter.EmitFunctionEnd()


def GenerateFunctions(emitter, result_type, lhs_add, rhs_add):
  for left_lanes in range(1, 4):
    for right_lanes in range(1, 4):
      GenerateMulNx8Mx8(emitter, result_type, lhs_add, rhs_add, left_lanes,
                        right_lanes)
      emitter.EmitNewline()

  GenerateMulNx8Mx8(emitter, result_type, lhs_add, rhs_add, 1, 4)
  emitter.EmitNewline()


if __name__ == '__main__':
  GenerateFunctions(neon_emitter.NeonEmitter(), 'int32', True, True)
