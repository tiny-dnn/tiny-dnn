"""Multiply primitive optimized for the gemv operation."""

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


def GenerateLoadMultiplyAggregate(emitter, registers, lanes_count, aggregators,
                                  count, lhs, rhs_1, rhs_2):
  """Emit inner loop for 1 row x M cols multiplication."""
  emitter.EmitComment('General 1xM lanes loop.')
  emitter.EmitNumericalLabel(1)
  emitter.EmitNewline()
  emitter.EmitComment('Subtract counter.')
  emitter.EmitSubs(count, count, emitter.ImmediateConstant(8))
  emitter.EmitNewline()

  right_load = [registers.DoubleRegister() for unused_i in range(4)]
  left_load = registers.DoubleRegister()

  emitter.EmitVLoad(1, 8, left_load, emitter.DereferenceIncrement(lhs, 64))
  emitter.EmitVLoadA(1, 8, right_load, emitter.DereferenceIncrement(rhs_1, 64))

  emitter.EmitPldOffset(lhs, emitter.ImmediateConstant(64))
  emitter.EmitPldOffset(rhs_1, emitter.ImmediateConstant(128))

  multiply_results = [registers.QuadRegister() for unused_i in range(4)]

  for i in range(4):
    emitter.EmitVMull('u8', multiply_results[i], right_load[i], left_load)

  emitter.EmitVLoadA(1, 8, right_load[:lanes_count],
                     emitter.DereferenceIncrement(rhs_2, 64))
  emitter.EmitPldOffset(rhs_2, emitter.ImmediateConstant(lanes_count * 32))

  for i in range(4):
    emitter.EmitVPadal('u16', aggregators[i], multiply_results[i])

  for i in range(lanes_count):
    emitter.EmitVMull('u8', multiply_results[i], right_load[i], left_load)

  for i in range(lanes_count):
    emitter.EmitVPadal('u16', aggregators[i + 4], multiply_results[i])

  emitter.EmitNewline()
  emitter.EmitComment('Loop break.')
  emitter.EmitBneBack(1)
  emitter.EmitNewline()

  registers.FreeRegister(left_load)
  registers.FreeRegisters(right_load)
  registers.FreeRegisters(multiply_results)


def ReadLeft(emitter, registers, lhs):
  register = registers.QuadRegister()
  emitter.EmitVLoadAllLanes(1, 32, register, emitter.Dereference(lhs, None))
  return register


def ReadRight(emitter, registers, rhs, count):
  register = RegisterForCols(registers, count)
  emitter.EmitVLoad(1, 32, register, emitter.Dereference(rhs, 64))
  return register


def DuplicateGeneralRegister(emitter, registers, general_register,
                             min_register):
  duplicated = registers.QuadRegister(min_register)
  emitter.EmitVDup('32', duplicated, general_register)
  return duplicated


def GenerateAggregatorReduceStore(emitter, registers, lanes_count, aggregators,
                                  result_type, lhs_add, rhs_add, lhs, rhs_1,
                                  rhs_2, results):
  """Generates assembly responsible for reducing the 4 way aggregators."""
  temp = registers.QuadRegister()
  temp_2 = RegisterForCols(registers, lanes_count)

  if lhs_add:
    left_offset = ReadLeft(emitter, registers, lhs)
  else:
    left_offset = None

  if rhs_add:
    right_offset_1 = ReadRight(emitter, registers, rhs_1, 4)
    right_offset_2 = ReadRight(emitter, registers, rhs_2, lanes_count)
  else:
    right_offset_1 = None
    right_offset_2 = None

  if result_type is 'float':
    result_scale = DuplicateGeneralRegister(
        emitter, registers, registers.MapParameter('result_scale'), 4)
  else:
    result_scale = None

  emitter.EmitNewline()
  emitter.EmitComment('Horizontal reduce aggregators.')

  emitter.EmitVSumReduce('u32', len(aggregators), 4, [temp, temp_2],
                         aggregators)

  if lhs_add:
    emitter.EmitNewline()
    emitter.EmitComment('Add lhs offsets to aggregated rows.')
    emitter.EmitVAdd('s32', temp, temp, left_offset)
    emitter.EmitVAdd('s32', temp_2, temp_2, left_offset)

  if rhs_add:
    emitter.EmitNewline()
    emitter.EmitComment('Add rhs offset to aggregated rows.')
    emitter.EmitVAdd('s32', temp, temp, right_offset_1)
    emitter.EmitVAdd('s32', temp_2, temp_2, right_offset_2)

  if result_type is 'float':
    emitter.EmitNewline()
    emitter.EmitComment('Convert to float and scale.')
    emitter.EmitVCvt('f32', 's32', temp, temp)
    emitter.EmitVCvt('f32', 's32', temp_2, temp_2)
    emitter.EmitVMul('f32', temp, temp, result_scale)
    emitter.EmitVMul('f32', temp_2, temp_2, result_scale)

  emitter.EmitNewline()
  emitter.EmitComment('Store results.')
  emitter.EmitVStore(1, 32, temp, emitter.DereferenceIncrement(results))
  emitter.EmitVStoreAE(32, lanes_count, [temp_2], results)


def BuildName(result_type, lhs_add, rhs_add, lanes):
  name = 'mul_1x8_%dx8_%s' % (lanes, result_type)
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
  params = [['const std::uint8_t*', 'lhs'], ['const std::uint8_t*', 'rhs_1'],
            ['const std::uint8_t*', 'rhs_2'], ['std::int32_t', 'count'],
            [CppResultType(result_type), 'result']]
  if result_type is 'float':
    params.append(['float', 'result_scale'])
  return params


def GenerateAndClearAggregators(emitter, registers, count):
  """Prepare aggregators and emit aggregator clear code."""
  emitter.EmitNewline()
  emitter.EmitComment('Clear aggregators.')
  aggregators = [registers.QuadRegister() for unused_i in range(count)]
  for i in range(count):
    if i < 3:
      emitter.EmitVMov('i32', aggregators[i], emitter.ImmediateConstant(0))
    else:
      emitter.EmitVMov('i32', aggregators[i], aggregators[i - 3])
  emitter.EmitNewline()
  return aggregators


def GenerateMul1x8Mx8(emitter, result_type, lhs_add, rhs_add, lanes_count):
  """Generates the 1xN multiplication primitive."""
  if lanes_count < 1 or lanes_count > 4:
    raise ConfigurationError('Lanes should be: 1, 2, 3 or 4.')

  emitter.EmitFunctionBeginA(
      BuildName(result_type, lhs_add, rhs_add, lanes_count + 4),
      GetParameters(result_type), 'inline void')

  emitter.EmitAssert('count % 8 == 0')
  emitter.EmitAssert('count >= 8')
  emitter.EmitAsmBegin()

  registers = emitter.CreateRegisters()

  count = registers.MapParameter('count')

  lhs = registers.MapParameter('lhs')
  rhs_1 = registers.MapParameter('rhs_1')
  rhs_2 = registers.MapParameter('rhs_2')

  emitter.EmitPld(lhs)
  emitter.EmitPld(rhs_1)
  emitter.EmitPld(rhs_2)

  aggregators = GenerateAndClearAggregators(emitter, registers, lanes_count + 4)

  GenerateLoadMultiplyAggregate(emitter, registers, lanes_count, aggregators,
                                count, lhs, rhs_1, rhs_2)
  GenerateAggregatorReduceStore(emitter, registers, lanes_count, aggregators,
                                result_type, lhs_add, rhs_add, lhs, rhs_1,
                                rhs_2, registers.MapParameter('result'))

  emitter.EmitAsmEnd(registers.MappedParameters(), [],
                     registers.Clobbers() + ['cc', 'memory'])
  emitter.EmitFunctionEnd()


def GenerateFunctions(emitter, result_type, lhs_add, rhs_add):
  for lanes in range(1, 5):
    GenerateMul1x8Mx8(emitter, result_type, lhs_add, rhs_add, lanes)
    emitter.EmitNewline()


if __name__ == '__main__':
  GenerateFunctions(neon_emitter.NeonEmitter(), 'int32', True, True)
