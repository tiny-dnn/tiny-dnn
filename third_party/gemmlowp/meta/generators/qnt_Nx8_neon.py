"""Qnt primitive used by the GEMM function."""


class Error(Exception):
  """Module level error."""


class ConfigurationError(Error):
  """Unsupported configuration."""


class QntLane(object):

  def __init__(self, source, output, offset, load_1, load_2):
    self.source = source
    self.output = output
    self.offset = offset
    self.load_1 = load_1
    self.load_2 = load_2


def BuildName(lanes, leftovers, aligned):
  name = 'qnt_%dx8' % lanes
  if leftovers:
    name += '_%d' % leftovers
  if aligned:
    name += '_aligned'
  return name


def LoadAndDuplicateOffsets(emitter, registers, lanes, offsets):
  if lanes == 1 or lanes == 2 or lanes == 3:
    offset_registers = []
    for unused_i in range(lanes):
      register = registers.QuadRegister()
      emitter.EmitVLoadAllLanes(1, 32, register,
                                emitter.DereferenceIncrement(offsets, 32))
      offset_registers.append(register)
    return offset_registers
  else:
    raise ConfigurationError('Unsupported number of lanes: %d' % lanes)


def GenerateQntLanes(emitter, registers, qnt_lanes, source, stride, destination,
                     destination_stride, offsets):
  """Prepare lanes for reading unquantized multiplication results."""
  offset_registers = LoadAndDuplicateOffsets(emitter, registers, qnt_lanes,
                                             offsets)

  lanes = []
  last_input_register = source
  last_output_register = destination
  for i in range(0, qnt_lanes):
    if not i:
      lanes.append(QntLane(source,
                           destination,
                           offset_registers[i],
                           registers.QuadRegister(),  # load 1
                           registers.QuadRegister()))  # load 2
    else:
      input_register = registers.GeneralRegister()
      output_register = registers.GeneralRegister()
      lanes.append(QntLane(input_register,
                           output_register,
                           offset_registers[i],
                           registers.QuadRegister(),  # load 1
                           registers.QuadRegister()))  # load 2
      emitter.EmitAdd(input_register, last_input_register, stride)
      emitter.EmitAdd(output_register, last_output_register, destination_stride)
      last_input_register = input_register
      last_output_register = output_register
  return lanes


def DuplicateRegister(emitter, registers, value):
  register = registers.QuadRegister()
  emitter.EmitVDup('32', register, value)
  return register


def GenerateQuantize(emitter, lanes, count, multiplicative_offset,
                     rounding_offset, shift):
  """Inner loop for quantization: add offsets, multiply, round, shift."""
  for lane in lanes:
    emitter.EmitVAdd('i32', lane.load_1, lane.load_1, lane.offset)
    if count > 4:
      emitter.EmitVAdd('i32', lane.load_2, lane.load_2, lane.offset)

  for lane in lanes:
    emitter.EmitVMul('i32', lane.load_1, lane.load_1, multiplicative_offset)
    if count > 4:
      emitter.EmitVMul('i32', lane.load_2, lane.load_2, multiplicative_offset)

  for lane in lanes:
    emitter.EmitVAdd('i32', lane.load_1, lane.load_1, rounding_offset)
    if count > 4:
      emitter.EmitVAdd('i32', lane.load_2, lane.load_2, rounding_offset)

  for lane in lanes:
    emitter.EmitVShl('s32', lane.load_1, lane.load_1, shift)
    if count > 4:
      emitter.EmitVShl('s32', lane.load_2, lane.load_2, shift)

  for lane in lanes:
    if count <= 4:
      emitter.EmitVQmovn('s32', lane.load_1, lane.load_1)
    else:
      emitter.EmitVQmovn2('s32', lane.load_1, lane.load_1, lane.load_2)

  for lane in lanes:
    emitter.EmitVQmovun('s16', lane.load_1, lane.load_1)


def GenerateLoadQuantizeStore(emitter, lanes, multiplicative_offset,
                              rounding_offset, shift, aligned):
  """Load unquantized data from lanes, quantize, store final result."""
  for lane in lanes:
    emitter.EmitVLoadAE(32, 8, [lane.load_1, lane.load_2], lane.source, 128)

  for lane in lanes:
    emitter.EmitPld(lane.source)

  GenerateQuantize(emitter, lanes, 8, multiplicative_offset, rounding_offset,
                   shift)

  for lane in lanes:
    emitter.EmitVStoreE(8, 8, lane.load_1, lane.output, 64 if aligned else None)


def GenerateLeftoverLoadQuantizeStore(emitter, leftovers, lanes,
                                      multiplicative_offset, rounding_offset,
                                      shift):
  """Handle leftovers if row size not a multiply of 8."""
  for lane in lanes:
    emitter.EmitVLoadAE(32, leftovers, [lane.load_1, lane.load_2], lane.source,
                        64)

  GenerateQuantize(emitter, lanes, leftovers, multiplicative_offset,
                   rounding_offset, shift)

  for lane in lanes:
    emitter.EmitVStoreE(8, leftovers, lane.load_1, lane.output)


def GenerateQntNx8(emitter, qnt_lanes, leftovers, aligned):
  """Emits optimized quantization code for given lanes and row size."""
  if leftovers < 0 or leftovers > 7:
    raise ConfigurationError('Leftovers should be between 0 and 7 inclusive.')
  if qnt_lanes < 1 or qnt_lanes > 3:
    raise ConfigurationError('Qnt_lanes should should be 1, 2 or 3.')

  name = BuildName(qnt_lanes, leftovers, aligned)

  emitter.EmitFunctionBeginA(name, [['const std::int32_t*', 'source'],
                                    ['std::int32_t', 'count'],
                                    ['std::int32_t', 'stride'],
                                    ['const std::int32_t*', 'offsets'],
                                    ['std::uint8_t*', 'destination'],
                                    ['std::int32_t', 'destination_stride'],
                                    ['std::int32_t', 'multiplicative_offset'],
                                    ['std::int32_t', 'rounding_offset'],
                                    ['std::int32_t', 'shift']], 'void')
  emitter.EmitAssert('count %% 8 == %d' % leftovers)
  emitter.EmitAssert('count >= 8')
  emitter.EmitAssert('reinterpret_cast<std::uintptr_t>(source) % 8 == 0')
  if aligned:
    emitter.EmitAssert('reinterpret_cast<std::uintptr_t>(destination) % 8 == 0')
    if qnt_lanes > 1:
      emitter.EmitAssert('destination_stride % 8 == 0')
  emitter.EmitAsmBegin()

  registers = emitter.CreateRegisters()

  count = registers.MapParameter('count')

  multiplicative_offset = DuplicateRegister(
      emitter, registers, registers.MapParameter('multiplicative_offset'))
  rounding_offset = DuplicateRegister(emitter, registers,
                                      registers.MapParameter('rounding_offset'))
  shift = DuplicateRegister(emitter, registers, registers.MapParameter('shift'))

  lanes = GenerateQntLanes(emitter, registers, qnt_lanes,
                           registers.MapParameter('source'),
                           registers.MapParameter('stride'),
                           registers.MapParameter('destination'),
                           registers.MapParameter('destination_stride'),
                           registers.MapParameter('offsets'))

  if leftovers:
    emitter.EmitSubs(count, count, emitter.ImmediateConstant(leftovers))
    emitter.EmitBeqFront(2)

  emitter.EmitNewline()
  emitter.EmitNumericalLabel(1)
  emitter.EmitSubs(count, count, emitter.ImmediateConstant(8))

  GenerateLoadQuantizeStore(emitter, lanes, multiplicative_offset,
                            rounding_offset, shift, aligned)

  emitter.EmitNewline()
  emitter.EmitBneBack(1)

  if leftovers:
    emitter.EmitNumericalLabel(2)
    GenerateLeftoverLoadQuantizeStore(emitter, leftovers, lanes,
                                      multiplicative_offset, rounding_offset,
                                      shift)

  emitter.EmitAsmEnd(registers.MappedParameters(), [],
                     registers.Clobbers() + ['cc', 'memory'])
  emitter.EmitFunctionEnd()


def BuildMultiQuantizeName(aligned, rows):
  name = 'multi_qnt_%dx8' % rows
  if aligned:
    name = '%s_aligned' % name
  return name


def GenerateMultiQuantize(emitter, aligned, rows):
  """Emit main quantization code that switches between optimized versions."""
  name = BuildMultiQuantizeName(aligned, rows)
  emitter.EmitFunctionBeginA(name, [['const std::int32_t*', 'source'],
                                    ['std::int32_t', 'count'],
                                    ['std::int32_t', 'stride'],
                                    ['const std::int32_t*', 'offsets'],
                                    ['std::uint8_t*', 'destination'],
                                    ['std::int32_t', 'destination_stride'],
                                    ['std::int32_t', 'multiplicative_offset'],
                                    ['std::int32_t', 'rounding_offset'],
                                    ['std::int32_t', 'shift']], 'void')
  emitter.EmitSwitch('count % 8')

  for leftovers in range(8):
    emitter.EmitCase(leftovers)
    emitter.PushIndent()
    emitter.EmitCall(
        BuildName(rows, leftovers, aligned),
        ['source', 'count', 'stride', 'offsets', 'destination',
         'destination_stride', 'multiplicative_offset', 'rounding_offset',
         'shift'])
    emitter.EmitBreak()
    emitter.PopIndent()

  emitter.EmitSwitchEnd()
  emitter.EmitFunctionEnd()


def GenerateFunctions(neon, cc):
  for aligned in [True, False]:
    for lanes in range(1, 4):
      for leftovers in range(0, 8):
        GenerateQntNx8(neon, lanes, leftovers, aligned)
        neon.EmitNewline()

  for aligned in [True, False]:
    for rows in range(1, 4):
      GenerateMultiQuantize(cc, aligned, rows)
      cc.EmitNewline()
