"""Zip primitive used by the GEMM function.

Takes 1 to 3 rows of data and interleaves them in 8 byte chunks. Pads to
multiply of 8 length with zeros. Calculates row sums and appends those at the
end.
"""


class Error(Exception):
  """Module level error."""


class ConfigurationError(Error):
  """Unsupported configuration."""


class ZipLane(object):

  def __init__(self, input_address, load, aggregator):
    self.input_address = input_address
    self.load = load
    self.aggregator = aggregator


def GenerateZipLanes(emitter, registers, zip_lanes, input_address, stride):
  """Prepares read lanes for the zip operation.

  Args:
    emitter: ARM/NEON emitter.
    registers: ARM/NEON registers state.
    zip_lanes: number of lanes to prepare.
    input_address: register that contains the input address for the first lane.
    stride: memory stride for lane inputs.

  Returns:
    Array of ZipLane objects.
  """
  lanes = []
  last_address_register = input_address
  for i in range(zip_lanes):
    if not i:
      lanes.append(ZipLane(input_address, registers.DoubleRegister(),
                           registers.QuadRegister(4)))
    else:
      address_register = registers.GeneralRegister()
      lanes.append(ZipLane(address_register, registers.DoubleRegister(),
                           registers.QuadRegister(4)))
      emitter.EmitAdd(address_register, last_address_register, stride)
      last_address_register = address_register
  return lanes


def BuildName(zip_lanes, leftovers, aligned):
  name = 'zip_%dx8' % zip_lanes
  if leftovers:
    name += '_%d' % leftovers
  if aligned:
    name += '_aligned'
  return name


def GenerateClearAggregators(emitter, lanes):
  for lane in lanes:
    emitter.EmitVMov('i16', lane.aggregator, emitter.ImmediateConstant(0))


def GenerateLoadAggregateStore(emitter, lanes, output_address, alignment):
  """Emit inner loop code for reading N lanes and interweaving them."""
  emitter.EmitNewline()
  emitter.EmitComment('Load Aggregate Store.')

  for lane in lanes:
    emitter.EmitVLoad(
        1, 8, lane.load,
        emitter.DereferenceIncrement(lane.input_address, alignment))

  for lane in lanes:
    emitter.EmitVAddw('u8', lane.aggregator, lane.aggregator, lane.load)

  emitter.EmitVStoreA(1, 8, [lane.load for lane in lanes],
                      emitter.DereferenceIncrement(output_address, 64))


def GenerateLeftoverLoadAggregateStore(emitter, leftovers, lanes,
                                       output_address):
  """Handle leftovers when count is not a multiply of 8."""
  emitter.EmitNewline()
  emitter.EmitComment('Leftover Load Aggregate Store.')

  # Clear load registers.
  for lane in lanes:
    emitter.EmitVMov('i8', lane.load, emitter.ImmediateConstant(0))

  for lane in lanes:
    emitter.EmitVLoadE(8, leftovers, lane.load, lane.input_address, None)

  # Aggregate.
  for lane in lanes:
    emitter.EmitVAddw('u8', lane.aggregator, lane.aggregator, lane.load)

  # Store.
  emitter.EmitVStoreA(1, 8, [lane.load for lane in lanes],
                      emitter.DereferenceIncrement(output_address, 64))


def GenerateAggregatorReduction(emitter, registers, lanes, output_address,
                                multiplicative_offset, additive_offset):
  """Reduce 4 lane sum aggregators to 1 value and store the sums."""
  emitter.EmitNewline()
  emitter.EmitComment('Aggregator Reduction.')

  multiplier = registers.DoubleRegister()
  emitter.EmitVMov('32', emitter.Lane(32, multiplier, 0), multiplicative_offset)

  if len(lanes) == 1 or len(lanes) == 2:
    offset = registers.DoubleRegister()
    temp = registers.DoubleRegister()
  elif len(lanes) == 3 or len(lanes) == 4:
    offset = registers.QuadRegister()
    temp = registers.QuadRegister()
  else:
    raise ConfigurationError('Unexpected number of aggregators to reduce: %d' %
                             len(lanes))
  emitter.EmitVDup('32', offset, additive_offset)

  for lane in lanes:
    emitter.EmitVPaddl('u16', lane.aggregator, lane.aggregator)

  emitter.EmitVSumReduce('u32', len(lanes), 4, [temp], [lane.aggregator
                                                        for lane in lanes])
  emitter.EmitVMulScalar('i32', temp, temp, emitter.Lane(32, multiplier, 0))
  emitter.EmitVAdd('i32', temp, temp, offset)
  emitter.EmitVStore(1, 32, temp, emitter.Dereference(output_address, 64))


def GenerateZipNx8(emitter, zip_lanes, leftovers, aligned):
  """Emit the zip function for a given number of rows and row size leftovers."""
  if leftovers < 0 or leftovers > 7:
    raise ConfigurationError('Leftovers should be between 0 and 7 inclusive.')
  if zip_lanes < 1 or zip_lanes > 4:
    raise ConfigurationError('Zip_lanes should should be 1, 2, 3 or 4.')

  name = BuildName(zip_lanes, leftovers, aligned)

  emitter.EmitFunctionBeginA(name, [['const std::uint8_t*', 'source'],
                                    ['std::int32_t', 'count'],
                                    ['std::int32_t', 'stride'],
                                    ['std::uint8_t*', 'destination'],
                                    ['std::int32_t', 'multiplicative_offset'],
                                    ['std::int32_t', 'additive_offset']],
                             'void')
  emitter.EmitAssert('count %% 8 == %d' % leftovers)
  emitter.EmitAssert('count <= 2048')
  emitter.EmitAssert('count >= 8')
  emitter.EmitAssert('reinterpret_cast<std::uintptr_t>(destination) % 8 == 0')
  if aligned:
    emitter.EmitAssert('reinterpret_cast<std::uintptr_t>(source) % 8 == 0')
    if zip_lanes > 1:
      emitter.EmitAssert('stride % 8 == 0')
  emitter.EmitAsmBegin()

  registers = emitter.CreateRegisters()

  count = registers.MapParameter('count')
  output_address = registers.MapParameter('destination')

  lanes = GenerateZipLanes(emitter, registers, zip_lanes,
                           registers.MapParameter('source'),
                           registers.MapParameter('stride'))

  if leftovers:
    emitter.EmitSub(count, count, emitter.ImmediateConstant(leftovers))

  GenerateClearAggregators(emitter, lanes)

  emitter.EmitNewline()
  emitter.EmitNumericalLabel(1)
  emitter.EmitSubs(count, count, emitter.ImmediateConstant(8))

  GenerateLoadAggregateStore(emitter, lanes, output_address, 64 if aligned else
                             None)

  emitter.EmitNewline()
  emitter.EmitBneBack(1)

  if leftovers:
    GenerateLeftoverLoadAggregateStore(emitter, leftovers, lanes,
                                       output_address)

  GenerateAggregatorReduction(emitter, registers, lanes, output_address,
                              registers.MapParameter('multiplicative_offset'),
                              registers.MapParameter('additive_offset'))

  emitter.EmitAsmEnd(registers.MappedParameters(), [],
                     registers.Clobbers() + ['cc', 'memory'])
  emitter.EmitFunctionEnd()


def GenerateFunctions(emitter):
  for aligned in [True, False]:
    for lanes in range(1, 5):
      for leftovers in range(0, 8):
        GenerateZipNx8(emitter, lanes, leftovers, aligned)
        emitter.EmitNewline()
