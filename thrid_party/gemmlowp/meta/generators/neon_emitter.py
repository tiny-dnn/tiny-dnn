"""32bit ARM/NEON assembly emitter.

Used by code generators to produce ARM assembly with NEON simd code.
Provides tools for easier register management: named register variable
allocation/deallocation, and offers a more procedural/structured approach
to generating assembly.

TODO: right now neon emitter prints out assembly instructions immediately,
it might be beneficial to keep the whole structure and emit the assembly after
applying some optimizations like: instruction reordering or register reuse.

TODO: NeonRegister object assigns explicit registers at allocation time.
Similarily to emiting code, register mapping and reuse can be performed and
optimized lazily.
"""


class Error(Exception):
  """Module level error."""


class RegisterAllocationError(Error):
  """Cannot alocate registers."""


class LaneError(Error):
  """Wrong lane number."""


class ArgumentError(Error):
  """Wrong argument."""


def _Low(register):
  assert register[0] == 'q'
  num = int(register[1:])
  return 'd%d' % (num * 2)


def _High(register):
  assert register[0] == 'q'
  num = int(register[1:])
  return 'd%d' % (num * 2 + 1)


def _ExpandQuads(registers):
  doubles = []
  for register in registers:
    if register[0] == 'q':
      doubles.append(_Low(register))
      doubles.append(_High(register))
    else:
      doubles.append(register)
  return doubles


def _MakeCompatible(op1, op2, op3):
  if op1[0] == 'd' or op2[0] == 'd' or op3[0] == 'd':
    if op1[0] == 'q':
      op1 = _Low(op1)
    if op2[0] == 'q':
      op2 = _Low(op2)
    if op3[0] == 'q':
      op3 = _Low(op3)
  return (op1, op2, op3)


class _NeonRegisters32Bit(object):
  """Utility that keeps track of used 32bit ARM/NEON registers."""

  def __init__(self):
    self.double = set()
    self.double_ever = set()
    self.general = set()
    self.general_ever = set()
    self.parameters = set()

  def MapParameter(self, parameter):
    self.parameters.add(parameter)
    return '%%[%s]' % parameter

  def DoubleRegister(self, min_val=0):
    for i in range(min_val, 32):
      if i not in self.double:
        self.double.add(i)
        self.double_ever.add(i)
        return 'd%d' % i
    raise RegisterAllocationError('Not enough double registers.')

  def QuadRegister(self, min_val=0):
    for i in range(min_val, 16):
      if ((i * 2) not in self.double) and ((i * 2 + 1) not in self.double):
        self.double.add(i * 2)
        self.double.add(i * 2 + 1)
        self.double_ever.add(i * 2)
        self.double_ever.add(i * 2 + 1)
        return 'q%d' % i
    raise RegisterAllocationError('Not enough quad registers.')

  def GeneralRegister(self):
    for i in range(0, 16):
      if i not in self.general:
        self.general.add(i)
        self.general_ever.add(i)
        return 'r%d' % i
    raise RegisterAllocationError('Not enough general registers.')

  def MappedParameters(self):
    return [x for x in self.parameters]

  def Clobbers(self):
    return (['r%d' % i
             for i in self.general_ever] + ['d%d' % i
                                            for i in self.DoubleClobbers()])

  def DoubleClobbers(self):
    return sorted(self.double_ever)

  def FreeRegister(self, register):
    assert len(register) > 1
    num = int(register[1:])

    if register[0] == 'r':
      assert num in self.general
      self.general.remove(num)
    elif register[0] == 'd':
      assert num in self.double
      self.double.remove(num)
    elif register[0] == 'q':
      assert num * 2 in self.double
      assert num * 2 + 1 in self.double
      self.double.remove(num * 2)
      self.double.remove(num * 2 + 1)
    else:
      raise RegisterDeallocationError('Register not allocated: %s' % register)

  def FreeRegisters(self, registers):
    for register in registers:
      self.FreeRegister(register)


class NeonEmitter(object):
  """Emits ARM/NEON assembly opcodes."""

  def __init__(self, debug=False):
    self.ops = {}
    self.indent = ''
    self.debug = debug

  def PushIndent(self):
    self.indent += '  '

  def PopIndent(self):
    self.indent = self.indent[:-2]

  def EmitIndented(self, what):
    print self.indent + what

  def PushOp(self, op):
    if op in self.ops.keys():
      self.ops[op] += 1
    else:
      self.ops[op] = 1

  def ClearCounters(self):
    self.ops.clear()

  def EmitNewline(self):
    print ''

  def EmitPreprocessor1(self, op, param):
    print '#%s %s' % (op, param)

  def EmitPreprocessor(self, op):
    print '#%s' % op

  def EmitInclude(self, include):
    self.EmitPreprocessor1('include', include)

  def EmitCall1(self, function, param):
    self.EmitIndented('%s(%s);' % (function, param))

  def EmitAssert(self, assert_expression):
    if self.debug:
      self.EmitCall1('assert', assert_expression)

  def EmitHeaderBegin(self, header_name, includes):
    self.EmitPreprocessor1('ifndef', (header_name + '_H_').upper())
    self.EmitPreprocessor1('define', (header_name + '_H_').upper())
    self.EmitNewline()
    if includes:
      for include in includes:
        self.EmitInclude(include)
      self.EmitNewline()

  def EmitHeaderEnd(self):
    self.EmitPreprocessor('endif')

  def EmitCode(self, code):
    self.EmitIndented('%s;' % code)

  def EmitFunctionBeginA(self, function_name, params, return_type):
    self.EmitIndented('%s %s(%s) {' %
                      (return_type, function_name,
                       ', '.join(['%s %s' % (t, n) for (t, n) in params])))
    self.PushIndent()

  def EmitFunctionEnd(self):
    self.PopIndent()
    self.EmitIndented('}')

  def EmitAsmBegin(self):
    self.EmitIndented('asm volatile(')
    self.PushIndent()

  def EmitAsmMapping(self, elements, modifier):
    if elements:
      self.EmitIndented(': ' + ', '.join(['[%s] "%s"(%s)' % (d, modifier, d)
                                          for d in elements]))
    else:
      self.EmitIndented(':')

  def EmitClobbers(self, elements):
    if elements:
      self.EmitIndented(': ' + ', '.join(['"%s"' % c for c in elements]))
    else:
      self.EmitIndented(':')

  def EmitAsmEnd(self, outputs, inputs, clobbers):
    self.EmitAsmMapping(outputs, '+r')
    self.EmitAsmMapping(inputs, 'r')
    self.EmitClobbers(clobbers)
    self.PopIndent()
    self.EmitIndented(');')

  def EmitComment(self, comment):
    self.EmitIndented('// ' + comment)

  def EmitNumericalLabel(self, label):
    self.EmitIndented('"%d:"' % label)

  def EmitOp1(self, op, param1):
    self.PushOp(op)
    self.EmitIndented('"%s %s\\n"' % (op, param1))

  def EmitOp2(self, op, param1, param2):
    self.PushOp(op)
    self.EmitIndented('"%s %s, %s\\n"' % (op, param1, param2))

  def EmitOp3(self, op, param1, param2, param3):
    self.PushOp(op)
    self.EmitIndented('"%s %s, %s, %s\\n"' % (op, param1, param2, param3))

  def EmitAdd(self, destination, source, param):
    self.EmitOp3('add', destination, source, param)

  def EmitSubs(self, destination, source, param):
    self.EmitOp3('subs', destination, source, param)

  def EmitSub(self, destination, source, param):
    self.EmitOp3('sub', destination, source, param)

  def EmitMul(self, destination, source, param):
    self.EmitOp3('mul', destination, source, param)

  def EmitMov(self, param1, param2):
    self.EmitOp2('mov', param1, param2)

  def EmitBeqBack(self, label):
    self.EmitOp1('beq', '%db' % label)

  def EmitBeqFront(self, label):
    self.EmitOp1('beq', '%df' % label)

  def EmitBneBack(self, label):
    self.EmitOp1('bne', '%db' % label)

  def EmitBneFront(self, label):
    self.EmitOp1('bne', '%df' % label)

  def EmitVAdd(self, add_type, destination, source_1, source_2):
    destination, source_1, source_2 = _MakeCompatible(destination, source_1,
                                                      source_2)
    self.EmitOp3('vadd.%s' % add_type, destination, source_1, source_2)

  def EmitVAddw(self, add_type, destination, source_1, source_2):
    self.EmitOp3('vaddw.%s' % add_type, destination, source_1, source_2)

  def EmitVCvt(self, cvt_to, cvt_from, destination, source):
    self.EmitOp2('vcvt.%s.%s' % (cvt_to, cvt_from), destination, source)

  def EmitVDup(self, dup_type, destination, source):
    self.EmitOp2('vdup.%s' % dup_type, destination, source)

  def EmitVMov(self, mov_type, destination, source):
    self.EmitOp2('vmov.%s' % mov_type, destination, source)

  def EmitVQmovn(self, mov_type, destination, source):
    if destination[0] == 'q':
      destination = _Low(destination)
    self.EmitOp2('vqmovn.%s' % mov_type, destination, source)

  def EmitVQmovn2(self, mov_type, destination, source_1, source_2):
    self.EmitVQmovn(mov_type, _Low(destination), source_1)
    self.EmitVQmovn(mov_type, _High(destination), source_2)

  def EmitVQmovun(self, mov_type, destination, source):
    if destination[0] == 'q':
      destination = _Low(destination)
    self.EmitOp2('vqmovun.%s' % mov_type, destination, source)

  def EmitVMul(self, mul_type, destination, source_1, source_2):
    destination, source_1, source_2 = _MakeCompatible(destination, source_1,
                                                      source_2)
    self.EmitOp3('vmul.%s' % mul_type, destination, source_1, source_2)

  def EmitVMulScalar(self, mul_type, destination, source_1, source_2):
    self.EmitOp3('vmul.%s' % mul_type, destination, source_1, source_2)

  def EmitVMull(self, mul_type, destination, source_1, source_2):
    self.EmitOp3('vmull.%s' % mul_type, destination, source_1, source_2)

  def EmitVPadd(self, add_type, destination, source_1, source_2):
    self.EmitOp3('vpadd.%s' % add_type, destination, source_1, source_2)

  def EmitVPaddl(self, add_type, destination, source):
    self.EmitOp2('vpaddl.%s' % add_type, destination, source)

  def EmitVPadal(self, add_type, destination, source):
    self.EmitOp2('vpadal.%s' % add_type, destination, source)

  def EmitVLoad(self, load_no, load_type, destination, source):
    self.EmitVLoadA(load_no, load_type, [destination], source)

  def EmitVLoadA(self, load_no, load_type, destinations, source):
    self.EmitOp2('vld%d.%d' % (load_no, load_type),
                 '{%s}' % ', '.join(_ExpandQuads(destinations)), source)

  def EmitVLoadAE(self,
                  load_type,
                  elem_count,
                  destinations,
                  source,
                  alignment=None):
    bits_to_load = load_type * elem_count
    destinations = _ExpandQuads(destinations)
    if len(destinations) * 64 < bits_to_load:
      raise ArgumentError('To few destinations: %d to load %d bits.' %
                          (len(destinations), bits_to_load))

    while bits_to_load > 0:
      if bits_to_load >= 256:
        self.EmitVLoadA(1, 32, destinations[:4],
                        self.DereferenceIncrement(source, alignment))
        bits_to_load -= 256
        destinations = destinations[4:]
      elif bits_to_load >= 192:
        self.EmitVLoadA(1, 32, destinations[:3],
                        self.DereferenceIncrement(source, alignment))
        bits_to_load -= 192
        destinations = destinations[3:]
      elif bits_to_load >= 128:
        self.EmitVLoadA(1, 32, destinations[:2],
                        self.DereferenceIncrement(source, alignment))
        bits_to_load -= 128
        destinations = destinations[2:]
      elif bits_to_load >= 64:
        self.EmitVLoad(1, 32, destinations[0],
                       self.DereferenceIncrement(source, alignment))
        bits_to_load -= 64
        destinations = destinations[1:]
      else:
        destination = destinations[0]
        if bits_to_load == 56:
          self.EmitVLoad(1, 32, self.Lane(32, destination, 0),
                         self.DereferenceIncrement(source))
          self.EmitVLoad(1, 16, self.Lane(16, destination, 2),
                         self.DereferenceIncrement(source))
          self.EmitVLoad(1, 8, self.Lane(8, destination, 6),
                         self.DereferenceIncrement(source))
        elif bits_to_load == 48:
          self.EmitVLoad(1, 32, self.Lane(32, destination, 0),
                         self.DereferenceIncrement(source))
          self.EmitVLoad(1, 16, self.Lane(16, destination, 2),
                         self.DereferenceIncrement(source))
        elif bits_to_load == 40:
          self.EmitVLoad(1, 32, self.Lane(32, destination, 0),
                         self.DereferenceIncrement(source))
          self.EmitVLoad(1, 8, self.Lane(8, destination, 4),
                         self.DereferenceIncrement(source))
        elif bits_to_load == 32:
          self.EmitVLoad(1, 32, self.Lane(32, destination, 0),
                         self.DereferenceIncrement(source))
        elif bits_to_load == 24:
          self.EmitVLoad(1, 16, self.Lane(16, destination, 0),
                         self.DereferenceIncrement(source))
          self.EmitVLoad(1, 8, self.Lane(8, destination, 2),
                         self.DereferenceIncrement(source))
        elif bits_to_load == 16:
          self.EmitVLoad(1, 16, self.Lane(16, destination, 0),
                         self.DereferenceIncrement(source))
        elif bits_to_load == 8:
          self.EmitVLoad(1, 8, self.Lane(8, destination, 0),
                         self.DereferenceIncrement(source))
        else:
          raise ArgumentError('Wrong leftover: %d' % bits_to_load)
        return

  def EmitVLoadE(self, load_type, count, destination, source, alignment=None):
    self.EmitVLoadAE(load_type, count, [destination], source, alignment)

  def EmitVLoadAllLanes(self, load_no, load_type, destination, source):
    destinations = []
    if destination[0] == 'q':
      destinations.append(self.AllLanes(_Low(destination)))
      destinations.append(self.AllLanes(_High(destination)))
    else:
      destinations.append(self.AllLanes(destination))
    self.EmitVLoadA(load_no, load_type, destinations, source)

  def EmitPld(self, load_address_register):
    self.EmitOp1('pld', '[%s]' % load_address_register)

  def EmitPldOffset(self, load_address_register, offset):
    self.EmitOp1('pld', '[%s, %s]' % (load_address_register, offset))

  def EmitVShl(self, shift_type, destination, source, shift):
    self.EmitOp3('vshl.%s' % shift_type, destination, source, shift)

  def EmitVStore(self, store_no, store_type, source, destination):
    self.EmitVStoreA(store_no, store_type, [source], destination)

  def EmitVStoreA(self, store_no, store_type, sources, destination):
    self.EmitOp2('vst%d.%d' % (store_no, store_type),
                 '{%s}' % ', '.join(_ExpandQuads(sources)), destination)

  def EmitVStoreAE(self,
                   store_type,
                   elem_count,
                   sources,
                   destination,
                   alignment=None):
    bits_to_store = store_type * elem_count
    sources = _ExpandQuads(sources)
    if len(sources) * 64 < bits_to_store:
      raise ArgumentError('To few sources: %d to store %d bits.' %
                          (len(sources), bits_to_store))

    while bits_to_store > 0:
      if bits_to_store >= 256:
        self.EmitVStoreA(1, 32, sources[:4],
                         self.DereferenceIncrement(destination, alignment))
        bits_to_store -= 256
        sources = sources[4:]
      elif bits_to_store >= 192:
        self.EmitVStoreA(1, 32, sources[:3],
                         self.DereferenceIncrement(destination, alignment))
        bits_to_store -= 192
        sources = sources[3:]
      elif bits_to_store >= 128:
        self.EmitVStoreA(1, 32, sources[:2],
                         self.DereferenceIncrement(destination, alignment))
        bits_to_store -= 128
        sources = sources[2:]
      elif bits_to_store >= 64:
        self.EmitVStore(1, 32, sources[0],
                        self.DereferenceIncrement(destination, alignment))
        bits_to_store -= 64
        sources = sources[1:]
      else:
        source = sources[0]
        if bits_to_store == 56:
          self.EmitVStore(1, 32, self.Lane(32, source, 0),
                          self.DereferenceIncrement(destination))
          self.EmitVStore(1, 16, self.Lane(16, source, 2),
                          self.DereferenceIncrement(destination))
          self.EmitVStore(1, 8, self.Lane(8, source, 6),
                          self.DereferenceIncrement(destination))
        elif bits_to_store == 48:
          self.EmitVStore(1, 32, self.Lane(32, source, 0),
                          self.DereferenceIncrement(destination))
          self.EmitVStore(1, 16, self.Lane(16, source, 2),
                          self.DereferenceIncrement(destination))
        elif bits_to_store == 40:
          self.EmitVStore(1, 32, self.Lane(32, source, 0),
                          self.DereferenceIncrement(destination))
          self.EmitVStore(1, 8, self.Lane(8, source, 4),
                          self.DereferenceIncrement(destination))
        elif bits_to_store == 32:
          self.EmitVStore(1, 32, self.Lane(32, source, 0),
                          self.DereferenceIncrement(destination))
        elif bits_to_store == 24:
          self.EmitVStore(1, 16, self.Lane(16, source, 0),
                          self.DereferenceIncrement(destination))
          self.EmitVStore(1, 8, self.Lane(8, source, 2),
                          self.DereferenceIncrement(destination))
        elif bits_to_store == 16:
          self.EmitVStore(1, 16, self.Lane(16, source, 0),
                          self.DereferenceIncrement(destination))
        elif bits_to_store == 8:
          self.EmitVStore(1, 8, self.Lane(8, source, 0),
                          self.DereferenceIncrement(destination))
        else:
          raise ArgumentError('Wrong leftover: %d' % bits_to_store)
        return

  def EmitVStoreE(self, store_type, count, source, destination, alignment=None):
    self.EmitVStoreAE(store_type, count, [source], destination, alignment)

  def EmitVStoreOffset(self, store_no, store_type, source, destination, offset):
    self.EmitVStoreOffsetA(store_no, store_type, [source], destination, offset)

  def EmitVStoreOffsetA(self, store_no, store_type, sources, destination,
                        offset):
    self.EmitOp3('vst%d.%d' % (store_no, store_type),
                 '{%s}' % ', '.join(_ExpandQuads(sources)), destination, offset)

  def EmitVStoreOffsetE(self, store_type, count, source, destination, offset):
    """Emit assembly to store a number elements from the source registers."""
    if store_type is not 32:
      raise ArgumentError('Unsupported store_type: %d' % store_type)

    sources = []
    if source[0] == 'q':
      sources.append(_Low(source))
      sources.append(_High(source))
      if count * store_type > 128:
        raise ArgumentError('To many %dbit elements in a q register: %d' %
                            (store_type, count))
    else:
      sources.append(source)
      if count * store_type > 64:
        raise ArgumentError('To many %dbit elements in a d register: %d' %
                            (store_type, count))

    if count == 1:
      self.EmitVStoreOffset(1, store_type, self.Lane(store_type, sources[0], 0),
                            self.Dereference(destination, None), offset)
    elif count == 2:
      self.EmitVStoreOffset(1, store_type, sources[0],
                            self.Dereference(destination, None), offset)
    elif count == 3:
      self.EmitVStore(1, store_type, sources[0],
                      self.DereferenceIncrement(destination, None))
      self.EmitVStoreOffset(1, store_type, self.Lane(store_type, sources[1], 0),
                            self.Dereference(destination, None), offset)
      self.EmitSub(destination, destination, self.ImmediateConstant(8))
    elif count == 4:
      self.EmitVStoreOffsetA(1, store_type, sources,
                             self.Dereference(destination, None), offset)
    else:
      raise ArgumentError('To many elements: %d' % count)

  def EmitVSumReduce(self, reduce_type, elem_count, reduce_count, destinations,
                     sources):
    """Emit assembly for n-fold horizontal sum reduction."""
    if reduce_type is not 'u32':
      raise ArgumentError('Unsupported reduce: %s' % reduce_type)

    sources = _ExpandQuads(sources)

    destinations = _ExpandQuads(destinations)

    if len(destinations) * 2 < elem_count:
      raise ArgumentError('Not enough space in destination: %d vs %d' %
                          (len(destinations) * 2, elem_count))

    if len(sources) * 2 != elem_count * reduce_count:
      raise ArgumentError('Wrong number of sources: %d vs %d' %
                          (len(sources) * 2, elem_count * reduce_count))

    if reduce_count <= 1:
      raise ArgumentError('Unsupported reduce_count: %d' % reduce_count)

    while reduce_count > 1:
      if len(sources) % 2 == 1:
        sources.append(sources[-1])

      if reduce_count == 2:
        for i in range(len(destinations)):
          self.EmitVPadd(reduce_type, destinations[i], sources[2 * i],
                         sources[2 * i + 1])
        return
      else:
        sources_2 = []
        for i in range(len(sources) / 2):
          self.EmitVPadd(reduce_type, sources[2 * i], sources[2 * i],
                         sources[2 * i + 1])
          sources_2.append(sources[2 * i])
        reduce_count /= 2
        sources = sources_2

  def Dereference(self, value, alignment=None):
    if alignment:
      return '[%s:%d]' % (value, alignment)
    else:
      return '[%s]' % value

  def DereferenceIncrement(self, value, alignment=None):
    return '%s!' % self.Dereference(value, alignment)

  def ImmediateConstant(self, value):
    return '#%d' % value

  def AllLanes(self, value):
    return '%s[]' % value

  def Lane(self, bits, value, lane):
    """Get the proper n-bit lane from the given register."""
    registers = []
    if value[0] == 'q':
      registers.append(_Low(value))
      registers.append(_High(value))
    else:
      registers.append(value)

    elems_per_register = 64 / bits
    register = lane / elems_per_register
    lane %= elems_per_register

    return '%s[%d]' % (registers[register], lane)

  def CreateRegisters(self):
    return _NeonRegisters32Bit()
