"""64bit ARM/NEON assembly emitter.

Used by code generators to produce ARM assembly with NEON simd code.
Provides tools for easier register management: named register variable
allocation/deallocation, and offers a more procedural/structured approach
to generating assembly.

"""

_WIDE_TYPES = {8: 16,
               16: 32,
               32: 64,
               '8': '16',
               '16': '32',
               '32': '64',
               'i8': 'i16',
               'i16': 'i32',
               'i32': 'i64',
               'u8': 'u16',
               'u16': 'u32',
               'u32': 'u64',
               's8': 's16',
               's16': 's32',
               's32': 's64'}

_NARROW_TYPES = {64: 32,
                 32: 16,
                 16: 8,
                 '64': '32',
                 '32': '16',
                 '16': '8',
                 'i64': 'i32',
                 'i32': 'i16',
                 'i16': 'i8',
                 'u64': 'u32',
                 'u32': 'u16',
                 'u16': 'u8',
                 's64': 's32',
                 's32': 's16',
                 's16': 's8'}

_TYPE_BITS = {8: 8,
              16: 16,
              32: 32,
              64: 64,
              '8': 8,
              '16': 16,
              '32': 32,
              '64': 64,
              'i8': 8,
              'i16': 16,
              'i32': 32,
              'i64': 64,
              'u8': 8,
              'u16': 16,
              'u32': 32,
              'u64': 64,
              's8': 8,
              's16': 16,
              's32': 32,
              's64': 64,
              'f32': 32,
              'f64': 64,
              'b': 8,
              'h': 16,
              's': 32,
              'd': 64}


class Error(Exception):
  """Module level error."""


class RegisterAllocationError(Error):
  """Cannot alocate registers."""


class LaneError(Error):
  """Wrong lane number."""


class RegisterSubtypeError(Error):
  """The register needs to be lane-typed."""


class ArgumentError(Error):
  """Wrong argument."""


def _AppendType(type_name, register):
  """Calculates sizes and attaches the type information to the register."""
  if register.register_type is not 'v':
    raise ArgumentError('Only vector registers can have type appended.')

  if type_name in set([8, '8', 'i8', 's8', 'u8']):
    subtype = 'b'
    subtype_bits = 8
  elif type_name in set([16, '16', 'i16', 's16', 'u16']):
    subtype = 'h'
    subtype_bits = 16
  elif type_name in set([32, '32', 'i32', 's32', 'u32', 'f32']):
    subtype = 's'
    subtype_bits = 32
  elif type_name in set([64, '64', 'i64', 's64', 'u64', 'f64']):
    subtype = 'd'
    subtype_bits = 64
  else:
    raise ArgumentError('Unknown type: %s' % type_name)

  new_register = register.Copy()
  new_register.register_subtype = subtype
  new_register.register_subtype_count = register.register_bits / subtype_bits
  return new_register


def _UnsignedType(type_name):
  return type_name in set(['u8', 'u16', 'u32', 'u64'])


def _FloatType(type_name):
  return type_name in set(['f32', 'f64'])


def _WideType(type_name):
  if type_name in _WIDE_TYPES.keys():
    return _WIDE_TYPES[type_name]
  else:
    raise ArgumentError('No wide type for: %s' % type_name)


def _NarrowType(type_name):
  if type_name in _NARROW_TYPES.keys():
    return _NARROW_TYPES[type_name]
  else:
    raise ArgumentError('No narrow type for: %s' % type_name)


def _LoadStoreSize(register):
  if register.lane is None:
    return register.register_bits
  else:
    return register.lane_bits


def _MakeCompatibleDown(reg_1, reg_2, reg_3):
  bits = min([reg_1.register_bits, reg_2.register_bits, reg_3.register_bits])
  return (_Cast(bits, reg_1), _Cast(bits, reg_2), _Cast(bits, reg_3))


def _MakeCompatibleUp(reg_1, reg_2, reg_3):
  bits = max([reg_1.register_bits, reg_2.register_bits, reg_3.register_bits])
  return (_Cast(bits, reg_1), _Cast(bits, reg_2), _Cast(bits, reg_3))


def _Cast(bits, reg):
  if reg.register_bits is bits:
    return reg
  else:
    new_reg = reg.Copy()
    new_reg.register_bits = bits
    return new_reg


def _TypeBits(type_name):
  if type_name in _TYPE_BITS.keys():
    return _TYPE_BITS[type_name]
  else:
    raise ArgumentError('Unknown type: %s' % type_name)


def _RegisterList(list_type, registers):
  lanes = list(set([register.lane for register in registers]))
  if len(lanes) > 1:
    raise ArgumentError('Cannot mix lanes on a register list.')
  typed_registers = [_AppendType(list_type, register) for register in registers]

  if lanes[0] is None:
    return '{%s}' % ', '.join(map(str, typed_registers))
  elif lanes[0] is -1:
    raise ArgumentError('Cannot construct a list with all lane indexing.')
  else:
    typed_registers_nolane = [register.Copy() for register in typed_registers]
    for register in typed_registers_nolane:
      register.lane = None
      register.register_subtype_count = None
    return '{%s}[%d]' % (', '.join(map(str, typed_registers_nolane)), lanes[0])


class _GeneralRegister(object):
  """Arm v8 general register: (x|w)n."""

  def __init__(self,
               register_bits,
               number,
               dereference=False,
               dereference_increment=False):
    self.register_type = 'r'
    self.register_bits = register_bits
    self.number = number
    self.dereference = dereference
    self.dereference_increment = dereference_increment

  def Copy(self):
    return _GeneralRegister(self.register_bits, self.number, self.dereference,
                            self.dereference_increment)

  def __repr__(self):
    if self.register_bits is 64:
      text = 'x%d' % self.number
    elif self.register_bits <= 32:
      text = 'w%d' % self.number
    else:
      raise RegisterSubtypeError('Wrong bits (%d) for general register: %d' %
                                 (self.register_bits, self.number))
    if self.dereference:
      return '[%s]' % text
    else:
      return text


class _MappedParameter(object):
  """Object representing a C variable mapped to a register."""

  def __init__(self,
               name,
               register_bits=64,
               dereference=False,
               dereference_increment=False):
    self.name = name
    self.register_bits = register_bits
    self.dereference = dereference
    self.dereference_increment = dereference_increment

  def Copy(self):
    return _MappedParameter(self.name, self.register_bits, self.dereference,
                            self.dereference_increment)

  def __repr__(self):
    if self.register_bits is 64:
      text = '%%x[%s]' % self.name
    elif self.register_bits <= 32:
      text = '%%w[%s]' % self.name
    else:
      raise RegisterSubtypeError('Wrong bits (%d) for mapped parameter: %s' %
                                 (self.register_bits, self.name))
    if self.dereference:
      return '[%s]' % text
    else:
      return text


class _VectorRegister(object):
  """Arm v8 vector register Vn.TT."""

  def __init__(self,
               register_bits,
               number,
               register_subtype=None,
               register_subtype_count=None,
               lane=None,
               lane_bits=None):
    self.register_type = 'v'
    self.register_bits = register_bits
    self.number = number
    self.register_subtype = register_subtype
    self.register_subtype_count = register_subtype_count
    self.lane = lane
    self.lane_bits = lane_bits

  def Copy(self):
    return _VectorRegister(self.register_bits, self.number,
                           self.register_subtype, self.register_subtype_count,
                           self.lane, self.lane_bits)

  def __repr__(self):
    if self.register_subtype is None:
      raise RegisterSubtypeError('Register: %s%d has no lane types defined.' %
                                 (self.register_type, self.number))
    if (self.register_subtype_count is None or (self.lane is not None and
                                                self.lane is not -1)):
      typed_name = '%s%d.%s' % (self.register_type, self.number,
                                self.register_subtype)
    else:
      typed_name = '%s%d.%d%s' % (self.register_type, self.number,
                                  self.register_subtype_count,
                                  self.register_subtype)

    if self.lane is None or self.lane is -1:
      return typed_name
    elif self.lane >= 0 and self.lane < self.register_subtype_count:
      return '%s[%d]' % (typed_name, self.lane)
    else:
      raise LaneError('Wrong lane: %d for: %s' % (self.lane, typed_name))


class _ImmediateConstant(object):

  def __init__(self, value):
    self.register_type = 'i'
    self.value = value

  def Copy(self):
    return _ImmediateConstant(self.value)

  def __repr__(self):
    return '#%d' % self.value


class _NeonRegisters64Bit(object):
  """Utility that keeps track of used 32bit ARM/NEON registers."""

  def __init__(self):
    self.vector = set()
    self.vector_ever = set()
    self.general = set()
    self.general_ever = set()
    self.parameters = set()

  def MapParameter(self, parameter):
    self.parameters.add(parameter)
    return _MappedParameter(parameter)

  def _VectorRegisterNum(self, min_val=0):
    for i in range(min_val, 32):
      if i not in self.vector:
        self.vector.add(i)
        self.vector_ever.add(i)
        return i
    raise RegisterAllocationError('Not enough vector registers.')

  def DoubleRegister(self, min_val=0):
    return _VectorRegister(64, self._VectorRegisterNum(min_val))

  def QuadRegister(self, min_val=0):
    return _VectorRegister(128, self._VectorRegisterNum(min_val))

  def GeneralRegister(self):
    for i in range(0, 30):
      if i not in self.general:
        self.general.add(i)
        self.general_ever.add(i)
        return _GeneralRegister(64, i)
    raise RegisterAllocationError('Not enough general registers.')

  def MappedParameters(self):
    return [x for x in self.parameters]

  def Clobbers(self):
    return (['x%d' % i
             for i in self.general_ever] + ['v%d' % i
                                            for i in self.vector_ever])

  def FreeRegister(self, register):
    if register.register_type == 'v':
      assert register.number in self.vector
      self.vector.remove(register.number)
    elif register.register_type == 'r':
      assert register.number in self.general
      self.general.remove(register.number)
    else:
      raise RegisterAllocationError('Register not allocated: %s%d' %
                                    (register.register_type, register.number))

  def FreeRegisters(self, registers):
    for register in registers:
      self.FreeRegister(register)


class NeonEmitter64(object):
  """Emits ARM/NEON 64bit assembly opcodes."""

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
    destination, source_1, source_2 = _MakeCompatibleDown(destination, source_1,
                                                          source_2)
    self.EmitOp3('add', _AppendType(add_type, destination),
                 _AppendType(add_type, source_1),
                 _AppendType(add_type, source_2))

  def EmitVAddw(self, add_type, destination, source_1, source_2):
    wide_type = _WideType(add_type)
    destination = _AppendType(wide_type, destination)
    source_1 = _AppendType(wide_type, source_1)
    source_2 = _AppendType(add_type, source_2)
    if _UnsignedType(add_type):
      self.EmitOp3('uaddw', destination, source_1, source_2)
    else:
      self.EmitOp3('saddw', destination, source_1, source_2)

  def EmitVCvt(self, cvt_to, cvt_from, destination, source):
    if cvt_to == 'f32' and cvt_from == 's32':
      self.EmitOp2('scvtf', _AppendType('s32', destination),
                   _AppendType('s32', source))
    elif cvt_to == 'f32' and cvt_from == 'u32':
      self.EmitOp2('ucvtf', _AppendType('u32', destination),
                   _AppendType('u32', source))
    else:
      raise ArgumentError('Convert not supported, to: %s from: %s' % (cvt_to,
                                                                      cvt_from))

  def EmitVDup(self, dup_type, destination, source):
    if (isinstance(source, _GeneralRegister) or
        isinstance(source, _MappedParameter)):
      self.EmitOp2('dup', _AppendType(dup_type, destination), _Cast(
          _TypeBits(dup_type), source))
    else:
      self.EmitOp2('dup', _AppendType(dup_type, destination),
                   _AppendType(dup_type, source))

  def EmitVMov(self, mov_type, destination, source):
    if isinstance(source, _ImmediateConstant):
      self.EmitOp2('movi', _AppendType(mov_type, destination), source)
    elif (isinstance(source, _GeneralRegister) or
          isinstance(source, _MappedParameter)):
      self.EmitOp2('mov', _AppendType(mov_type, destination), _Cast(
          _TypeBits(mov_type), source))
    else:
      self.EmitOp2('mov', _AppendType(8, destination), _AppendType(8, source))

  def EmitVQmovn(self, mov_type, destination, source):
    narrow_type = _NarrowType(mov_type)
    if destination.register_bits * 2 == source.register_bits:
      self.EmitOp2('sqxtn', _AppendType(narrow_type, destination),
                   _AppendType(mov_type, source))
    elif destination.register_bits == source.register_bits:
      self.EmitOp2('sqxtn', _AppendType(narrow_type,
                                        _Cast(destination.register_bits / 2,
                                              destination)),
                   _AppendType(mov_type, source))

  def EmitVQmovn2(self, mov_type, destination, source_1, source_2):
    narrow_type = _NarrowType(mov_type)
    if (destination.register_bits != source_1.register_bits or
        destination.register_bits != source_2.register_bits):
      raise ArgumentError('Register sizes do not match.')
    self.EmitOp2('sqxtn', _AppendType(narrow_type,
                                      _Cast(destination.register_bits / 2,
                                            destination)),
                 _AppendType(mov_type, source_1))
    self.EmitOp2('sqxtn2', _AppendType(narrow_type, destination),
                 _AppendType(mov_type, source_2))

  def EmitVQmovun(self, mov_type, destination, source):
    narrow_type = _NarrowType(mov_type)
    if destination.register_bits * 2 == source.register_bits:
      self.EmitOp2('sqxtun', _AppendType(narrow_type, destination),
                   _AppendType(mov_type, source))
    elif destination.register_bits == source.register_bits:
      self.EmitOp2('sqxtun', _AppendType(narrow_type,
                                         _Cast(destination.register_bits / 2,
                                               destination)),
                   _AppendType(mov_type, source))

  def EmitVMul(self, mul_type, destination, source_1, source_2):
    destination, source_1, source_2 = _MakeCompatibleDown(destination, source_1,
                                                          source_2)
    if _FloatType(mul_type):
      self.EmitOp3('fmul', _AppendType(mul_type, destination),
                   _AppendType(mul_type, source_1),
                   _AppendType(mul_type, source_2))
    else:
      self.EmitOp3('mul', _AppendType(mul_type, destination),
                   _AppendType(mul_type, source_1),
                   _AppendType(mul_type, source_2))

  def EmitVMulScalar(self, mul_type, destination, source_1, source_2):
    self.EmitOp3('mul', _AppendType(mul_type, destination),
                 _AppendType(mul_type, source_1),
                 _AppendType(mul_type, source_2))

  def EmitVMull(self, mul_type, destination, source_1, source_2):
    wide_type = _WideType(mul_type)
    if _UnsignedType(mul_type):
      self.EmitOp3('umull', _AppendType(wide_type, destination),
                   _AppendType(mul_type, source_1),
                   _AppendType(mul_type, source_2))
    else:
      self.EmitOp3('smull', _AppendType(wide_type, destination),
                   _AppendType(mul_type, source_1),
                   _AppendType(mul_type, source_2))

  def EmitVPadd(self, add_type, destination, source_1, source_2):
    self.EmitOp3('addp', _AppendType(add_type, destination),
                 _AppendType(add_type, source_1),
                 _AppendType(add_type, source_2))

  def EmitVPaddl(self, add_type, destination, source):
    wide_type = _WideType(add_type)
    if _UnsignedType(add_type):
      self.EmitOp2('uaddlp', _AppendType(wide_type, destination),
                   _AppendType(add_type, source))
    else:
      self.EmitOp2('saddlp', _AppendType(wide_type, destination),
                   _AppendType(add_type, source))

  def EmitVPadal(self, add_type, destination, source):
    wide_type = _WideType(add_type)
    if _UnsignedType(add_type):
      self.EmitOp2('uadalp', _AppendType(wide_type, destination),
                   _AppendType(add_type, source))
    else:
      self.EmitOp2('sadalp', _AppendType(wide_type, destination),
                   _AppendType(add_type, source))

  def EmitVLoad(self, load_no, load_type, destination, source):
    self.EmitVLoadA(load_no, load_type, [destination], source)

  def EmitVLoadA(self, load_no, load_type, destinations, source):
    if source.dereference_increment:
      increment = sum([_LoadStoreSize(destination) for destination in
                       destinations]) / 8
      self.EmitVLoadAPostIncrement(load_no, load_type, destinations, source,
                                   self.ImmediateConstant(increment))
    else:
      self.EmitVLoadAPostIncrement(load_no, load_type, destinations, source,
                                   None)

  def EmitVLoadAPostIncrement(self, load_no, load_type, destinations, source,
                              increment):
    """Generate assembly to load memory to registers and increment source."""
    if len(destinations) == 1 and destinations[0].lane is -1:
      destination = '{%s}' % _AppendType(load_type, destinations[0])
      if increment:
        self.EmitOp3('ld%dr' % load_no, destination, source, increment)
      else:
        self.EmitOp2('ld%dr' % load_no, destination, source)
      return

    destination_list = _RegisterList(load_type, destinations)
    if increment:
      self.EmitOp3('ld%d' % load_no, destination_list, source, increment)
    else:
      self.EmitOp2('ld%d' % load_no, destination_list, source)

  def EmitVLoadAE(self,
                  load_type,
                  elem_count,
                  destinations,
                  source,
                  alignment=None):
    """Generate assembly to load an array of elements of given size."""
    bits_to_load = load_type * elem_count
    min_bits = min([destination.register_bits for destination in destinations])
    max_bits = max([destination.register_bits for destination in destinations])

    if min_bits is not max_bits:
      raise ArgumentError('Cannot mix double and quad loads.')

    if len(destinations) * min_bits < bits_to_load:
      raise ArgumentError('To few destinations: %d to load %d bits.' %
                          (len(destinations), bits_to_load))

    leftover_loaded = 0
    while bits_to_load > 0:
      if bits_to_load >= 4 * min_bits:
        self.EmitVLoadA(1, 32, destinations[:4],
                        self.DereferenceIncrement(source, alignment))
        bits_to_load -= 4 * min_bits
        destinations = destinations[4:]
      elif bits_to_load >= 3 * min_bits:
        self.EmitVLoadA(1, 32, destinations[:3],
                        self.DereferenceIncrement(source, alignment))
        bits_to_load -= 3 * min_bits
        destinations = destinations[3:]
      elif bits_to_load >= 2 * min_bits:
        self.EmitVLoadA(1, 32, destinations[:2],
                        self.DereferenceIncrement(source, alignment))
        bits_to_load -= 2 * min_bits
        destinations = destinations[2:]
      elif bits_to_load >= min_bits:
        self.EmitVLoad(1, 32, destinations[0],
                       self.DereferenceIncrement(source, alignment))
        bits_to_load -= min_bits
        destinations = destinations[1:]
      elif bits_to_load >= 64:
        self.EmitVLoad(1, 32, _Cast(64, destinations[0]),
                       self.DereferenceIncrement(source))
        bits_to_load -= 64
        leftover_loaded += 64
      elif bits_to_load >= 32:
        self.EmitVLoad(1, 32,
                       self.Lane(32, destinations[0], leftover_loaded / 32),
                       self.DereferenceIncrement(source))
        bits_to_load -= 32
        leftover_loaded += 32
      elif bits_to_load >= 16:
        self.EmitVLoad(1, 16,
                       self.Lane(16, destinations[0], leftover_loaded / 16),
                       self.DereferenceIncrement(source))
        bits_to_load -= 16
        leftover_loaded += 16
      elif bits_to_load is 8:
        self.EmitVLoad(1, 8, self.Lane(8, destinations[0], leftover_loaded / 8),
                       self.DereferenceIncrement(source))
        bits_to_load -= 8
        leftover_loaded += 8
      else:
        raise ArgumentError('Wrong leftover: %d' % bits_to_load)

  def EmitVLoadE(self, load_type, count, destination, source, alignment=None):
    self.EmitVLoadAE(load_type, count, [destination], source, alignment)

  def EmitVLoadAllLanes(self, load_no, load_type, destination, source):
    new_destination = destination.Copy()
    new_destination.lane = -1
    new_destination.lane_bits = load_type
    self.EmitVLoad(load_no, load_type, new_destination, source)

  def EmitPld(self, load_address_register):
    self.EmitOp2('prfm', 'pldl1keep', '[%s]' % load_address_register)

  def EmitPldOffset(self, load_address_register, offset):
    self.EmitOp2('prfm', 'pldl1keep',
                 '[%s, %s]' % (load_address_register, offset))

  def EmitVShl(self, shift_type, destination, source, shift):
    self.EmitOp3('sshl', _AppendType(shift_type, destination),
                 _AppendType(shift_type, source), _AppendType('i32', shift))

  def EmitVStore(self, store_no, store_type, source, destination):
    self.EmitVStoreA(store_no, store_type, [source], destination)

  def EmitVStoreA(self, store_no, store_type, sources, destination):
    if destination.dereference_increment:
      increment = sum([_LoadStoreSize(source) for source in sources]) / 8
      self.EmitVStoreAPostIncrement(store_no, store_type, sources, destination,
                                    self.ImmediateConstant(increment))
    else:
      self.EmitVStoreAPostIncrement(store_no, store_type, sources, destination,
                                    None)

  def EmitVStoreAPostIncrement(self, store_no, store_type, sources, destination,
                               increment):
    source_list = _RegisterList(store_type, sources)
    if increment:
      self.EmitOp3('st%d' % store_no, source_list, destination, increment)
    else:
      self.EmitOp2('st%d' % store_no, source_list, destination)

  def EmitVStoreAE(self,
                   store_type,
                   elem_count,
                   sources,
                   destination,
                   alignment=None):
    """Generate assembly to store an array of elements of given size."""
    bits_to_store = store_type * elem_count
    min_bits = min([source.register_bits for source in sources])
    max_bits = max([source.register_bits for source in sources])

    if min_bits is not max_bits:
      raise ArgumentError('Cannot mix double and quad stores.')

    if len(sources) * min_bits < bits_to_store:
      raise ArgumentError('To few destinations: %d to store %d bits.' %
                          (len(sources), bits_to_store))

    leftover_stored = 0
    while bits_to_store > 0:
      if bits_to_store >= 4 * min_bits:
        self.EmitVStoreA(1, 32, sources[:4],
                         self.DereferenceIncrement(destination, alignment))
        bits_to_store -= 4 * min_bits
        sources = sources[4:]
      elif bits_to_store >= 3 * min_bits:
        self.EmitVStoreA(1, 32, sources[:3],
                         self.DereferenceIncrement(destination, alignment))
        bits_to_store -= 3 * min_bits
        sources = sources[3:]
      elif bits_to_store >= 2 * min_bits:
        self.EmitVStoreA(1, 32, sources[:2],
                         self.DereferenceIncrement(destination, alignment))
        bits_to_store -= 2 * min_bits
        sources = sources[2:]
      elif bits_to_store >= min_bits:
        self.EmitVStore(1, 32, sources[0],
                        self.DereferenceIncrement(destination, alignment))
        bits_to_store -= min_bits
        sources = sources[1:]
      elif bits_to_store >= 64:
        self.EmitVStore(1, 32, _Cast(64, sources[0]),
                        self.DereferenceIncrement(destination, alignment))
        bits_to_store -= 64
        leftover_stored += 64
      elif bits_to_store >= 32:
        self.EmitVStore(1, 32, self.Lane(32, sources[0], leftover_stored / 32),
                        self.DereferenceIncrement(destination))
        bits_to_store -= 32
        leftover_stored += 32
      elif bits_to_store >= 16:
        self.EmitVStore(1, 16, self.Lane(16, sources[0], leftover_stored / 16),
                        self.DereferenceIncrement(destination))
        bits_to_store -= 16
        leftover_stored += 16
      elif bits_to_store >= 8:
        self.EmitVStore(1, 8, self.Lane(8, sources[0], leftover_stored / 8),
                        self.DereferenceIncrement(destination))
        bits_to_store -= 8
        leftover_stored += 8
      else:
        raise ArgumentError('Wrong leftover: %d' % bits_to_store)

  def EmitVStoreE(self, store_type, count, source, destination, alignment=None):
    self.EmitVStoreAE(store_type, count, [source], destination, alignment)

  def EmitVStoreOffset(self, store_no, store_type, source, destination, offset):
    self.EmitVStoreOffsetA(store_no, store_type, [source], destination, offset)

  def EmitVStoreOffsetA(self, store_no, store_type, sources, destination,
                        offset):
    self.EmitOp3('st%d' % store_no, _RegisterList(store_type, sources),
                 destination, offset)

  def EmitVStoreOffsetE(self, store_type, count, source, destination, offset):
    if store_type is not 32:
      raise ArgumentError('Unsupported store_type: %d' % store_type)

    if count == 1:
      self.EmitVStoreOffset(1, 32, self.Lane(32, source, 0),
                            self.Dereference(destination, None), offset)
    elif count == 2:
      self.EmitVStoreOffset(1, 32, _Cast(64, source),
                            self.Dereference(destination, None), offset)
    elif count == 3:
      self.EmitVStore(1, 32, _Cast(64, source),
                      self.DereferenceIncrement(destination, None))
      self.EmitVStoreOffset(1, 32, self.Lane(32, source, 2),
                            self.Dereference(destination, None), offset)
      self.EmitSub(destination, destination, self.ImmediateConstant(8))
    elif count == 4:
      self.EmitVStoreOffset(1, 32, source, self.Dereference(destination, None),
                            offset)
    else:
      raise ArgumentError('To many elements: %d' % count)

  def EmitVSumReduce(self, reduce_type, elem_count, reduce_count, destinations,
                     sources):
    """Generate assembly to perform n-fold horizontal sum reduction."""
    if reduce_type is not 'u32':
      raise ArgumentError('Unsupported reduce: %s' % reduce_type)

    if (elem_count + 3) / 4 > len(destinations):
      raise ArgumentError('To few destinations: %d (%d needed)' %
                          (len(destinations), (elem_count + 3) / 4))

    if elem_count * reduce_count > len(sources) * 4:
      raise ArgumentError('To few sources: %d' % len(sources))

    if reduce_count <= 1:
      raise ArgumentError('Unsupported reduce_count: %d' % reduce_count)

    sources = [_Cast(128, source) for source in sources]
    destinations = [_Cast(128, destination) for destination in destinations]

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

  def Dereference(self, value, unused_alignment=None):
    new_value = value.Copy()
    new_value.dereference = True
    return new_value

  def DereferenceIncrement(self, value, alignment=None):
    new_value = self.Dereference(value, alignment).Copy()
    new_value.dereference_increment = True
    return new_value

  def ImmediateConstant(self, value):
    return _ImmediateConstant(value)

  def AllLanes(self, value):
    return '%s[]' % value

  def Lane(self, bits, value, lane):
    new_value = value.Copy()
    if bits * (lane + 1) > new_value.register_bits:
      raise ArgumentError('Lane to big: (%d + 1) x %d > %d' %
                          (lane, bits, new_value.register_bits))
    new_value.lane = lane
    new_value.lane_bits = bits
    return new_value

  def CreateRegisters(self):
    return _NeonRegisters64Bit()
