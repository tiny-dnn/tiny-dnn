"""CC code emitter.

Used by generators to programatically prepare C++ code. Contains some simple
tools that allow generating nicely indented code and do basic correctness
checking.
"""


class Error(Exception):
  """Module level error."""


class NamespaceError(Error):
  """Invalid namespace operation."""


class HeaderError(Error):
  """Invalid cc header structure."""


class CCEmitter(object):
  """Emits c++ code."""

  def __init__(self, debug=False):
    self.indent = ''
    self.debug = debug
    self.namespaces = []
    self.header_name = None

  def PushIndent(self):
    self.indent += '  '

  def PopIndent(self):
    self.indent = self.indent[:-2]

  def EmitIndented(self, what):
    print self.indent + what

  def EmitNewline(self):
    print ''

  def EmitPreprocessor1(self, op, param):
    print '#%s %s' % (op, param)

  def EmitPreprocessor(self, op):
    print '#%s' % op

  def EmitInclude(self, include):
    self.EmitPreprocessor1('include', include)

  def EmitAssign(self, variable, value):
    self.EmitBinaryOp(variable, '=', value)

  def EmitAssignIncrement(self, variable, value):
    self.EmitBinaryOp(variable, '+=', value)

  def EmitBinaryOp(self, operand_1, op, operand_2):
    self.EmitCode('%s %s %s' % (operand_1, op, operand_2))

  def EmitCall(self, function, params=[]):
    self.EmitCode('%s(%s)' % (function, ', '.join(map(str, params))))

  def EmitCode(self, code):
    self.EmitIndented('%s;' % code)

  def EmitCodeNoSemicolon(self, code):
    self.EmitIndented('%s' % code)

  def EmitDeclare(self, decl_type, name, value):
    self.EmitAssign('%s %s' % (decl_type, name), value)

  def EmitAssert(self, assert_expression):
    if self.debug:
      self.EmitCall1('assert', assert_expression)

  def EmitHeaderBegin(self, header_name, includes=None):
    if includes is None:
      includes = []
    if self.header_name:
      raise HeaderError('Header already defined.')
    self.EmitPreprocessor1('ifndef', (header_name + '_H_').upper())
    self.EmitPreprocessor1('define', (header_name + '_H_').upper())
    self.EmitNewline()
    if includes:
      for include in includes:
        self.EmitInclude(include)
      self.EmitNewline()
    self.header_name = header_name

  def EmitHeaderEnd(self):
    if not self.header_name:
      raise HeaderError('Header undefined.')
    self.EmitPreprocessor1('endif',
                           ' // %s' % (self.header_name + '_H_').upper())
    self.header_name = None

  def EmitFunctionBeginA(self, function_name, params, return_type):
    self.EmitIndented('%s %s(%s) {' %
                      (return_type, function_name,
                       ', '.join(['%s %s' % (t, n) for (t, n) in params])))
    self.PushIndent()

  def EmitFunctionEnd(self):
    self.PopIndent()
    self.EmitIndented('}')

  def EmitNamespaceBegin(self, namespace):
    self.EmitCodeNoSemicolon('namespace %s {' % namespace)
    self.namespaces.append(namespace)

  def EmitNamespaceEnd(self):
    if not self.namespaces:
      raise NamespaceError('No namespace on stack.')
    self.EmitCodeNoSemicolon('}  // namespace %s' % self.namespaces.pop())

  def EmitComment(self, comment):
    self.EmitIndented('// ' + comment)

  def EmitOpenBracket(self, pre_bracket=None):
    if pre_bracket:
      self.EmitIndented('%s {' % pre_bracket)
    else:
      self.EmitIndented('{')
    self.PushIndent()

  def EmitCloseBracket(self):
    self.PopIndent()
    self.EmitIndented('}')

  def EmitSwitch(self, switch):
    self.EmitOpenBracket('switch (%s)' % switch)

  def EmitSwitchEnd(self):
    self.EmitCloseBracket()

  def EmitCase(self, value):
    self.EmitCodeNoSemicolon('case %s:' % value)

  def EmitBreak(self):
    self.EmitCode('break')

  def EmitIf(self, condition):
    self.EmitOpenBracket('if (%s)' % condition)

  def EmitElse(self):
    self.PopIndent()
    self.EmitCodeNoSemicolon('} else {')
    self.PushIndent()

  def EmitEndif(self):
    self.EmitCloseBracket()

  def Scope(self, scope, value):
    return '%s::%s' % (scope, value)
