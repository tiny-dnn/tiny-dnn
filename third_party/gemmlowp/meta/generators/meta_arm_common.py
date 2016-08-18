"""Common functions for the arm subroutine generation."""

import mul_1x8_Mx8_neon
import mul_Nx8_Mx8_neon
import qnt_Nx8_neon
import zip_Nx8_neon

_HEADER_COPYRIGHT = """// Copyright 2015 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
"""


def GenerateHeader(cc, header_name, preprocessor_directive):
  cc.EmitCodeNoSemicolon(_HEADER_COPYRIGHT)
  cc.EmitHeaderBegin(header_name)

  cc.EmitPreprocessor1('ifdef', preprocessor_directive)
  cc.EmitNewline()

  cc.EmitInclude('<cassert>')
  cc.EmitNewline()


def GenerateInternalFunctions(cc, neon):
  """Generate all the functions hidden in the internal namespace."""
  zip_Nx8_neon.GenerateFunctions(neon)
  cc.EmitNewline()

  mul_Nx8_Mx8_neon.GenerateFunctions(neon, 'int32', False, True)
  cc.EmitNewline()

  mul_Nx8_Mx8_neon.GenerateFunctions(neon, 'int32', True, True)
  cc.EmitNewline()

  mul_Nx8_Mx8_neon.GenerateFunctions(neon, 'float', True, True)
  cc.EmitNewline()

  mul_1x8_Mx8_neon.GenerateFunctions(neon, 'int32', False, True)
  cc.EmitNewline()

  mul_1x8_Mx8_neon.GenerateFunctions(neon, 'int32', True, True)
  cc.EmitNewline()

  mul_1x8_Mx8_neon.GenerateFunctions(neon, 'float', True, True)
  cc.EmitNewline()

  qnt_Nx8_neon.GenerateFunctions(neon, cc)
  cc.EmitNewline()


def GenerateFooter(cc, message):
  cc.EmitPreprocessor('else')
  cc.EmitPreprocessor1('warning', '"%s"' % message)
  cc.EmitPreprocessor('endif')
  cc.EmitNewline()
  cc.EmitHeaderEnd()
