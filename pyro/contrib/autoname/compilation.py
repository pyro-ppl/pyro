# This code is MODIFIED from the version in Tangent (github.com/google/tangent)
# found at https://github.com/google/tangent/blob/master/tangent/compile.py
# and at https://github.com/google/tangent/blob/master/tangent/quoting.py
#
# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#      Unless required by applicable law or agreed to in writing, software
#      distributed under the License is distributed on an "AS IS" BASIS,
#      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#      See the License for the specific language governing permissions and
#      limitations under the License.
"""Going from AST or source code to executable code."""
from __future__ import absolute_import

from uuid import uuid4

import os
import tempfile

import inspect
import textwrap

import astor
import gast
import six
if six.PY3:
    from importlib import util
else:
    import imp


def compile_file(source, globals_=None):
    """
    Compile by saving to file and importing that.

    Compiling the AST/source code this way ensures that the source code is
    readable by e.g. `pdb` or `inspect`.

    Args:
      source: The code to compile, either as a string or as an AST.
      globals_: A dictionary of variables that should be available as globals in
          the compiled module. They will be monkey patched after importing the
          module.

    Returns:
      A module object containing the compiled source code.
    """
    if isinstance(source, gast.AST):
        source = to_source(source)

    # Write source to temporary file
    tempdir = tempfile.mkdtemp()
    uuid = str(uuid4().hex[:4])
    tmpname = os.path.join(tempdir, 'autoname_%s.py' % uuid)
    with open(tmpname, 'w') as f:
        f.write(source)

    # Load the temporary file as a module
    module_name = 'autoname_%s' % uuid
    if six.PY3:
        spec = util.spec_from_file_location(module_name, tmpname)
        m = util.module_from_spec(spec)
        spec.loader.exec_module(m)
    else:
        m = imp.load_source(module_name, tmpname)

    # Update the modules namespace
    if globals_:
        m.__dict__.update(globals_)
    return m


def compile_function(node, globals_=None):
    """
    Convert an AST or string into a function with inspectable source.

    This function uses `compile_file` internally, but instead of returning the
    entire module it will return the function only.

    Args:
      node: A `FunctionDef` node or a `Module` node which contains at least one
          `FunctionDef` node. If a module contains multiple functions, a handle
          to the first one will be returned.
      globals_: See `compile_file`

    Returns:
      A handle to the compiled function.

    Raises:
      TypeError: If the input is not a string or AST.
      ValueError: If no function can be found.
    """
    if not isinstance(node, gast.AST):
        if not isinstance(node, six.string_types):
            raise TypeError
        node = gast.parse(node)
    if isinstance(node, gast.Module):
        for succ in node.body:
            if isinstance(succ, gast.FunctionDef):
                name = succ.name
                break
        else:
            raise ValueError('no function found')
    elif isinstance(node, gast.FunctionDef):
        name = node.name
    else:
        raise TypeError
    module = compile_file(node, globals_)
    return getattr(module, name)


def to_source(node, indentation=' ' * 4):
    """
    Return source code of a given AST.
    """
    if isinstance(node, gast.AST):
        node = gast.gast_to_ast(node)
    generator = astor.code_gen.SourceGenerator(indentation, False,
                                               astor.string_repr.pretty_string)
    generator.visit(node)
    generator.result.append('\n')
    return astor.source_repr.pretty_source(generator.result).lstrip()


def parse_function(fn):
    """
    Get the source of a function and return its AST.
    """
    try:
        return parse_string(inspect.getsource(fn))
    except (IOError, OSError) as e:
        raise ValueError('Cannot parse function: %s' % e)


def parse_string(src):
    """
    Parse a string into an AST.
    """
    return gast.parse(textwrap.dedent(src))


def quote(src_string, return_expr=False):
    """
    Go from source code to AST nodes.

    This function returns a tree without enclosing `Module` or `Expr` nodes.

    Args:
      src_string: The source code to parse.
      return_expr: Whether or not to return a containing expression. This can be
          set to `True` if the result is to be part of a series of statements.

    Returns:
      An AST of the given source code.
    """
    node = parse_string(src_string)
    body = node.body
    if len(body) == 1:
        if isinstance(body[0], gast.Expr) and not return_expr:
            out = body[0].value
        else:
            out = body[0]
    else:
        out = node
    return out


def unquote(node):
    """
    Go from an AST to source code.
    """
    return to_source(node).strip()
