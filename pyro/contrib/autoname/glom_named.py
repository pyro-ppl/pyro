import copy
from glom import T as _TTT  # noqa: F401
import gast
import functools
from .compilation import quote, unquote, compile_function, parse_function


class PrimitiveDetector(gast.NodeVisitor):
    """
    Checks whether a Call node contains ``pyro.sample`` or ``pyro.param``
    """
    def __init__(self):
        self._ret = False

    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, gast.Attribute):
            self._ret = self._ret or \
                        isinstance(node.func.value, gast.Name) and \
                        node.func.value.id == "pyro" and \
                        node.func.attr in ("sample", "param")

    def visit(self, node):
        super(PrimitiveDetector, self).visit(node)
        return self._ret


class NameRewriter(gast.NodeTransformer):

    def _make_glom(self, node):
        new_node = copy.copy(node)
        new_node.ctx = gast.Load()
        return quote("str(_TTT." + unquote(new_node) + ")[2:]")

    def visit_FunctionDef(self, node):
        node = self.generic_visit(node)
        def is_glom_decorator(d):
            if isinstance(d, gast.Name):
                return d.id == 'glom_name'
            elif isinstance(d, gast.Attribute):
                return is_glom_decorator(d.attr)
            return d == 'glom'
        node.decorator_list = list(filter(lambda d: not is_glom_decorator(d),
                                          node.decorator_list))
        return node

    def visit_Assign(self, node):
        if isinstance(node.value, gast.Call) and \
           PrimitiveDetector().visit(node.value) and \
           len(node.targets) == 1:
            new_name_node = self._make_glom(node.targets[0])
            if isinstance(node.value.args[0], gast.Str):
                node.value.args[0] = new_name_node
            else:
                node.value.args.insert(0, new_name_node)
        return node


def glom_name(fn):
    node = NameRewriter().visit(parse_function(fn))
    fn.__globals__.update({"_TTT": _TTT})
    return functools.wraps(fn)(compile_function(node, globals_=fn.__globals__))
