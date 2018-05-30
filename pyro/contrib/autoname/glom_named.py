import copy
import glom
import gast
from .compilation import quote, unquote, compile_function, parse_function


class PrimitiveDetector(gast.NodeVisitor):

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
        return quote("str(glom.T." + unquote(new_node) + ")[6:]")

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


def name(fn):
    node = NameRewriter().visit(parse_function(fn))
    g = {"glom": glom}.update(fn.__globals__)
    return compile_function(node, g)
