import ast
import math
import operator
from functools import lru_cache

_ALLOWED_FUNCS = {
    "log10": math.log10,
    "exp": math.exp,
    "max": max,
    "min": min,
    "abs": abs,
}

_ALLOWED_ATTRS = {
    "pi": math.pi,
    "e": math.e,
}

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}

_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_CMP_OPS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}


class _SafeExpressionEvaluator(ast.NodeVisitor):
    def __init__(self, variables):
        self._variables = variables

    def visit_Expression(self, node):
        return self.visit(node.body)

    def visit_Constant(self, node):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value).__name__}")

    def visit_Num(self, node):
        return node.n

    def visit_Name(self, node):
        if node.id in self._variables:
            return self._variables[node.id]
        if node.id in _ALLOWED_FUNCS:
            return _ALLOWED_FUNCS[node.id]
        raise ValueError(f"Unknown identifier: {node.id}")

    def visit_Attribute(self, node):
        if isinstance(node.value, ast.Name) and node.value.id == "np":
            if node.attr in _ALLOWED_ATTRS:
                return _ALLOWED_ATTRS[node.attr]
        raise ValueError("Unsupported attribute access")

    def visit_Call(self, node):
        if node.keywords:
            raise ValueError("Keyword arguments are not supported")
        func = self.visit(node.func)
        if func not in _ALLOWED_FUNCS.values():
            raise ValueError("Call to unsupported function")
        args = [self.visit(arg) for arg in node.args]
        return func(*args)

    def visit_BinOp(self, node):
        op_type = type(node.op)
        if op_type not in _BIN_OPS:
            raise ValueError(f"Unsupported operator: {op_type.__name__}")
        return _BIN_OPS[op_type](self.visit(node.left), self.visit(node.right))

    def visit_UnaryOp(self, node):
        op_type = type(node.op)
        if op_type not in _UNARY_OPS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        return _UNARY_OPS[op_type](self.visit(node.operand))

    def visit_Compare(self, node):
        left = self.visit(node.left)
        for op, comparator in zip(node.ops, node.comparators):
            right = self.visit(comparator)
            op_type = type(op)
            if op_type not in _CMP_OPS:
                raise ValueError(f"Unsupported comparison: {op_type.__name__}")
            if not _CMP_OPS[op_type](left, right):
                return False
            left = right
        return True

    def visit_BoolOp(self, node):
        if isinstance(node.op, ast.And):
            return all(self.visit(value) for value in node.values)
        if isinstance(node.op, ast.Or):
            return any(self.visit(value) for value in node.values)
        raise ValueError("Unsupported boolean operator")

    def visit_IfExp(self, node):
        return self.visit(node.body) if self.visit(node.test) else self.visit(node.orelse)

    def generic_visit(self, node):
        raise ValueError(f"Unsupported expression: {type(node).__name__}")


@lru_cache(maxsize=256)
def _parse_expression(expression: str) -> ast.Expression:
    return ast.parse(expression, mode="eval")


def safe_eval(expression: str, variables: dict) -> float:
    evaluator = _SafeExpressionEvaluator(variables)
    parsed = _parse_expression(expression)
    return evaluator.visit(parsed)
