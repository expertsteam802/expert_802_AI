"""Deterministic reasoning primitives with explicit safety boundaries."""
import ast
import operator
import re

_OPERATORS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}

def _evaluate_node(node):
    if isinstance(node, ast.Expression):
        return _evaluate_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        left, right = _evaluate_node(node.left), _evaluate_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 8:
            raise ValueError("Exponent is outside the supported range.")
        value = _OPERATORS[type(node.op)](left, right)
        if abs(value) > 10**15:
            raise ValueError("Result is outside the supported range.")
        return value
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_evaluate_node(node.operand))
    raise ValueError("Unsupported arithmetic expression.")

def calculate(expression):
    value = (expression or "").lower()
    replacements = [
        (r"\bdivided by\b", "/"), (r"\bover\b", "/"),
        (r"\bmultiplied by\b|\btimes\b", "*"), (r"\bplus\b", "+"),
        (r"\bminus\b", "-"), (r"\bmodulo\b|\bmod\b", "%"),
        (r"\bto the power of\b|\bpower of\b", "**"),
        (r"\bx\b", "*"),
    ]
    for pattern, replacement in replacements:
        value = re.sub(pattern, replacement, value)
    value = value.replace("^", "**")
    value = value.strip().rstrip("?.! ")
    if not re.fullmatch(r"[0-9+\-*/()%.*\s]+", value):
        raise ValueError("The expression contains unsupported content.")
    tree = ast.parse(value, mode="eval")
    return _evaluate_node(tree)

def derive_simple_facts(facts):
    """Return only explicit user-provided facts; never invent a relation."""
    return [fact for fact in (facts or []) if isinstance(fact, dict) and fact.get("text")]
