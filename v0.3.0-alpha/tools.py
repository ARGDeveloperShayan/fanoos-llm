import re
import ast
import operator
from datetime import datetime, timezone


# ─────────────────────────────────────────────────────────────
#  CALC – safe arithmetic evaluator
# ─────────────────────────────────────────────────────────────
_OPERATORS = {
    ast.Add:      operator.add,
    ast.Sub:      operator.sub,
    ast.Mult:     operator.mul,
    ast.Div:      operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod:      operator.mod,
    ast.Pow:      operator.pow,
    ast.USub:     operator.neg,
    ast.UAdd:     lambda x: x,
}


def _eval_node(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Non-numeric literal: {node.value!r}")
    if isinstance(node, ast.BinOp):
        op = _OPERATORS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        left  = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
            raise ZeroDivisionError("Division by zero")
        return op(left, right)
    if isinstance(node, ast.UnaryOp):
        op = _OPERATORS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported unary op: {type(node.op).__name__}")
        return op(_eval_node(node.operand))
    raise ValueError(f"Unsupported AST node: {type(node).__name__}")


def safe_calc(expression: str) -> str:
    try:
        tree   = ast.parse(expression.strip(), mode="eval")
        result = _eval_node(tree.body)
        if isinstance(result, float) and result.is_integer():
            return str(int(result))
        return str(round(result, 6))
    except ZeroDivisionError:
        return "[Error: division by zero]"
    except Exception as e:
        return f"[Calc error: {e}]"


# ─────────────────────────────────────────────────────────────
#  CLCK – current date / time (UTC)
# ─────────────────────────────────────────────────────────────
def clock_tool(args: str) -> str:
    now  = datetime.now(timezone.utc)
    arg  = args.strip().upper()
    if   arg == "DATE":  return now.strftime("%Y-%m-%d")
    elif arg == "TIME":  return now.strftime("%H:%M:%S UTC")
    elif arg == "YEAR":  return str(now.year)
    elif arg == "DAY":   return now.strftime("%A")
    else:                return now.strftime("%Y-%m-%d %H:%M:%S UTC")


# ─────────────────────────────────────────────────────────────
#  Tool dispatcher
# ─────────────────────────────────────────────────────────────
_TOOL_PATTERN = re.compile(r"\[(\w+):([^\]]*)\]")


def process_tools(text: str) -> str:
    def _dispatch(match):
        tool = match.group(1).upper()
        args = match.group(2).strip()
        if   tool == "CALC": return safe_calc(args)
        elif tool == "CLCK": return clock_tool(args)
        # ── Add future tools here ─────────────────────────────
        # elif tool == "SEARCH": return web_search(args)
        # elif tool == "WIKI":   return wikipedia_summary(args)
        # ─────────────────────────────────────────────────────
        return match.group(0)

    return _TOOL_PATTERN.sub(_dispatch, text)


# ─────────────────────────────────────────────────────────────
#  Quick self-test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        "[CALC:347+518]",
        "[CALC:100*25//100]",
        "[CALC:1/0]",
        "[CLCK:UTC]",
        "[CLCK:DATE]",
        "[CLCK:TIME]",
        "[CLCK:YEAR]",
        "[CLCK:DAY]",
        "The answer is [CALC:6*7]. Today is [CLCK:DAY].",
    ]
    for t in tests:
        print(f"  {t!r:45s} →  {process_tools(t)}")