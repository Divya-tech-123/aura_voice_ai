"""Safe mathematical evaluation tool for AURA.

Executes arithmetic computations securely using Abstract Syntax Tree (AST) parsing
without arbitrary code execution or unsafe eval() vulnerabilities.
"""

import ast
import math
from typing import Dict, Any, Union, Callable
import logging

from app.tools.base import BaseTool

logger = logging.getLogger(__name__)

# Permitted math functions
ALLOWED_FUNCTIONS: Dict[str, Callable] = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "ceil": math.ceil,
    "floor": math.floor,
    "abs": abs,
    "round": round,
    "pow": pow,
}

# Permitted mathematical constants
ALLOWED_CONSTANTS: Dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
}


def _eval_ast_node(node: ast.AST) -> Union[int, float]:
    """Recursively evaluate an AST node strictly restricting to allowed operations.

    Args:
        node: AST node to evaluate.

    Returns:
        Evaluated numeric result.

    Raises:
        ValueError: For forbidden nodes, unsafe constructs, or illegal operands.
        ZeroDivisionError: When division or modulo by zero occurs.
        OverflowError: When numeric computation exceeds allowable limits.
    """
    if isinstance(node, ast.Expression):
        return _eval_ast_node(node.body)

    # Numeric constants
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Disallowed constant type: {type(node.value).__name__}")

    # Named constants (pi, e, tau)
    if isinstance(node, ast.Name):
        if node.id in ALLOWED_CONSTANTS:
            return ALLOWED_CONSTANTS[node.id]
        raise ValueError(f"Undefined or unauthorized identifier: '{node.id}'")

    # Unary operators (+, -)
    if isinstance(node, ast.UnaryOp):
        operand = _eval_ast_node(node.operand)
        if isinstance(node.op, ast.UAdd):
            return +operand
        if isinstance(node.op, ast.USub):
            return -operand
        raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")

    # Binary operators (+, -, *, /, //, %, **)
    if isinstance(node, ast.BinOp):
        left = _eval_ast_node(node.left)
        right = _eval_ast_node(node.right)

        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise ZeroDivisionError("Division by zero")
            return left / right
        if isinstance(node.op, ast.FloorDiv):
            if right == 0:
                raise ZeroDivisionError("Division by zero")
            return left // right
        if isinstance(node.op, ast.Mod):
            if right == 0:
                raise ZeroDivisionError("Modulo by zero")
            return left % right
        if isinstance(node.op, ast.Pow):
            # Guard against resource exhaustion denial-of-service
            if abs(right) > 1000 or (abs(left) > 1e10 and right > 10):
                raise ValueError("Exponent or base is too large to safely compute.")
            return left ** right

        raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")

    # Function calls (e.g. sqrt(144), sin(0))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Direct function calls only. Method calls and attribute lookups are forbidden.")

        func_name = node.func.id
        if func_name not in ALLOWED_FUNCTIONS:
            raise ValueError(f"Function '{func_name}' is not permitted.")

        if node.keywords:
            raise ValueError("Keyword arguments are not supported.")

        args = [_eval_ast_node(arg) for arg in node.args]
        return ALLOWED_FUNCTIONS[func_name](*args)

    # Any other AST construct is considered unsafe and rejected
    raise ValueError(f"Unsupported syntax or potentially unsafe operation: {type(node).__name__}")


def calculate(expression: str) -> Dict[str, Any]:
    """Safely evaluate a mathematical expression.

    Args:
        expression: Mathematical expression string (e.g., '25 * 48', 'sqrt(144)').

    Returns:
        Dict with 'success' (bool), 'result' (float/int/None), and 'error' (str/None).
    """
    if not expression or not str(expression).strip():
        return {
            "success": False,
            "result": None,
            "error": "Expression cannot be empty.",
        }

    clean_expr = str(expression).strip()

    try:
        parsed_tree = ast.parse(clean_expr, mode="eval")
        raw_result = _eval_ast_node(parsed_tree)

        # Simplify floats that are exact integers (e.g. 1200.0 -> 1200)
        if isinstance(raw_result, float) and raw_result.is_integer() and not math.isinf(raw_result):
            result_val: Union[int, float] = int(raw_result)
        else:
            result_val = raw_result

        return {
            "success": True,
            "result": result_val,
            "error": None,
        }

    except ZeroDivisionError:
        return {
            "success": False,
            "result": None,
            "error": "Division by zero is undefined.",
        }
    except SyntaxError as exc:
        return {
            "success": False,
            "result": None,
            "error": f"Invalid mathematical syntax: {exc.msg}",
        }
    except (ValueError, OverflowError) as exc:
        return {
            "success": False,
            "result": None,
            "error": str(exc),
        }
    except Exception as exc:
        logger.warning(f"Unexpected error during calculation of '{clean_expr}': {exc}")
        return {
            "success": False,
            "result": None,
            "error": f"Calculation error: {str(exc)}",
        }


class CalculatorTool(BaseTool):
    """AURA executable tool for safe arithmetic and mathematical evaluation."""

    name: str = "calculator"
    description: str = (
        "Safely evaluates mathematical expressions and arithmetic. "
        "Supports operators (+, -, *, /, //, %, **), constants (pi, e), "
        "and functions (sqrt, sin, cos, tan, abs, round, ceil, floor, log, log10, pow)."
    )
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "The mathematical expression to evaluate (e.g. '25 * 48' or 'sqrt(144)').",
            }
        },
        "required": ["expression"],
    }

    def execute(self, expression: str = "", **kwargs: Any) -> Dict[str, Any]:
        """Execute the calculator tool."""
        return calculate(expression=expression)
