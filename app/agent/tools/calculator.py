"""
计算器工具

**安全约束：绝不用 `eval()`。**
  `eval` 能执行任意 Python 代码（`__import__('os').system(...)`），
  而工具参数是**模型生成的文本** —— 模型被提示注入诱导时会传入恶意表达式。
  这里用 AST 白名单求值：只允许算术运算的节点类型，其余一律拒绝。

支持:
  - 四则运算、幂、取余、整除
  - 括号与一元负号
  - 常用数学函数（sqrt/log/abs/round 等）与常量（pi/e）
"""
import ast
import logging
import math
import operator
from typing import Any

from app.agent.tools.base import Tool, ToolResult

logger = logging.getLogger("rag_api.agent.tools.calculator")

# 允许的二元运算符
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

# 允许的一元运算符
_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

# 允许的函数与常量（白名单，不含任何能接触文件/网络/导入的入口）
_FUNCTIONS = {
    "abs": abs, "round": round, "min": min, "max": max,
    "sqrt": math.sqrt, "log": math.log, "log10": math.log10,
    "exp": math.exp, "floor": math.floor, "ceil": math.ceil,
    "pow": pow, "sum": sum,
}
_CONSTANTS = {"pi": math.pi, "e": math.e, "tau": math.tau}

# 表达式长度上限（防止构造超深表达式耗尽栈）
_MAX_EXPR_LEN = 500

# 幂运算的指数上限。
#
# 这是一个**真实的 DoS 漏洞**（由测试 `test_huge_exponent_does_not_hang` 暴露）：
# AST 白名单只防住了"执行任意代码"，但没防住"消耗任意资源"。
# `9**9**9` = 9^387420489，结果是个约 3.7 亿位的整数 ——
# CPython 会尝试完整算出它，吃光内存并卡死进程（无法中断）。
#
# 上限取 10000：足以覆盖业务场景（2**10、10**6 之类），
# 又让结果规模保持在可控范围内。
_MAX_EXPONENT = 10_000

# 单个数值的位数上限（防止 2**9999 * 2**9999 这类"不超指数限制但结果爆炸"的构造）
_MAX_DIGITS = 100_000


class CalculatorTool(Tool):
    name = "calculator"
    description = (
        "计算数学表达式。当问题涉及算术运算、数值计算时使用。"
        "支持 + - * / // % ** 与括号，以及 sqrt/log/abs/round/floor/ceil 等函数。"
        "示例表达式：'(123 * 456) / 7'、'sqrt(2)'、'2 ** 10'"
    )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "要计算的数学表达式，例如 '(123 * 456) / 7'",
                }
            },
            "required": ["expression"],
        }

    async def _run(self, expression: str = "", **kwargs: Any) -> ToolResult:
        expr = (expression or "").strip()
        if not expr:
            return ToolResult(ok=False, content="表达式为空", error="empty")

        if len(expr) > _MAX_EXPR_LEN:
            return ToolResult(
                ok=False,
                content=f"表达式过长（超过 {_MAX_EXPR_LEN} 字符）",
                error="too_long",
            )

        try:
            value = self._safe_eval(expr)
        except _UnsafeExpression as e:
            logger.warning("拒绝不安全的表达式: %r (%s)", expr, e)
            return ToolResult(
                ok=False,
                content=f"表达式不被允许：{e}。只支持算术运算与常用数学函数。",
                error="unsafe_expression",
            )
        except ZeroDivisionError:
            return ToolResult(ok=False, content="除数不能为零", error="zero_division")
        except Exception as e:
            return ToolResult(
                ok=False, content=f"计算失败：{type(e).__name__}: {e}", error="eval_error"
            )

        # 整数结果去掉小数点，避免显示成 56088.0
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        return ToolResult(
            ok=True,
            content=f"{expr} = {value}",
            meta={"expression": expr, "value": value},
        )

    # ═══════════════════════════════════════════════════════════
    # 安全求值（AST 白名单）
    # ═══════════════════════════════════════════════════════════

    def _safe_eval(self, expr: str) -> float:
        try:
            tree = ast.parse(expr, mode="eval")
        except SyntaxError as e:
            raise _UnsafeExpression(f"语法错误: {e.msg}") from e
        return self._eval_node(tree.body)

    def _eval_node(self, node: ast.AST) -> Any:
        # 常量
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise _UnsafeExpression(f"不支持的常量类型 {type(node.value).__name__}")

        # 二元运算
        if isinstance(node, ast.BinOp):
            op = _BIN_OPS.get(type(node.op))
            if op is None:
                raise _UnsafeExpression(f"不支持的运算符 {type(node.op).__name__}")

            left = self._eval_node(node.left)
            right = self._eval_node(node.right)

            # 幂运算必须先检查指数规模 —— 见 _MAX_EXPONENT 的注释
            if isinstance(node.op, ast.Pow):
                self._guard_pow(left, right)

            result = op(left, right)
            self._guard_size(result)
            return result

        # 一元运算
        if isinstance(node, ast.UnaryOp):
            op = _UNARY_OPS.get(type(node.op))
            if op is None:
                raise _UnsafeExpression(f"不支持的一元运算符 {type(node.op).__name__}")
            return op(self._eval_node(node.operand))

        # 函数调用（仅限白名单函数，且不接受关键字参数）
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise _UnsafeExpression("只支持直接调用白名单函数")
            fname = node.func.id
            if fname not in _FUNCTIONS:
                raise _UnsafeExpression(f"未授权的函数 '{fname}'")
            if node.keywords:
                raise _UnsafeExpression("不支持关键字参数")
            args = [self._eval_node(a) for a in node.args]
            return _FUNCTIONS[fname](*args)

        # 常量（pi/e/tau）
        if isinstance(node, ast.Name):
            if node.id in _CONSTANTS:
                return _CONSTANTS[node.id]
            raise _UnsafeExpression(f"未授权的名称 '{node.id}'")

        # 元组（min/max 的多参数形式）
        if isinstance(node, ast.Tuple):
            return tuple(self._eval_node(e) for e in node.elts)

        raise _UnsafeExpression(f"不支持的语法 {type(node).__name__}")

    # ═══════════════════════════════════════════════════════════
    # 资源耗尽防护
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    def _guard_pow(base: Any, exponent: Any) -> None:
        """
        拦截会爆炸的幂运算。

        **必须在求值前检查** —— `9**9**9` 一旦开始计算就无法中断，
        只能等它耗尽内存（实测会卡死进程）。
        """
        if not isinstance(exponent, (int, float)):
            raise _UnsafeExpression("指数必须是数值")

        # 右结合：9**9**9 会先算 9**9=387420489，再算 9**387420489
        # 所以只要指数绝对值超限就拒绝
        if abs(exponent) > _MAX_EXPONENT:
            raise _UnsafeExpression(
                f"指数过大（{exponent}），上限 {_MAX_EXPONENT}。"
                f"如果你要算的是天文数字，请换用对数或说明意图。"
            )

        # 底数也很大时，小指数同样会爆炸（如 10**9**5）
        if isinstance(base, (int, float)) and abs(base) > 1:
            import math

            try:
                digits = abs(exponent) * math.log10(abs(base))
                if digits > _MAX_DIGITS:
                    raise _UnsafeExpression(
                        f"结果规模过大（约 {digits:.0f} 位数字），上限 {_MAX_DIGITS} 位"
                    )
            except (ValueError, OverflowError):
                pass

    @staticmethod
    def _guard_size(value: Any) -> None:
        """拦截结果位数超限的运算（乘法链也会爆炸，如 9999! 类的构造）"""
        if isinstance(value, int) and value.bit_length() > _MAX_DIGITS * 3.33:
            raise _UnsafeExpression(
                f"计算结果过大（约 {value.bit_length()} 位二进制），已拒绝"
            )


class _UnsafeExpression(Exception):
    """表达式不在白名单内"""
