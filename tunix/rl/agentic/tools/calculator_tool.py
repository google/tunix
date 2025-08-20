from tunix.rl.multi_turn.tools.base_tool import BaseTool, ToolOutput


class CalculatorTool(BaseTool):
    """
    A basic calculator that supports addition, subtraction, multiplication, and division.
    """

    @property
    def json(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {
                            "type": "number",
                            "description": "The first operand"
                        },
                        "b": {
                            "type": "number",
                            "description": "The second operand"
                        },
                        "op": {
                            "type": "string",
                            "enum": ["+", "-", "*", "/"],
                            "description": "Operator, one of: + - * /"
                        }
                    },
                    "required": ["a", "b", "op"]
                }
            }
        }

    def forward(self, a: float, b: float, op: str) -> ToolOutput:
        try:
            if op == "+":
                result = a + b
            elif op == "-":
                result = a - b
            elif op == "*":
                result = a * b
            elif op == "/":
                if b == 0:
                    return ToolOutput(name=self.name, error="Division by zero is not allowed")
                result = a / b
            else:
                return ToolOutput(name=self.name, error=f"Unsupported operator: {op}")

            return ToolOutput(name=self.name, output=result)

        except Exception as e:
            return ToolOutput(name=self.name, error=f"{type(e).__name__}: {e}")
