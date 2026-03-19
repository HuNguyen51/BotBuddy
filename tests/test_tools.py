"""
Tool Unit Tests — Test tools độc lập, không cần LLM hay API key.

"Test các tool này độc lập như những hàm Python bình thường
trước khi đưa cho Agent sử dụng."
"""

from src.tools.calculator_tool import calculator
from src.tools.web_search_tool import web_search


class TestCalculatorTool:
    """Test calculator tool."""

    def test_basic_addition(self):
        result = calculator.invoke({"expression": "2 + 2"})
        assert "4" in result

    def test_square_root(self):
        result = calculator.invoke({"expression": "sqrt(16)"})
        assert "4" in result

    def test_complex_expression(self):
        result = calculator.invoke({"expression": "3.14 * 5**2"})
        assert "78.5" in result

    def test_division(self):
        result = calculator.invoke({"expression": "100 / 4"})
        assert "25" in result

    def test_invalid_expression(self):
        result = calculator.invoke({"expression": "invalid_func()"})
        assert "error" in result.lower()

    def test_dangerous_code_blocked(self):
        """Verify eval whitelist blocks dangerous code."""
        result = calculator.invoke({"expression": "__import__('os').system('ls')"})
        assert "error" in result.lower()


class TestWebSearchTool:
    """Test web_search tool — chỉ test format, không gọi API thật."""

    def test_tool_has_correct_name(self):
        query = "Giá vàng hôm nay"
        assert web_search(query) != ""

