import asyncio
import re
from typing import List, Optional, Type

from dotenv import load_dotenv
from langchain.agents import AgentType, initialize_agent
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()


class MathInput(BaseModel):
    """Input for math operations"""

    expression: str = Field(description="Mathematical expression to evaluate")


class AsyncAdditionTool(BaseTool):
    name: str = "addition"
    description: str = "Adds two numbers together. Input should be in format 'number1 + number2'"
    args_schema: Type[BaseModel] = MathInput

    def _run(self, expression: str) -> str:
        """Synchronous version"""
        try:
            numbers = re.findall(r"-?\d+\.?\d*", expression)
            if len(numbers) >= 2:
                result = float(numbers[0]) + float(numbers[1])
                return f"The result of {numbers[0]} + {numbers[1]} is {result}"
            else:
                return "Please provide two numbers for addition"
        except Exception as e:
            return f"Error in addition: {str(e)}"

    async def _arun(self, expression: str) -> str:
        """Asynchronous version"""
        try:
            # Simulate async operation (in real scenarios, this might be a database call or API request)
            await asyncio.sleep(0.01)  # Small delay to demonstrate async behavior

            numbers = re.findall(r"-?\d+\.?\d*", expression)
            if len(numbers) >= 2:
                result = float(numbers[0]) + float(numbers[1])
                return f"[ASYNC] The result of {numbers[0]} + {numbers[1]} is {result}"
            else:
                return "[ASYNC] Please provide two numbers for addition"
        except Exception as e:
            return f"[ASYNC] Error in addition: {str(e)}"


class AsyncSubtractionTool(BaseTool):
    name: str = "subtraction"
    description: str = "Subtracts second number from first number. Input should be in format 'number1 - number2'"
    args_schema: Type[BaseModel] = MathInput

    def _run(self, expression: str) -> str:
        """Synchronous version"""
        try:
            numbers = re.findall(r"-?\d+\.?\d*", expression)
            if len(numbers) >= 2:
                result = float(numbers[0]) - float(numbers[1])
                return f"The result of {numbers[0]} - {numbers[1]} is {result}"
            else:
                return "Please provide two numbers for subtraction"
        except Exception as e:
            return f"Error in subtraction: {str(e)}"

    async def _arun(self, expression: str) -> str:
        """Asynchronous version"""
        try:
            await asyncio.sleep(0.01)

            numbers = re.findall(r"-?\d+\.?\d*", expression)
            if len(numbers) >= 2:
                result = float(numbers[0]) - float(numbers[1])
                return f"[ASYNC] The result of {numbers[0]} - {numbers[1]} is {result}"
            else:
                return "[ASYNC] Please provide two numbers for subtraction"
        except Exception as e:
            return f"[ASYNC] Error in subtraction: {str(e)}"


class AsyncMultiplicationTool(BaseTool):
    name: str = "multiplication"
    description: str = "Multiplies two numbers together. Input should be in format 'number1 * number2'"
    args_schema: Type[BaseModel] = MathInput

    def _run(self, expression: str) -> str:
        """Synchronous version"""
        try:
            numbers = re.findall(r"-?\d+\.?\d*", expression)
            if len(numbers) >= 2:
                result = float(numbers[0]) * float(numbers[1])
                return f"The result of {numbers[0]} * {numbers[1]} is {result}"
            else:
                return "Please provide two numbers for multiplication"
        except Exception as e:
            return f"Error in multiplication: {str(e)}"

    async def _arun(self, expression: str) -> str:
        """Asynchronous version"""
        try:
            await asyncio.sleep(0.01)

            numbers = re.findall(r"-?\d+\.?\d*", expression)
            if len(numbers) >= 2:
                result = float(numbers[0]) * float(numbers[1])
                return f"[ASYNC] The result of {numbers[0]} * {numbers[1]} is {result}"
            else:
                return "[ASYNC] Please provide two numbers for multiplication"
        except Exception as e:
            return f"[ASYNC] Error in multiplication: {str(e)}"


class AsyncDivisionTool(BaseTool):
    name: str = "division"
    description: str = "Divides first number by second number. Input should be in format 'number1 / number2'"
    args_schema: Type[BaseModel] = MathInput

    def _run(self, expression: str) -> str:
        """Synchronous version"""
        try:
            numbers = re.findall(r"-?\d+\.?\d*", expression)
            if len(numbers) >= 2:
                divisor = float(numbers[1])
                if divisor == 0:
                    return "Error: Division by zero is not allowed"
                result = float(numbers[0]) / divisor
                return f"The result of {numbers[0]} / {numbers[1]} is {result}"
            else:
                return "Please provide two numbers for division"
        except Exception as e:
            return f"Error in division: {str(e)}"

    async def _arun(self, expression: str) -> str:
        """Asynchronous version"""
        try:
            await asyncio.sleep(0.01)

            numbers = re.findall(r"-?\d+\.?\d*", expression)
            if len(numbers) >= 2:
                divisor = float(numbers[1])
                if divisor == 0:
                    return "[ASYNC] Error: Division by zero is not allowed"
                result = float(numbers[0]) / divisor
                return f"[ASYNC] The result of {numbers[0]} / {numbers[1]} is {result}"
            else:
                return "[ASYNC] Please provide two numbers for division"
        except Exception as e:
            return f"[ASYNC] Error in division: {str(e)}"


class AsyncCalculatorTool(BaseTool):
    name: str = "calculator"
    description: str = "Evaluates complex mathematical expressions. Use this for expressions with multiple operations."
    args_schema: Type[BaseModel] = MathInput

    def _run(self, expression: str) -> str:
        """Synchronous version"""
        try:
            clean_expr = re.sub(r"[^0-9+\-*/().\s]", "", expression)

            if re.search(r"[a-zA-Z]", clean_expr):
                return "Error: Only numeric expressions are allowed"

            result = eval(clean_expr)
            return f"The result of '{expression}' is {result}"
        except ZeroDivisionError:
            return "Error: Division by zero is not allowed"
        except Exception as e:
            return f"Error evaluating expression: {str(e)}"

    async def _arun(self, expression: str) -> str:
        """Asynchronous version"""
        try:
            await asyncio.sleep(0.02)  # Slightly longer for complex calculations

            clean_expr = re.sub(r"[^0-9+\-*/().\s]", "", expression)

            if re.search(r"[a-zA-Z]", clean_expr):
                return "[ASYNC] Error: Only numeric expressions are allowed"

            result = eval(clean_expr)
            return f"[ASYNC] The result of '{expression}' is {result}"
        except ZeroDivisionError:
            return "[ASYNC] Error: Division by zero is not allowed"
        except Exception as e:
            return f"[ASYNC] Error evaluating expression: {str(e)}"


class MathAgent:
    def __init__(self, openai_api_key: Optional[str] = None):
        """
        Initialize the Math Agent with both sync and async support

        Args:
            openai_api_key: Your OpenAI API key. If None, uses OPENAI_API_KEY environment variable
        """
        if openai_api_key:
            self.llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
        else:
            self.llm = ChatOpenAI(temperature=0)

        # Create async-capable tools
        self.tools = [
            AsyncAdditionTool(),
            AsyncSubtractionTool(),
            AsyncMultiplicationTool(),
            AsyncDivisionTool(),
            AsyncCalculatorTool(),
        ]

        # Initialize the agent
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True,
        )

    def calculate(self, query: str) -> str:
        """
        Process a mathematical query synchronously

        Args:
            query: Natural language query about math operations

        Returns:
            String result of the calculation
        """
        try:
            result = self.agent.run(query)
            return result
        except Exception as e:
            return f"Error processing query: {str(e)}"

    async def calculate_async(self, query: str) -> str:
        """
        Process a mathematical query asynchronously

        Args:
            query: Natural language query about math operations

        Returns:
            String result of the calculation
        """
        try:
            # For async agent execution, we'd need to use arun if available
            # Currently, we'll simulate async behavior
            await asyncio.sleep(0.01)
            result = self.agent.run(query)  # LangChain agents don't all support arun yet
            return result
        except Exception as e:
            return f"Error processing async query: {str(e)}"

    async def batch_calculate(self, queries: List[str]) -> List[str]:
        """
        Process multiple queries concurrently using async

        Args:
            queries: List of mathematical queries

        Returns:
            List of results corresponding to each query
        """
        tasks = [self.calculate_async(query) for query in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions in the results
        formatted_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                formatted_results.append(f"Error in query '{queries[i]}': {str(result)}")
            else:
                formatted_results.append(result)

        return formatted_results


# Simplified example for direct tool usage (bypassing agent for testing)
async def test_tools_directly():
    """Test the async tools directly without the agent"""
    print("Testing Tools Directly (Async)")
    print("=" * 40)

    tools = [
        AsyncAdditionTool(),
        AsyncSubtractionTool(),
        AsyncMultiplicationTool(),
        AsyncDivisionTool(),
        AsyncCalculatorTool(),
    ]

    test_cases = [
        ("10 + 5", "addition"),
        ("20 - 8", "subtraction"),
        ("6 * 7", "multiplication"),
        ("36 / 6", "division"),
        ("2 + 3 * 4", "calculator"),
    ]

    for expression, tool_type in test_cases:
        tool = next((t for t in tools if t.name == tool_type), None)
        if tool:
            print(f"\nTesting {tool_type}: {expression}")

            # Test sync version
            sync_result = tool._run(expression)
            print(f"Sync: {sync_result}")

            # Test async version
            async_result = await tool._arun(expression)
            print(f"Async: {async_result}")


# Example usage demonstrating both sync and async capabilities
async def main():
    """Demonstrate both synchronous and asynchronous usage"""

    # print("Mathematical Operations Agent - Sync & Async Demo")
    # print("=" * 50)

    # # # First test tools directly
    # await test_tools_directly()

    # print("\n" + "=" * 50)
    # print("Testing with Agent (requires OpenAI API key)")

    # Uncomment and add your API key to test with the full agent
    math_agent = MathAgent()

    # # Synchronous examples
    print("\n1. Synchronous Operations:")
    # sync_queries = ["What is 15 + 27?", "Calculate 100 - 35", "Multiply 8 by 9"]
    sync_queries = input("Provide your query: ")
    sync_queries = [sync_queries]

    for query in sync_queries:
        print(f"Query: {query}")
        result = math_agent.calculate(query)
        print(f"Result: {result}\n")

    # # Batch processing (concurrent)
    # print("2. Batch Processing (Concurrent):")
    # batch_queries = ["What is 5 + 5?", "Calculate 20 - 8", "Multiply 6 by 7", "Divide 36 by 6"]

    # print("Processing all queries concurrently...")
    # results = await math_agent.batch_calculate(batch_queries)

    # for query, result in zip(batch_queries, results):
    #     print(f"Query: {query}")
    #     print(f"Result: {result}\n")


# Traditional synchronous usage
def sync_example():
    """Example of traditional synchronous usage without OpenAI"""
    print("Testing Individual Tools (Sync)")
    print("=" * 35)

    tools = [
        AsyncAdditionTool(),
        AsyncSubtractionTool(),
        AsyncMultiplicationTool(),
        AsyncDivisionTool(),
        AsyncCalculatorTool(),
    ]

    test_cases = [
        ("25 + 75", "addition"),
        ("200 - 50", "subtraction"),
        ("12 * 12", "multiplication"),
        ("144 / 12", "division"),
        ("(10 + 5) * 2", "calculator"),
    ]

    for expression, tool_type in test_cases:
        tool = next((t for t in tools if t.name == tool_type), None)
        if tool:
            result = tool._run(expression)
            print(f"{expression} -> {result}")


if __name__ == "__main__":
    print("Mathematical Operations Agent")
    print("Choose mode:")
    print("1. Test tools directly (sync)")
    print("2. Test tools directly (async)")
    print("3. Test with full agent (requires OpenAI API key)")

    mode = input("Enter mode (1/2/3): ").strip()

    if mode == "1":
        sync_example()
    elif mode == "2":
        asyncio.run(test_tools_directly())
    elif mode == "3":
        print("To test with full agent, uncomment the agent code in main() and add your OpenAI API key")
        asyncio.run(main())
    else:
        print("Running sync example by default...")
        sync_example()
