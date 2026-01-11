import logging
from typing import Dict, List, Any, Optional
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.tools import BaseTool
from langchain_ollama import ChatOllama

logger = logging.getLogger(__name__)


class CodeAgent:
    """AI Agent for code-related tasks using LangChain"""

    def __init__(self, llm=None):
        self.llm = llm or ChatOllama(model="codellama:7b", temperature=0.3)

        self.tools = [
            self._create_code_generation_tool(),
            self._create_code_analysis_tool(),
            self._create_code_debugging_tool(),
            self._create_code_optimization_tool(),
            self._create_documentation_tool(),
        ]

        # Create agent
        self.agent = self._create_agent()

        logger.info("Code Agent initialized")

    def _create_agent(self):
        """Create LangChain agent"""
        prompt = PromptTemplate.from_template(
            """
You are an expert AI coding assistant. You have access to various tools to help with coding tasks.

Available tools:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original question

Begin!

Question: {input}
Thought:{agent_scratchpad}
"""
        )

        agent = create_react_agent(llm=self.llm, tools=self.tools, prompt=prompt)

        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=5,
            early_stopping_method="generate",
        )

    def _create_code_generation_tool(self) -> BaseTool:
        """Create tool for code generation"""
        from langchain.tools import tool

        @tool
        def generate_code_tool(prompt: str, language: str = "python") -> str:
            """Generate code based on a prompt and language."""
            try:
                system_message = f"""You are an expert {language} programmer. Generate clean, efficient code.

Requirements:
1. Follow {language} best practices
2. Include error handling
3. Add comments
4. Make it production-ready
5. Provide usage examples
"""

                response = self.llm.invoke(
                    [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt},
                    ]
                )

                return f"Generated {language} code:\n```{language}\n{response.content}\n```"

            except Exception as e:
                return f"Error generating code: {str(e)}"

        return generate_code_tool

    def _create_code_analysis_tool(self) -> BaseTool:
        """Create tool for code analysis"""
        from langchain.tools import tool

        @tool
        def analyze_code_tool(code: str, language: str = "auto") -> str:
            """Analyze code for quality, complexity, and issues."""
            try:
                system_message = """You are an expert code analyst. Analyze the provided code.

Provide analysis in these areas:
1. Code quality and best practices
2. Complexity and maintainability
3. Potential bugs and issues
4. Security concerns
5. Performance considerations
6. Recommendations for improvement
"""

                response = self.llm.invoke(
                    [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": f"Code to analyze:\n{code}"},
                    ]
                )

                return f"Code Analysis:\n{response.content}"

            except Exception as e:
                return f"Error analyzing code: {str(e)}"

        return analyze_code_tool

    def _create_code_debugging_tool(self) -> BaseTool:
        """Create tool for debugging"""
        from langchain.tools import tool

        @tool
        def debug_code_tool(code: str, error_message: str = "") -> str:
            """Debug code and fix issues."""
            try:
                system_message = """You are an expert debugger. Find and fix issues in code.

Instructions:
1. Identify syntax errors
2. Find logical errors
3. Check for runtime issues
4. Fix the problems
5. Explain what was wrong and how you fixed it
"""

                prompt = f"Code to debug:\n{code}"
                if error_message:
                    prompt += f"\n\nError message: {error_message}"

                response = self.llm.invoke(
                    [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt},
                    ]
                )

                return f"Debug Results:\n{response.content}"

            except Exception as e:
                return f"Error debugging code: {str(e)}"

        return debug_code_tool

    def _create_code_optimization_tool(self) -> BaseTool:
        """Create tool for code optimization"""
        from langchain.tools import tool

        @tool
        def optimize_code_tool(
            code: str, optimization_type: str = "performance"
        ) -> str:
            """Optimize code for performance, readability, or memory."""
            try:
                focus_map = {
                    "performance": "execution speed and efficiency",
                    "readability": "code clarity and maintainability",
                    "memory": "memory usage and footprint",
                }

                focus = focus_map.get(optimization_type, "performance")

                system_message = f"""You are an expert code optimizer. Optimize code for {focus}.

Provide:
1. Optimized code
2. Explanation of optimizations
3. Performance improvements
4. Trade-offs if any
"""

                response = self.llm.invoke(
                    [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": f"Code to optimize:\n{code}"},
                    ]
                )

                return (
                    f"Optimization Results ({optimization_type}):\n{response.content}"
                )

            except Exception as e:
                return f"Error optimizing code: {str(e)}"

        return optimize_code_tool

    def _create_documentation_tool(self) -> BaseTool:
        """Create tool for documentation"""
        from langchain.tools import tool

        @tool
        def document_code_tool(code: str) -> str:
            """Add documentation to code."""
            try:
                system_message = """You are an expert technical writer. Add comprehensive documentation.

Include:
1. Function/class docstrings
2. Parameter descriptions
3. Return value explanations
4. Usage examples
5. Error handling notes
"""

                response = self.llm.invoke(
                    [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": f"Code to document:\n{code}"},
                    ]
                )

                return f"Documentation:\n{response.content}"

            except Exception as e:
                return f"Error documenting code: {str(e)}"

        return document_code_tool

    def run(self, task: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run the agent on a task"""
        try:
            logger.info(f"Agent running task: {task[:100]}...")

            input_text = task
            if context:
                input_text += f"\n\nContext: {str(context)}"

            result = self.agent.invoke(
                {
                    "input": input_text,
                    "tool_names": ", ".join([tool.name for tool in self.tools]),
                }
            )

            return {
                "success": True,
                "result": result["output"],
                "intermediate_steps": len(result.get("intermediate_steps", [])),
                "execution_time": result.get("execution_time", 0),
            }

        except Exception as e:
            logger.error(f"Agent error: {str(e)}")
            return {"success": False, "error": str(e), "result": ""}
