from typing import Dict, List, Any, TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, SystemMessage
import logging

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """State for LangGraph workflow"""

    messages: Annotated[List[Any], add_messages]
    task: str
    language: str
    code: str
    context: str
    analysis: str
    issues: List[str]
    suggestions: List[str]
    final_output: str


class CodeWorkflow:
    """Orchestrates multiple agents for complex coding tasks"""

    def __init__(self, llm_service):
        self.llm_service = llm_service
        self.workflow = self._create_workflow()

    def _create_workflow(self):
        """Create LangGraph workflow for code tasks"""
        workflow = StateGraph(AgentState)

        workflow.add_node("analyze_requirements", self._analyze_requirements)
        workflow.add_node("design_architecture", self._design_architecture)
        workflow.add_node("generate_code", self._generate_code)
        workflow.add_node("review_code", self._review_code)
        workflow.add_node("test_code", self._test_code)
        workflow.add_node("optimize_code", self._optimize_code)
        workflow.add_node("document_code", self._document_code)
        workflow.add_node("generate_final_output", self._generate_final_output)

        workflow.set_entry_point("analyze_requirements")
        workflow.add_edge("analyze_requirements", "design_architecture")
        workflow.add_edge("design_architecture", "generate_code")
        workflow.add_edge("generate_code", "review_code")
        workflow.add_edge("review_code", "test_code")
        workflow.add_conditional_edges(
            "test_code",
            self._should_optimize,
            {"optimize": "optimize_code", "document": "document_code"},
        )
        workflow.add_edge("optimize_code", "document_code")
        workflow.add_edge("document_code", "generate_final_output")
        workflow.add_edge("generate_final_output", END)

        return workflow.compile()

    def _analyze_requirements(self, state: AgentState) -> AgentState:
        """Analyze requirements and constraints"""
        logger.info("Analyzing requirements")

        prompt = f"""
        Analyze the following coding task:
        
        Task: {state['task']}
        Language: {state['language']}
        Context: {state['context']}
        
        Provide:
        1. Key requirements
        2. Constraints and edge cases
        3. Input/output specifications
        4. Performance considerations
        """

        messages = [
            SystemMessage(content="You are a requirements analyst."),
            HumanMessage(content=prompt),
        ]

        response = self.llm_service.creative_llm.invoke(messages)

        state["messages"].append(response)
        state["analysis"] = response.content

        return state

    def _design_architecture(self, state: AgentState) -> AgentState:
        """Design solution architecture"""
        logger.info("Designing architecture")

        prompt = f"""
        Based on the analysis, design the architecture:
        
        Analysis: {state['analysis']}
        Task: {state['task']}
        Language: {state['language']}
        
        Design:
        1. Overall architecture
        2. Modules and components
        3. Data structures
        4. Algorithms
        5. Error handling strategy
        """

        messages = [
            SystemMessage(content="You are a software architect."),
            HumanMessage(content=prompt),
        ]

        response = self.llm_service.creative_llm.invoke(messages)

        state["messages"].append(response)
        state["analysis"] += f"\n\nArchitecture Design:\n{response.content}"

        return state

    def _generate_code(self, state: AgentState) -> AgentState:
        """Generate code based on design"""
        logger.info("Generating code")

        prompt = f"""
        Generate code based on the design:
        
        Task: {state['task']}
        Language: {state['language']}
        Design: {state['analysis']}
        
        Requirements:
        1. Complete, working code
        2. Follow language best practices
        3. Include error handling
        4. Add comments
        5. Make it production-ready
        """

        messages = [
            SystemMessage(content=f"You are an expert {state['language']} programmer."),
            HumanMessage(content=prompt),
        ]

        response = self.llm_service.precise_llm.invoke(messages)

        state["messages"].append(response)
        state["code"] = response.content

        return state

    def _review_code(self, state: AgentState) -> AgentState:
        """Review generated code"""
        logger.info("Reviewing code")

        prompt = f"""
        Review the generated code:
        
        Code:\n```{state['language']}\n{state['code']}\n```
        Task: {state['task']}
        
        Review for:
        1. Bugs and errors
        2. Code style and conventions
        3. Performance issues
        4. Security vulnerabilities
        5. Edge cases
        6. Suggestions for improvement
        """

        messages = [
            SystemMessage(content="You are a senior code reviewer."),
            HumanMessage(content=prompt),
        ]

        response = self.llm_service.analytical_llm.invoke(messages)

        state["messages"].append(response)

        content = response.content
        issues = []
        for line in content.split("\n"):
            if any(
                word in line.lower()
                for word in ["bug", "error", "issue", "problem", "warning"]
            ):
                issues.append(line.strip())

        state["issues"] = issues[:5]  

        return state

    def _test_code(self, state: AgentState) -> AgentState:
        """Write tests for the code"""
        logger.info("Writing tests")

        prompt = f"""
        Write tests for the code:
        
        Code:\n```{state['language']}\n{state['code']}\n```
        Issues found: {state['issues']}
        
        Write comprehensive tests covering:
        1. Unit tests for all functions
        2. Edge cases
        3. Error cases
        4. Integration tests if needed
        """

        messages = [
            SystemMessage(content="You are a testing expert."),
            HumanMessage(content=prompt),
        ]

        response = self.llm_service.precise_llm.invoke(messages)

        state["messages"].append(response)

        if state["issues"] and len(state["issues"]) > 2:
            state["suggestions"].append("Code needs optimization before finalizing")

        return state

    def _should_optimize(self, state: AgentState) -> str:
        """Decide whether to optimize code"""
        if state["issues"] and len(state["issues"]) > 2:
            return "optimize"
        return "document"

    def _optimize_code(self, state: AgentState) -> AgentState:
        """Optimize the code"""
        logger.info("Optimizing code")

        prompt = f"""
        Optimize the code based on review findings:
        
        Code:\n```{state['language']}\n{state['code']}\n```
        Issues: {state['issues']}
        
        Optimize for:
        1. Fixing all identified issues
        2. Performance improvements
        3. Better readability
        4. Enhanced maintainability
        """

        messages = [
            SystemMessage(content="You are a code optimization expert."),
            HumanMessage(content=prompt),
        ]

        response = self.llm_service.precise_llm.invoke(messages)

        state["messages"].append(response)

        state["code"] = response.content
        state["suggestions"].append("Code optimized successfully")

        return state

    def _document_code(self, state: AgentState) -> AgentState:
        """Document the code"""
        logger.info("Documenting code")

        prompt = f"""
        Add comprehensive documentation to the code:
        
        Code:\n```{state['language']}\n{state['code']}\n```
        
        Add:
        1. Function/class docstrings
        2. Parameter descriptions
        3. Return value explanations
        4. Usage examples
        5. Notes on optimization decisions
        """

        messages = [
            SystemMessage(content="You are a technical documentation expert."),
            HumanMessage(content=prompt),
        ]

        response = self.llm_service.creative_llm.invoke(messages)

        state["messages"].append(response)

        state["code"] = response.content

        return state

    def _generate_final_output(self, state: AgentState) -> AgentState:
        """Generate final output"""
        logger.info("Generating final output")

        prompt = f"""
        Create final comprehensive output for the coding task:
        
        Task: {state['task']}
        Language: {state['language']}
        Final Code:\n```{state['language']}\n{state['code']}\n```
        Process Summary: {len(state['messages'])} steps completed
        
        Provide:
        1. Final code with documentation
        2. Summary of what was accomplished
        3. Key decisions made
        4. Testing recommendations
        5. Future improvements
        """

        messages = [
            SystemMessage(content="You are a final output generator."),
            HumanMessage(content=prompt),
        ]

        response = self.llm_service.creative_llm.invoke(messages)

        state["final_output"] = response.content
        state["messages"].append(response)

        return state

    def run(
        self, task: str, language: str = "python", context: str = ""
    ) -> Dict[str, Any]:
        """Run the complete workflow"""
        try:
            logger.info(f"Starting workflow for task: {task[:50]}...")

            initial_state = AgentState(
                messages=[],
                task=task,
                language=language,
                context=context,
                code="",
                analysis="",
                issues=[],
                suggestions=[],
                final_output="",
            )

            final_state = self.workflow.invoke(initial_state)

            return {
                "success": True,
                "final_code": final_state.get("code", ""),
                "final_output": final_state.get("final_output", ""),
                "analysis": final_state.get("analysis", ""),
                "issues": final_state.get("issues", []),
                "suggestions": final_state.get("suggestions", []),
                "steps": len(final_state.get("messages", [])),
                "execution_summary": f"Completed {len(final_state.get('messages', []))} steps",
            }

        except Exception as e:
            logger.error(f"Workflow error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "final_code": "",
                "final_output": "",
            }
