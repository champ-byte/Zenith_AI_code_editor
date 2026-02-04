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


class EditState(TypedDict):
    """State for Multi-File Edit Workflow"""
    messages: Annotated[List[Any], add_messages]
    task: str
    files: List[str]  # List of relevant files
    plan: str
    edits: List[Dict[str, str]] # List of {file_path, new_content, original_content, diff}
    analysis: str

class MultiFileEditWorkflow:
    """Orchestrates multi-file edits with AST analysis and planning"""

    def __init__(self, llm_service, rag_service=None, tree_sitter_service=None):
        self.llm_service = llm_service
        self.rag_service = rag_service
        self.tree_sitter_service = tree_sitter_service
        self.workflow = self._create_workflow()

    def _create_workflow(self):
        workflow = StateGraph(EditState)

        workflow.add_node("analyze_context", self._analyze_context)
        workflow.add_node("plan_changes", self._plan_changes)
        workflow.add_node("generate_content", self._generate_content)

        workflow.set_entry_point("analyze_context")
        workflow.add_edge("analyze_context", "plan_changes")
        workflow.add_edge("plan_changes", "generate_content")
        workflow.add_edge("generate_content", END)

        return workflow.compile()

    def _analyze_context(self, state: EditState) -> EditState:
        """Analyze files using RAG for discovery and AST/Regex for structure"""
        logger.info("Analyzing context for multi-file edit")
        
        # 1. RAG Discovery (Hybrid Strategy)
        discovered_files = set(state.get("files", []))
        rag_context = ""
        
        if self.rag_service:
            try:
                # Infer relevant files from the task even if not explicitly mentioned
                logger.info(f"Querying RAG for task: {state['task']}")
                search_result = self.rag_service.query_with_context(state['task'])
                rag_context = search_result.get("context", "")
                
                # Extract filenames from RAG content (File: ...) logic or sources
                for source in search_result.get("sources", []):
                    # Only add if it's a file in the project (rag sources are usually relative paths or filenames)
                    if source and source != "unknown":
                         # Heuristic: if it looks like a file path, add it
                         # Sources from rag_service are usually filenames
                         import os
                         if os.path.exists(source) or (self.rag_service.current_indexed_path and os.path.exists(os.path.join(self.rag_service.current_indexed_path, source))):
                             if self.rag_service.current_indexed_path and not os.path.isabs(source):
                                 full_path = os.path.join(self.rag_service.current_indexed_path, source)
                                 discovered_files.add(full_path)
                             else:
                                 discovered_files.add(source)
                                 
                logger.info(f"RAG discovered files: {discovered_files}")
                
            except Exception as e:
                logger.error(f"RAG discovery failed: {e}")

        # Update state with discovered files so subsequent steps see them
        state["files"] = list(discovered_files)

        # 2. Structure Analysis (AST/Content/Tree-sitter)
        analysis_results = []
        if rag_context:
            analysis_results.append("--- RAG Context (Relevant Snippets) ---")
            analysis_results.append(rag_context + "\n")

        import os
        
        for file_path in state.get("files", []):
            if not os.path.exists(file_path):
                continue
                
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                analysis_results.append(f"--- File: {os.path.basename(file_path)} ---")
                
                # Use Tree-sitter if available
                if self.tree_sitter_service:
                    structure = self.tree_sitter_service.parse_file(file_path, content)
                    if "error" in structure:
                         analysis_results.append(f"Structure: (Error: {structure['error']})")
                    else:
                         analysis_results.append(f"Structure: Language={structure.get('language')}")
                         analysis_results.append(f"Classes={structure.get('classes', [])}")
                         analysis_results.append(f"Functions={structure.get('functions', [])}")
                else:
                    # Fallback (Simple read) or old AST logic if desired, but we are replacing it.
                    analysis_results.append("Structure: (Tree-sitter not available)")
                    
            except Exception as e:
                logger.error(f"Error analyzing file {file_path}: {e}")
        
        state["analysis"] = "\n".join(analysis_results)
        return state

    def _plan_changes(self, state: EditState) -> EditState:
        """Plan which files to edit and how"""
        logger.info("Planning changes")
        
        prompt = f"""
        Plan the following coding task across multiple files.
        You can MODIFY existing files or CREATE NEW files if needed.
        
        Task: {state['task']}
        Analysis of provided files:
        {state['analysis']}
        
        Files currently available: {state['files']}
        
        Provide a plan that:
        1. Identifies which files need modification.
        2. Identifies any NEW files to be created.
        3. Describes the specific changes/content for each file.
        4. Ensures consistency across files.
        """
        
        messages = [
            SystemMessage(content="You are a senior software architect planning a multi-file refactor."),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm_service.creative_llm.invoke(messages)
        state["messages"].append(response)
        state["plan"] = response.content
        return state



    def _generate_content(self, state: EditState) -> EditState:
        """Generate the new content for each file"""
        logger.info("Generating content")
        
        plan = state["plan"]
        edits = []
        
        # Ask LLM for a structured list of files to edit/create based on the plan
        candidates_prompt = f"""
        Based on this plan, list ALL files that need to be created or modified. 
        Return strictly a valid JSON list of strings.
        Example: ["app.py", "utils.py", "new_service.py"]
        
        Plan:
        {plan}
        
        Existing files: {state['files']}
        """
        
        candidates_msg = [
             SystemMessage(content="You are a helper. Return strictly a JSON list of strings. Do not use markdown blocks."),
             HumanMessage(content=candidates_prompt)
        ]
        
        try:
             import json
             import re
             import ast
             
             resp = self.llm_service.precise_llm.invoke(candidates_msg)
             content = resp.content.strip()
             
             # Robust Parsing Strategy
             target_files = []
             match = re.search(r"\[.*\]", content, re.DOTALL)
             if match:
                 try:
                     target_files = json.loads(match.group(0))
                 except:
                     try: target_files = ast.literal_eval(match.group(0))
                     except: pass
             
             if not target_files:
                  if "," in content: target_files = [f.strip().strip('"') for f in content.split(",")]
                  else: target_files = state['files']
                  
             if not isinstance(target_files, list): target_files = state['files']
                  
        except Exception as e:
             logger.error(f"Failed to parse target files: {e}")
             target_files = state['files']

        import os
        
        for file_path in target_files:
             # Check if file exists to decide prompt nuance
             file_exists = os.path.exists(file_path)
             original_content = ""
             
             if file_exists:
                 with open(file_path, "r", encoding="utf-8") as f:
                     original_content = f.read()
                 
                 # TOKEN SAVER: Use SEARCH/REPLACE for existing files
                 prompt = f"""
                 Based on the plan, apply changes to: {file_path}
                 
                 Plan: {plan}
                 
                 Use SEARCH/REPLACE blocks to modify the code.
                 Format:
                 <<<<<<< SEARCH
                 [exact code to replace]
                 =======
                 [new code]
                 >>>>>>> REPLACE
                 
                 Rules:
                 1. SEARCH block must exactly match existing code (include indentation).
                 2. Include multiple blocks if needed.
                 3. Do NOT rewrite the whole file.
                 
                 Original File Content:
                 ```
                 {original_content}
                 ```
                 """
                 
                 prompt_type = "PATCH"
             else:
                 prompt = f"""
                 Create new file: {file_path}
                 Plan: {plan}
                 Output the FULL content of the new file.
                 """
                 prompt_type = "CREATE"
             
             messages = [
                SystemMessage(content="You are an expert coder. Output clean code or patch blocks."),
                HumanMessage(content=prompt)
             ]
             
             response = self.llm_service.precise_llm.invoke(messages)
             generated_text = response.content
             

            
             if prompt_type == "PATCH":
                 # Extract hunks manually without applying them
                 # Parse blocks: <<<<<<< SEARCH ... ======= ... >>>>>>> REPLACE
                 import re
                 pattern = r"<{7}\s*SEARCH\s*\n(.*?)\n={7}\s*\n(.*?)\n>{7}\s*REPLACE"
                 matches = list(re.finditer(pattern, generated_text, re.DOTALL))
                 
                 hunks = []
                 for match in matches:
                     hunks.append({
                         "search": match.group(1),
                         "replace": match.group(2)
                     })
                 
                 # Optimization: Don't calculate diff here, just send hunks.
                 edits.append({
                    "file_path": file_path,
                    "hunks": hunks,
                    "is_new": False
                 })
                 
             else:
                 # New file
                 new_content = generated_text
                 match = re.search(r"```(?:\w+)?\n(.*?)\n```", new_content, re.DOTALL)
                 if match: new_content = match.group(1)
            
                 edits.append({
                    "file_path": file_path,
                    "new_content": new_content,
                    "hunks": [], # No hunks for new file
                    "is_new": True
                 })
        
        state["edits"] = edits
        return state

    def run(self, task: str, files: List[str]) -> Dict[str, Any]:
        """Run the multi-file edit workflow"""
        try:
            logger.info(f"Starting multi-file edit for: {task[:50]}...")
            
            initial_state = EditState(
                messages=[],
                task=task,
                files=files,
                plan="",
                edits=[],
                analysis=""
            )
            
            final_state = self.workflow.invoke(initial_state)
            
            return {
                "success": True,
                "analysis": final_state.get("analysis", ""),
                "plan": final_state.get("plan", ""),
                "edits": final_state.get("edits", []),
                "summary": f"Generated {len(final_state.get('edits', []))} edits"
            }
            
        except Exception as e:
            logger.error(f"Multi-file workflow error: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e),
                "edits": []
            }
