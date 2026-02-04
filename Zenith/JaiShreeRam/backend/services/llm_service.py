import os
import json
import logging
from typing import Dict, List, Any, Optional
import requests
# from langchain_community.llms import Ollama
# from langchain_community.chat_models import ChatOllama
# from langchain.schema import HumanMessage, SystemMessage, AIMessage
# from langchain.prompts import (
#     ChatPromptTemplate,
#     HumanMessagePromptTemplate,
#     SystemMessagePromptTemplate,
# )
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
import google.generativeai as genai

logger = logging.getLogger(__name__)

from dotenv import load_dotenv

load_dotenv()

class GeminiService:
    def __init__(self):
        """Initialize Gemini service"""
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        
        # Initialize attributes
        self.llm = None
        self.creative_llm = None
        self.precise_llm = None
        self.analytical_llm = None
        self.model = f"{self.model_name} (unconfigured)"
        
        if not self.api_key:
            logger.error("GEMINI_API_KEY is not set")
            logger.warning("Generative AI features will be unavailable.")
        else:
            # Set GOOGLE_API_KEY for LangChain components that might look for it
            os.environ["GOOGLE_API_KEY"] = self.api_key
            self._initialize_llms()

    def _initialize_llms(self):
        """Initialize all variations of the LLM"""
        try:
            if self.api_key:
                logger.info(f"Configuring Gemini with key: {self.api_key[:5]}...{self.api_key[-5:]} (Len: {len(self.api_key)})")
            genai.configure(api_key=self.api_key)
            # Use specific model names and ensure api_key is passed explicitly
            self.llm = ChatGoogleGenerativeAI(model=self.model_name, google_api_key=self.api_key, temperature=0.7)
            self.model = self.model_name
            
            # Initialize specialized LLMs
            self.creative_llm = ChatGoogleGenerativeAI(model=self.model_name, google_api_key=self.api_key, temperature=0.9)
            self.precise_llm = ChatGoogleGenerativeAI(model=self.model_name, google_api_key=self.api_key, temperature=0.1)
            self.analytical_llm = ChatGoogleGenerativeAI(model=self.model_name, google_api_key=self.api_key, temperature=0.3)
            
            logger.info(f"Gemini Service initialized with model: {self.model}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini components: {str(e)}")
            self.llm = None
            self.creative_llm = None
            self.precise_llm = None
            self.analytical_llm = None
            self.model = f"{self.model_name} (error)"

    def _ensure_configured(self):
        if not self.llm:
            # Try to reload key in case it was added later
            self.api_key = os.getenv("GEMINI_API_KEY")
            if self.api_key:
                os.environ["GOOGLE_API_KEY"] = self.api_key
                self._initialize_llms()
                if not self.llm:
                    raise ValueError("Gemini initialization failed even with API key. Check logs.")
            else:
                raise ValueError("GEMINI_API_KEY is missing. Please add it to .env file and RESTART the server.")

    def generate_code(self, prompt: str, language: str = "python", context: str = "") -> Dict[str, Any]:
        """Generate code using Gemini"""
        try:
            self._ensure_configured()
            system_prompt = f"""You are an expert {language} programmer. Generate clean, efficient, and well-documented code.

Requirements:
1. Follow {language} best practices and conventions
2. Include proper error handling
3. Add meaningful comments
4. Make it production-ready
5. Include usage examples if applicable

Context: {context}
"""
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Generate {language} code for: {prompt}"),
            ]
            
            response = self.llm.invoke(messages)
            
            return {
                "success": True,
                "code": response.content,
                "explanation": f"Generated using {self.model}",
                "language": language,
            }

        except Exception as e:
            logger.error(f"Error generating code: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "code": f"// Error generating code: {str(e)}",
                "explanation": "Failed to generate code",
            }

    def explain_code(self, code: str, language: str = "auto", detail_level: str = "comprehensive") -> str:
        try:
            self._ensure_configured()
            if language == "auto":
                language = self._detect_language(code)
                
            system_prompt = f"""You are an expert code explainer. Explain the {language} code in {detail_level} detail.

Explanation should include:
1. What the code does
2. How it works (step by step)
3. Key algorithms and data structures
4. Time and space complexity analysis
5. Use cases and applications
6. Potential improvements

Make the explanation clear and accessible.
"""
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Explain this {language} code:\n```{language}\n{code}\n```"),
            ]
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"Error explaining code: {str(e)}")
            return f"Error explaining code: {str(e)}"

    def debug_code(self, code: str, language: str = "auto", error_message: str = "") -> Dict[str, Any]:
        try:
            self._ensure_configured()
            if language == "auto":
                language = self._detect_language(code)
                
            system_prompt = f"""You are an expert debugger for {language}. Find and fix issues in the code.

Instructions:
1. Analyze the code for syntax errors and bugs
2. Check for runtime issues and security vulnerabilities
3. Suggest specific fixes
4. Provide corrected code

Error message: {error_message}
"""
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Debug this {language} code:\n```{language}\n{code}\n```"),
            ]
            response = self.llm.invoke(messages)
            content = response.content
            corrected_code = self._extract_code_blocks(content)
            
            return {
                "success": True,
                "debugged_code": corrected_code[0] if corrected_code else code,
                "explanation": content,
                "issues_found": self._extract_issues(content),
                "fixes_applied": ["Fixed issues using AI analysis"],
            }
        except Exception as e:
            logger.error(f"Error debugging code: {str(e)}")
            return {
                "success": False,
                "debugged_code": code,
                "explanation": f"Error: {str(e)}",
                "issues_found": [],
                "fixes_applied": [],
            }

    def optimize_code(self, code: str, language: str = "auto", optimization_type: str = "performance") -> Dict[str, Any]:
        try:
            self._ensure_configured()
            if language == "auto":
                language = self._detect_language(code)
                
            system_prompt = f"""You are an expert {language} optimizer. Optimize the code for {optimization_type}.

Provide both the optimized code and explanation of changes.
"""
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Optimize this {language} code:\n```{language}\n{code}\n```"),
            ]
            response = self.llm.invoke(messages)
            content = response.content
            optimized_code = self._extract_code_blocks(content)
            
            return {
                "success": True,
                "optimized_code": optimized_code[0] if optimized_code else code,
                "explanation": content,
                "improvements": [f"Optimized for {optimization_type}"],
                "before_metrics": {"lines": len(code.split("\n"))},
                "after_metrics": {"lines": len(optimized_code[0].split("\n")) if optimized_code else len(code.split("\n"))},
            }
        except Exception as e:
            logger.error(f"Error optimizing code: {str(e)}")
            return {
                "success": False,
                "optimized_code": code,
                "explanation": f"Error: {str(e)}",
                "improvements": [],
                "before_metrics": {},
                "after_metrics": {},
            }

    def chat(self, message: str, history: List[Dict] = None, context: Dict = None) -> Dict[str, Any]:
        try:
            self._ensure_configured()
            messages = []
            system_content = """You are an AI coding assistant. You help with code generation, explanation, debugging, optimization, and answering questions.
Be helpful, accurate, and provide code examples when needed.
Format code blocks with proper syntax highlighting.

GIT CAPABILITIES:
You are an agentic assistant that can perform Git operations. If the user asks you to push, pull, commit, or manage branches, you MUST include a `<git_action>` tag at the very end of your response with the appropriate JSON command.

Supported commands (JSON):
- { "action": "status" }
- { "action": "create-branch", "data": { "name": "branch-name" } }
- { "action": "checkout", "data": { "name": "branch-name" } }
- { "action": "commit", "data": "Commit message" }
- { "action": "push" }
- { "action": "pull" }
- { "action": "sync" }

Example Use Case:
User: "Push my current changes to the 'main' branch."
Response: "I will push your current changes to the main branch for you locally. <git_action>{ \"action\": \"push\" }</git_action>"

Always confirm the action in your text response.
"""
            if context:
                # Special handling for RAG components
                file_tree = context.get("file_tree")
                rag_context = context.get("rag_context")
                
                if file_tree:
                    system_content += f"\nPROJECT STRUCTURE (Use this to understand file organization):\n{file_tree}\n"
                    
                if rag_context:
                    system_content += f"\nRETRIEVED CODE CONTEXT (Use this to understand implementation details):\n{rag_context}\n"
                
                # Add other context items
                other_context = {k:v for k,v in context.items() if k not in ["file_tree", "rag_context"]}
                if other_context:
                    system_content += f"\nADDITIONAL CONTEXT:\n{json.dumps(other_context, indent=2)}"
            
            messages.append(SystemMessage(content=system_content))
            
            if history:
                 for msg in history[-6:]:
                    if msg["role"] == "user":
                        messages.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        messages.append(AIMessage(content=msg["content"]))
            
            messages.append(HumanMessage(content=message))
            
            response = self.llm.invoke(messages)
            
            new_history = (history or []) + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": response.content},
            ]
            
            return {
                "success": True,
                "response": response.content,
                "history": new_history[-10:],
                "model": self.model,
            }
        except Exception as e:
            logger.error(f"Chat error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "response": f"Error: {str(e)}",
                "history": history or [],
            }

    def reframe_query(self, message: str, history: List[Dict] = None) -> str:
        """Reframe user query into a precise search query based on history"""
        try:
            self._ensure_configured()
            
            system_prompt = """You are an expert search query generator for a code RAG system.
Your task is to rewrite the user's latest message into a precise, standalone search query.

Guidelines:
1. Use conversation history to resolve pronouns (it, that, the file).
2. If the user mentions specific filenames (e.g., app.py), INCLUDE them in the query.
3. If the user asks "how does it work", focus on "implementation details", "logic", or "flow".
4. If the message is already clear (e.g., "Search for login"), return it as is.
5. Return ONLY the search query.

Examples:
- History: [User: Show me app.py] -> User: "How does it handle errors?"
  Output: error handling implementation in app.py

- History: [User: What is RAG?] -> User: "Where is the code for that?"
  Output: RAG implementation code location

- User: "Explain the main.js file"
  Output: main.js file explanation code logic
"""
            
            messages = [SystemMessage(content=system_prompt)]
            
            if history:
                # meaningful_history = history[-4:] # Keep it short
                for msg in history[-4:]:
                    role = "User" if msg["role"] == "user" else "AI"
                    messages.append(HumanMessage(content=f"{role}: {msg['content']}"))
            
            messages.append(HumanMessage(content=f"User: {message}\nOutput Search Query:"))
            
            response = self.llm.invoke(messages)
            reframed = response.content.strip()
            logger.info(f"Reframed query: '{message}' -> '{reframed}'")
            return reframed
            
        except Exception as e:
            logger.error(f"Error reframing query: {str(e)}")
            return message # Fallback to original message

    def write_tests(self, code: str, language: str = "auto", test_framework: str = "") -> Dict[str, Any]:
        try:
            self._ensure_configured()
            if language == "auto":
                language = self._detect_language(code)
            
            system_prompt = f"""You are an expert in writing tests for {language}."""
            if test_framework:
                system_prompt += f" Use {test_framework}."
                
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Write tests for this {language} code:\n```{language}\n{code}\n```"),
            ]
            response = self.llm.invoke(messages)
            content = response.content
            tests = self._extract_code_blocks(content)
            
            return {
                "success": True,
                "tests": tests[0] if tests else "",
                "test_explanation": content,
                "coverage": 80.0,
                "test_cases": ["Generated tests"],
            }
        except Exception as e:
            logger.error(f"Error writing tests: {str(e)}")
            return {"success": False, "tests": "", "test_explanation": f"Error: {str(e)}", "coverage": 0.0, "test_cases": []}

    def analyze_code(self, code: str, language: str = "auto", analysis_type: str = "comprehensive") -> Dict[str, Any]:
        try:
            self._ensure_configured()
            if language == "auto":
                language = self._detect_language(code)
                
            system_prompt = f"You are an expert code analyst. Perform {analysis_type} analysis of this {language} code."
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Analyze this {language} code:\n```{language}\n{code}\n```"),
            ]
            response = self.llm.invoke(messages)
            content = response.content
            
            return {
                "success": True,
                "analysis": content,
                "complexity": {"cyclomatic": "Unknown", "cognitive": "Unknown"},
                "quality_score": 80.0,
                "issues": self._extract_issues(content),
                "recommendations": ["See analysis"],
            }
        except Exception as e:
            logger.error(f"Error analyzing code: {str(e)}")
            return {"success": False, "analysis": f"Error: {str(e)}", "complexity": {}, "quality_score": 0.0, "issues": [], "recommendations": []}

    def convert_code(self, code: str, source_language: str, target_language: str) -> Dict[str, Any]:
        try:
            self._ensure_configured()
            if source_language == "auto":
                source_language = self._detect_language(code)
                
            system_prompt = f"You are an expert code converter. Convert code from {source_language} to {target_language}."
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Convert this {source_language} code to {target_language}:\n```{source_language}\n{code}\n```"),
            ]
            response = self.llm.invoke(messages)
            content = response.content
            converted_code = self._extract_code_blocks(content)
            
            return {
                "success": True,
                "converted_code": converted_code[0] if converted_code else "",
                "explanation": content,
                "compatibility_notes": [],
            }
        except Exception as e:
            logger.error(f"Error converting code: {str(e)}")
            return {"success": False, "converted_code": "", "explanation": f"Error: {str(e)}", "compatibility_notes": []}

    def document_code(self, code: str, language: str = "auto", documentation_style: str = "comprehensive") -> Dict[str, Any]:
        try:
            self._ensure_configured()
            if language == "auto":
                language = self._detect_language(code)
                
            system_prompt = f"You are an expert technical writer. Add {documentation_style} documentation to this {language} code."
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Add documentation to this {language} code:\n```{language}\n{code}\n```"),
            ]
            response = self.llm.invoke(messages)
            content = response.content
            documented_code = self._extract_code_blocks(content)
            
            return {
                "success": True,
                "documented_code": documented_code[0] if documented_code else code,
                "documentation": content,
                "summary": f"Added {documentation_style} documentation",
            }
        except Exception as e:
            logger.error(f"Error documenting code: {str(e)}")
            return {"success": False, "documented_code": code, "documentation": f"Error: {str(e)}", "summary": ""}

    def _detect_language(self, code: str) -> str:
        code_lower = code.lower()
        language_patterns = {
            "python": ["def ", "import ", "from ", "print(", "class "],
            "javascript": ["function ", "const ", "let ", "var ", "console.log", "=>"],
            "java": ["public class", "public static", "System.out.println", "import java"],
            "cpp": ["#include", "using namespace", "cout <<", "std::"],
            "html": ["<!DOCTYPE", "<html", "<head", "<body", "<div"],
            "css": ["{", "}", ":", ";", ".class", "#id"],
            "sql": ["SELECT", "FROM", "WHERE", "INSERT", "UPDATE"],
        }
        for lang, patterns in language_patterns.items():
            if any(pattern in code_lower for pattern in patterns):
                return lang
        return "python"

    def _extract_code_blocks(self, text: str) -> List[str]:
        import re
        pattern = r"```(?:\w+)?\n(.*?)\n```"
        matches = re.findall(pattern, text, re.DOTALL)
        return [match.strip() for match in matches]

    def _extract_issues(self, text: str) -> List[str]:
        issues = []
        lines = text.split("\n")
        for line in lines:
            if any(keyword in line.lower() for keyword in ["error", "bug", "issue", "problem", "warning", "vulnerability"]):
                issues.append(line.strip())
        return issues[:5]

    def _get_default_test_framework(self, language: str) -> str:
        frameworks = {
            "python": "pytest",
            "javascript": "jest",
            "java": "junit",
            "cpp": "gtest",
            "csharp": "nunit",
            "go": "testing",
            "rust": "cargo test",
        }
        return frameworks.get(language, "unit testing")


# ==========================================
# OLLAMA SERVICE (Commented out)
# ==========================================
# class OllamaService:
#     def __init__(self):
#         """Initialize Ollama service"""
#         self.base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11435")
#         self.model = os.getenv("OLLAMA_MODEL", "codellama:7b")
# 
#         self.llm = ChatOllama(
#             base_url=self.base_url, model=self.model, temperature=0.7, num_predict=2000
#         )
# 
#         self.api_url = f"{self.base_url}/api"
# 
#         self.check_ollama_health()
# 
#         logger.info(f"Ollama Service initialized with model: {self.model}")
# 
#     def check_ollama_health(self):
#         """Check if Ollama is running"""
#         try:
#             response = requests.get(f"{self.base_url}/api/tags", timeout=5)
#             if response.status_code == 200:
#                 models = response.json().get("models", [])
#                 logger.info(
#                     f"Ollama is running. Available models: {[m['name'] for m in models]}"
#                 )
#                 return True
#             else:
#                 logger.error(f"Ollama health check failed: {response.status_code}")
#                 return False
#         except Exception as e:
#             logger.error(f"Ollama is not running: {str(e)}")
#             logger.info("Please start Ollama with: ollama serve")
#             return False
# 
#     def generate_with_api(
#         self, prompt: str, system_prompt: str = "", **kwargs
#     ) -> Dict[str, Any]:
#         """Generate response using Ollama REST API directly"""
#         try:
#             messages = []
#             if system_prompt:
#                 messages.append({"role": "system", "content": system_prompt})
#             messages.append({"role": "user", "content": prompt})
# 
#             payload = {
#                 "model": self.model,
#                 "messages": messages,
#                 "stream": False,
#                 "options": {
#                     "temperature": kwargs.get("temperature", 0.7),
#                     "num_predict": kwargs.get("num_predict", 2000),
#                     "top_p": kwargs.get("top_p", 0.9),
#                     "repeat_penalty": kwargs.get("repeat_penalty", 1.1),
#                 },
#             }
# 
#             response = requests.post(f"{self.api_url}/chat", json=payload, timeout=60)
# 
#             if response.status_code == 200:
#                 result = response.json()
#                 return {
#                     "success": True,
#                     "content": result["message"]["content"],
#                     "model": result["model"],
#                     "total_duration": result.get("total_duration", 0),
#                 }
#             else:
#                 logger.error(
#                     f"Ollama API error: {response.status_code} - {response.text}"
#                 )
#                 return {
#                     "success": False,
#                     "error": f"API Error: {response.status_code}",
#                     "content": "",
#                 }
# 
#         except Exception as e:
#             logger.error(f"Error calling Ollama API: {str(e)}")
#             return {"success": False, "error": str(e), "content": ""}
# 
#     def generate_code(
#         self, prompt: str, language: str = "python", context: str = ""
#     ) -> Dict[str, Any]:
#         """Generate code using Ollama"""
#         try:
#             system_prompt = f"You are an expert {language} programmer. Generate clean, efficient, and well-documented code.\n\nRequirements:\n1. Follow {language} best practices and conventions\n2. Include proper error handling\n3. Add meaningful comments\n4. Make it production-ready\n5. Include usage examples if applicable\n\nContext: {context}\n"
# 
#             user_prompt = f"Generate {language} code for: {prompt}"
# 
#             result = self.generate_with_api(
#                 prompt=user_prompt,
#                 system_prompt=system_prompt,
#                 temperature=0.3,  
#             )
# 
#             if result["success"]:
#                 return {
#                     "success": True,
#                     "code": result["content"],
#                     "explanation": f"Generated using {self.model}",
#                     "language": language,
#                 }
#             else:
#                 return self._generate_with_langchain(prompt, language, context)
# 
#         except Exception as e:
#             logger.error(f"Error generating code: {str(e)}")
#             return {
#                 "success": False,
#                 "error": str(e),
#                 "code": f"// Error generating code: {str(e)}",
#                 "explanation": "Failed to generate code",
#             }
# 
#     def _generate_with_langchain(
#         self, prompt: str, language: str, context: str
#     ) -> Dict[str, Any]:
#         """Generate code using LangChain Ollama"""
#         try:
#             system_message = f"You are an expert {language} programmer. Generate clean, efficient, and well-documented code.\n\nRequirements:\n1. Follow {language} best practices and conventions\n2. Include proper error handling\n3. Add meaningful comments\n4. Make it production-ready\n5. Include usage examples if applicable\n\nContext: {context}\n"
# 
#             messages = [
#                 SystemMessage(content=system_message),
#                 HumanMessage(content=f"Generate {language} code for: {prompt}"),
#             ]
# 
#             response = self.llm.invoke(messages)
# 
#             return {
#                 "success": True,
#                 "code": response.content,
#                 "explanation": f"Generated using {self.model} via LangChain",
#                 "language": language,
#             }
# 
#         except Exception as e:
#             logger.error(f"LangChain generation error: {str(e)}")
#             return {
#                 "success": False,
#                 "error": str(e),
#                 "code": f"// Error: {str(e)}",
#                 "explanation": "Generation failed",
#             }
# 
#     def explain_code(
#         self, code: str, language: str = "auto", detail_level: str = "comprehensive"
#     ) -> str:
#         """Explain code using Ollama"""
#         try:
#             if language == "auto":
#                 language = self._detect_language(code)
# 
#             system_prompt = f"You are an expert code explainer. Explain the {language} code in {detail_level} detail.\n\nExplanation should include:\n1. What the code does\n2. How it works (step by step)\n3. Key algorithms and data structures\n4. Time and space complexity analysis\n5. Use cases and applications\n6. Potential improvements\n\nMake the explanation clear and accessible to developers of all levels.\n"
# 
#             user_prompt = f"Explain this {language} code:\n```{language}\n{code}\n```"
# 
#             result = self.generate_with_api(
#                 prompt=user_prompt, system_prompt=system_prompt, temperature=0.3
#             )
# 
#             if result["success"]:
#                 return result["content"]
#             else:
#                 # Fallback
#                 return f"Explanation failed: {result.get('error', 'Unknown error')}"
# 
#         except Exception as e:
#             logger.error(f"Error explaining code: {str(e)}")
#             return f"Error explaining code: {str(e)}"
# 
#     def debug_code(
#         self, code: str, language: str = "auto", error_message: str = ""
#     ) -> Dict[str, Any]:
#         """Debug code using Ollama"""
#         try:
#             if language == "auto":
#                 language = self._detect_language(code)
# 
#             system_prompt = f"You are an expert debugger for {language}. Find and fix issues in the code.\n\nInstructions:\n1. Analyze the code for syntax errors\n2. Identify logical errors and bugs\n3. Check for runtime issues\n4. Look for security vulnerabilities\n5. Suggest specific fixes\n6. Provide corrected code\n\nError message: {error_message}\n"
# 
#             user_prompt = f"Debug this {language} code:\n```{language}\n{code}\n```"
# 
#             result = self.generate_with_api(
#                 prompt=user_prompt,
#                 system_prompt=system_prompt,
#                 temperature=0.2, 
#             )
# 
#             if result["success"]:
#                 content = result["content"]
# 
#                 corrected_code = self._extract_code_blocks(content)
# 
#                 return {
#                     "success": True,
#                     "debugged_code": corrected_code[0] if corrected_code else code,
#                     "explanation": content,
#                     "issues_found": self._extract_issues(content),
#                     "fixes_applied": ["Fixed issues using AI analysis"],
#                 }
#             else:
#                 return {
#                     "success": False,
#                     "debugged_code": code,
#                     "explanation": f"Debugging failed: {result.get('error', 'Unknown error')}",
#                     "issues_found": [],
#                     "fixes_applied": [],
#                 }
# 
#         except Exception as e:
#             logger.error(f"Error debugging code: {str(e)}")
#             return {
#                 "success": False,
#                 "debugged_code": code,
#                 "explanation": f"Error: {str(e)}",
#                 "issues_found": [],
#                 "fixes_applied": [],
#             }
# 
#     def optimize_code(
#         self, code: str, language: str = "auto", optimization_type: str = "performance"
#     ) -> Dict[str, Any]:
#         """Optimize code using Ollama"""
#         try:
#             if language == "auto":
#                 language = self._detect_language(code)
# 
#             optimization_focus = {
#                 "performance": "execution speed and efficiency",
#                 "readability": "code clarity and maintainability",
#                 "memory": "memory usage and footprint",
#             }.get(optimization_type, "performance")
# 
#             system_prompt = f"You are an expert {language} optimizer. Optimize the code for {optimization_focus}.\n\nOptimization should focus on:\n1. {optimization_type.capitalize()} improvements\n2. Best practices and patterns\n3. Error handling and robustness\n4. Code maintainability\n5. Documentation\n\nProvide both the optimized code and explanation of changes.\n"
# 
#             user_prompt = f"Optimize this {language} code:\n```{language}\n{code}\n```"
# 
#             result = self.generate_with_api(
#                 prompt=user_prompt, system_prompt=system_prompt, temperature=0.3
#             )
# 
#             if result["success"]:
#                 content = result["content"]
#                 optimized_code = self._extract_code_blocks(content)
# 
#                 return {
#                     "success": True,
#                     "optimized_code": optimized_code[0] if optimized_code else code,
#                     "explanation": content,
#                     "improvements": [f"Optimized for {optimization_type}"],
#                     "before_metrics": {"lines": len(code.split("\n"))},
#                     "after_metrics": {
#                         "lines": (
#                             len(optimized_code[0].split("\n"))
#                             if optimized_code
#                             else len(code.split("\n"))
#                         )
#                     },
#                 }
#             else:
#                 return {
#                     "success": False,
#                     "optimized_code": code,
#                     "explanation": f"Optimization failed: {result.get('error', 'Unknown error')}",
#                     "improvements": [],
#                     "before_metrics": {},
#                     "after_metrics": {},
#                 }
# 
#         except Exception as e:
#             logger.error(f"Error optimizing code: {str(e)}")
#             return {
#                 "success": False,
#                 "optimized_code": code,
#                 "explanation": f"Error: {str(e)}",
#                 "improvements": [],
#                 "before_metrics": {},
#                 "after_metrics": {},
#             }
# 
#     def chat(
#         self, message: str, history: List[Dict] = None, context: Dict = None
#     ) -> Dict[str, Any]:
#         """Chat with Ollama"""
#         try:
#             messages = []
#             system_content = "You are an AI coding assistant. You help with:\n1. Code generation and completion\n2. Code explanation and documentation\n3. Debugging and error fixing\n4. Code optimization and refactoring\n5. Best practices and design patterns\n6. Answering programming questions\n\nBe helpful, accurate, and provide code examples when needed.\nFormat code blocks with proper syntax highlighting.\n"
# 
#             if context:
#                 system_content += f"\nContext:\n{json.dumps(context, indent=2)}"
#             if history:
#                 for msg in history[-6:]:  
#                     if msg["role"] == "user":
#                         messages.append({"role": "user", "content": msg["content"]})
#                     elif msg["role"] == "assistant":
#                         messages.append(
#                             {"role": "assistant", "content": msg["content"]}
#                         )
# 
#             messages.append({"role": "user", "content": message})
# 
#             payload = {
#                 "model": self.model,
#                 "messages": [{"role": "system", "content": system_content}] + messages,
#                 "stream": False,
#                 "options": {"temperature": 0.7, "num_predict": 2000},
#             }
# 
#             response = requests.post(f"{self.api_url}/chat", json=payload, timeout=60)
# 
#             if response.status_code == 200:
#                 result = response.json()
# 
#                 new_history = (history or []) + [
#                     {"role": "user", "content": message},
#                     {"role": "assistant", "content": result["message"]["content"]},
#                 ]
# 
#                 return {
#                     "success": True,
#                     "response": result["message"]["content"],
#                     "history": new_history[-10:], 
#                     "model": result["model"],
#                 }
#             else:
#                 return {
#                     "success": False,
#                     "error": f"API Error: {response.status_code}",
#                     "response": f"Error: Failed to get response from Ollama",
#                     "history": history or [],
#                 }
# 
#         except Exception as e:
#             logger.error(f"Chat error: {str(e)}")
#             return {
#                 "success": False,
#                 "error": str(e),
#                 "response": f"Error: {str(e)}",
#                 "history": history or [],
#             }
# 
#     def write_tests(
#         self, code: str, language: str = "auto", test_framework: str = ""
#     ) -> Dict[str, Any]:
#         """Write tests using Ollama"""
#         try:
#             if language == "auto":
#                 language = self._detect_language(code)
# 
#             if not test_framework:
#                 test_framework = self._get_default_test_framework(language)
# 
#             system_prompt = f"You are an expert in writing tests for {language} using {test_framework}.\n\nWrite comprehensive tests that cover:\n1. Unit tests for individual functions\n2. Edge cases and boundary conditions\n3. Error cases and exception handling\n4. Integration tests if applicable\n5. Mocking dependencies if needed\n\nProvide complete test code with setup, teardown, and assertions.\n"
# 
#             user_prompt = (
#                 f"Write tests for this {language} code:\n```{language}\n{code}\n```"
#             )
# 
#             result = self.generate_with_api(
#                 prompt=user_prompt, system_prompt=system_prompt, temperature=0.3
#             )
# 
#             if result["success"]:
#                 content = result["content"]
#                 tests = self._extract_code_blocks(content)
# 
#                 return {
#                     "success": True,
#                     "tests": tests[0] if tests else "",
#                     "test_explanation": content,
#                     "coverage": 80.0,  
#                     "test_cases": ["Unit tests", "Edge cases", "Error handling"],
#                 }
#             else:
#                 return {
#                     "success": False,
#                     "tests": "",
#                     "test_explanation": f"Test generation failed: {result.get('error', 'Unknown error')}",
#                     "coverage": 0.0,
#                     "test_cases": [],
#                 }
# 
#         except Exception as e:
#             logger.error(f"Error writing tests: {str(e)}")
#             return {
#                 "success": False,
#                 "tests": "",
#                 "test_explanation": f"Error: {str(e)}",
#                 "coverage": 0.0,
#                 "test_cases": [],
#             }
# 
#     def analyze_code(
#         self, code: str, language: str = "auto", analysis_type: str = "comprehensive"
#     ) -> Dict[str, Any]:
#         """Analyze code using Ollama"""
#         try:
#             if language == "auto":
#                 language = self._detect_language(code)
# 
#             system_prompt = f"You are an expert code analyst. Perform {analysis_type} analysis of this {language} code.\n\nAnalysis should include:\n1. Code quality assessment\n2. Complexity analysis\n3. Maintainability score\n4. Security vulnerabilities\n5. Performance bottlenecks\n6. Style and consistency issues\n7. Recommendations for improvement\n\nProvide detailed analysis with specific examples.\n"
# 
#             user_prompt = f"Analyze this {language} code:\n```{language}\n{code}\n```"
# 
#             result = self.generate_with_api(
#                 prompt=user_prompt, system_prompt=system_prompt, temperature=0.3
#             )
# 
#             if result["success"]:
#                 content = result["content"]
# 
#                 return {
#                     "success": True,
#                     "analysis": content,
#                     "complexity": {"cyclomatic": "Medium", "cognitive": "Medium"},
#                     "quality_score": 75.0,
#                     "issues": self._extract_issues(content),
#                     "recommendations": [
#                         "Improve documentation",
#                         "Add error handling",
#                         "Optimize loops",
#                     ],
#                 }
#             else:
#                 return {
#                     "success": False,
#                     "analysis": f"Analysis failed: {result.get('error', 'Unknown error')}",
#                     "complexity": {},
#                     "quality_score": 0.0,
#                     "issues": [],
#                     "recommendations": [],
#                 }
# 
#         except Exception as e:
#             logger.error(f"Error analyzing code: {str(e)}")
#             return {
#                 "success": False,
#                 "analysis": f"Error: {str(e)}",
#                 "complexity": {},
#                 "quality_score": 0.0,
#                 "issues": [],
#                 "recommendations": [],
#             }
# 
#     def convert_code(
#         self, code: str, source_language: str, target_language: str
#     ) -> Dict[str, Any]:
#         """Convert code from one language to another"""
#         try:
#             if source_language == "auto":
#                 source_language = self._detect_language(code)
# 
#             system_prompt = f"You are an expert code converter. Convert code from {source_language} to {target_language}.\n\nConversion should:\n1. Preserve functionality exactly\n2. Use idiomatic {target_language} patterns\n3. Handle language-specific differences\n4. Maintain comments and documentation\n5. Note any compatibility issues\n"
# 
#             user_prompt = f"Convert this {source_language} code to {target_language}:\n```{source_language}\n{code}\n```"
# 
#             result = self.generate_with_api(
#                 prompt=user_prompt, system_prompt=system_prompt, temperature=0.3
#             )
# 
#             if result["success"]:
#                 content = result["content"]
#                 converted_code = self._extract_code_blocks(content)
# 
#                 return {
#                     "success": True,
#                     "converted_code": converted_code[0] if converted_code else "",
#                     "explanation": content,
#                     "compatibility_notes": [
#                         "Converted successfully",
#                         f"Used {target_language} idioms",
#                     ],
#                 }
#             else:
#                 return {
#                     "success": False,
#                     "converted_code": "",
#                     "explanation": f"Conversion failed: {result.get('error', 'Unknown error')}",
#                     "compatibility_notes": [],
#                 }
# 
#         except Exception as e:
#             logger.error(f"Error converting code: {str(e)}")
#             return {
#                 "success": False,
#                 "converted_code": "",
#                 "explanation": f"Error: {str(e)}",
#                 "compatibility_notes": [],
#             }
# 
#     def document_code(
#         self,
#         code: str,
#         language: str = "auto",
#         documentation_style: str = "comprehensive",
#     ) -> Dict[str, Any]:
#         """Add documentation to code"""
#         try:
#             if language == "auto":
#                 language = self._detect_language(code)
# 
#             system_prompt = f"You are an expert technical writer. Add {documentation_style} documentation to this {language} code.\n\nDocumentation should include:\n1. Function/class docstrings\n2. Parameter descriptions\n3. Return value explanations\n4. Usage examples\n5. Error handling notes\n6. Performance considerations\n"
# 
#             user_prompt = f"Add documentation to this {language} code:\n```{language}\n{code}\n```"
# 
#             result = self.generate_with_api(
#                 prompt=user_prompt, system_prompt=system_prompt, temperature=0.3
#             )
# 
#             if result["success"]:
#                 content = result["content"]
#                 documented_code = self._extract_code_blocks(content)
# 
#                 return {
#                     "success": True,
#                     "documented_code": documented_code[0] if documented_code else code,
#                     "documentation": content,
#                     "summary": f"Added {documentation_style} documentation",
#                 }
#             else:
#                 return {
#                     "success": False,
#                     "documented_code": code,
#                     "documentation": f"Documentation failed: {result.get('error', 'Unknown error')}",
#                     "summary": "",
#                 }
# 
#         except Exception as e:
#             logger.error(f"Error documenting code: {str(e)}")
#             return {
#                 "success": False,
#                 "documented_code": code,
#                 "documentation": f"Error: {str(e)}",
#                 "summary": "",
#             }
# 
#     def _detect_language(self, code: str) -> str:
#         """Detect programming language from code snippet"""
#         code_lower = code.lower()
# 
#         language_patterns = {
#             "python": ["def ", "import ", "from ", "print(", "class "],
#             "javascript": ["function ", "const ", "let ", "var ", "console.log", "=>"],
#             "java": [
#                 "public class",
#                 "public static",
#                 "System.out.println",
#                 "import java",
#             ],
#             "cpp": ["#include", "using namespace", "cout <<", "std::"],
#             "html": ["<!DOCTYPE", "<html", "<head", "<body", "<div"],
#             "css": ["{", "}", ":", ";", ".class", "#id"],
#             "sql": ["SELECT", "FROM", "WHERE", "INSERT", "UPDATE"],
#         }
# 
#         for lang, patterns in language_patterns.items():
#             if any(pattern in code_lower for pattern in patterns):
#                 return lang
# 
#         return "python"
# 
#     def _extract_code_blocks(self, text: str) -> List[str]:
#         """Extract code blocks from text"""
#         import re
# 
#         pattern = r"```(?:\w+)?\n(.*?)\n```"
#         matches = re.findall(pattern, text, re.DOTALL)
#         return [match.strip() for match in matches]
# 
#     def _extract_issues(self, text: str) -> List[str]:
#         """Extract issues from text"""
#         issues = []
#         lines = text.split("\n")
#         for line in lines:
#             if any(
#                 keyword in line.lower()
#                 for keyword in [
#                     "error",
#                     "bug",
#                     "issue",
#                     "problem",
#                     "warning",
#                     "vulnerability",
#                 ]
#             ):
#                 issues.append(line.strip())
#         return issues[:5]
# 
#     def _get_default_test_framework(self, language: str) -> str:
#         """Get default test framework for language"""
#         frameworks = {
#             "python": "pytest",
#             "javascript": "jest",
#             "java": "junit",
#             "cpp": "gtest",
#             "csharp": "nunit",
#             "go": "testing",
#             "rust": "cargo test",
#         }
#         return frameworks.get(language, "unit testing")
