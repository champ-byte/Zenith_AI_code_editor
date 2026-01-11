import os
import json
import logging
from typing import Dict, List, Any, Optional
import requests
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

logger = logging.getLogger(__name__)


class OllamaService:
    def __init__(self):
        """Initialize Ollama service"""
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11435")
        self.model = os.getenv("OLLAMA_MODEL", "codellama:7b")

        self.llm = ChatOllama(
            base_url=self.base_url, model=self.model, temperature=0.7, num_predict=2000
        )

        self.api_url = f"{self.base_url}/api"

        self.check_ollama_health()

        logger.info(f"Ollama Service initialized with model: {self.model}")

    def check_ollama_health(self):
        """Check if Ollama is running"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                logger.info(
                    f"Ollama is running. Available models: {[m['name'] for m in models]}"
                )
                return True
            else:
                logger.error(f"Ollama health check failed: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Ollama is not running: {str(e)}")
            logger.info("Please start Ollama with: ollama serve")
            return False

    def generate_with_api(
        self, prompt: str, system_prompt: str = "", **kwargs
    ) -> Dict[str, Any]:
        """Generate response using Ollama REST API directly"""
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": kwargs.get("temperature", 0.7),
                    "num_predict": kwargs.get("num_predict", 2000),
                    "top_p": kwargs.get("top_p", 0.9),
                    "repeat_penalty": kwargs.get("repeat_penalty", 1.1),
                },
            }

            response = requests.post(f"{self.api_url}/chat", json=payload, timeout=60)

            if response.status_code == 200:
                result = response.json()
                return {
                    "success": True,
                    "content": result["message"]["content"],
                    "model": result["model"],
                    "total_duration": result.get("total_duration", 0),
                }
            else:
                logger.error(
                    f"Ollama API error: {response.status_code} - {response.text}"
                )
                return {
                    "success": False,
                    "error": f"API Error: {response.status_code}",
                    "content": "",
                }

        except Exception as e:
            logger.error(f"Error calling Ollama API: {str(e)}")
            return {"success": False, "error": str(e), "content": ""}

    def generate_code(
        self, prompt: str, language: str = "python", context: str = ""
    ) -> Dict[str, Any]:
        """Generate code using Ollama"""
        try:
            system_prompt = f"""You are an expert {language} programmer. Generate clean, efficient, and well-documented code.

Requirements:
1. Follow {language} best practices and conventions
2. Include proper error handling
3. Add meaningful comments
4. Make it production-ready
5. Include usage examples if applicable

Context: {context}
"""

            user_prompt = f"Generate {language} code for: {prompt}"

            result = self.generate_with_api(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.3, 
            )

            if result["success"]:
                return {
                    "success": True,
                    "code": result["content"],
                    "explanation": f"Generated using {self.model}",
                    "language": language,
                }
            else:
                return self._generate_with_langchain(prompt, language, context)

        except Exception as e:
            logger.error(f"Error generating code: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "code": f"// Error generating code: {str(e)}",
                "explanation": "Failed to generate code",
            }

    def _generate_with_langchain(
        self, prompt: str, language: str, context: str
    ) -> Dict[str, Any]:
        """Generate code using LangChain Ollama"""
        try:
            system_message = f"""You are an expert {language} programmer. Generate clean, efficient, and well-documented code.

Requirements:
1. Follow {language} best practices and conventions
2. Include proper error handling
3. Add meaningful comments
4. Make it production-ready
5. Include usage examples if applicable

Context: {context}
"""

            messages = [
                SystemMessage(content=system_message),
                HumanMessage(content=f"Generate {language} code for: {prompt}"),
            ]

            response = self.llm.invoke(messages)

            return {
                "success": True,
                "code": response.content,
                "explanation": f"Generated using {self.model} via LangChain",
                "language": language,
            }

        except Exception as e:
            logger.error(f"LangChain generation error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "code": f"// Error: {str(e)}",
                "explanation": "Generation failed",
            }

    def explain_code(
        self, code: str, language: str = "auto", detail_level: str = "comprehensive"
    ) -> str:
        """Explain code using Ollama"""
        try:
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

Make the explanation clear and accessible to developers of all levels.
"""

            user_prompt = f"Explain this {language} code:\n```{language}\n{code}\n```"

            result = self.generate_with_api(
                prompt=user_prompt, system_prompt=system_prompt, temperature=0.3
            )

            if result["success"]:
                return result["content"]
            else:
                # Fallback
                return f"Explanation failed: {result.get('error', 'Unknown error')}"

        except Exception as e:
            logger.error(f"Error explaining code: {str(e)}")
            return f"Error explaining code: {str(e)}"

    def debug_code(
        self, code: str, language: str = "auto", error_message: str = ""
    ) -> Dict[str, Any]:
        """Debug code using Ollama"""
        try:
            if language == "auto":
                language = self._detect_language(code)

            system_prompt = f"""You are an expert debugger for {language}. Find and fix issues in the code.

Instructions:
1. Analyze the code for syntax errors
2. Identify logical errors and bugs
3. Check for runtime issues
4. Look for security vulnerabilities
5. Suggest specific fixes
6. Provide corrected code

Error message: {error_message}
"""

            user_prompt = f"Debug this {language} code:\n```{language}\n{code}\n```"

            result = self.generate_with_api(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.2,  
            )

            if result["success"]:
                content = result["content"]
                corrected_code = self._extract_code_blocks(content)

                return {
                    "success": True,
                    "debugged_code": corrected_code[0] if corrected_code else code,
                    "explanation": content,
                    "issues_found": self._extract_issues(content),
                    "fixes_applied": ["Fixed issues using AI analysis"],
                }
            else:
                return {
                    "success": False,
                    "debugged_code": code,
                    "explanation": f"Debugging failed: {result.get('error', 'Unknown error')}",
                    "issues_found": [],
                    "fixes_applied": [],
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

    def optimize_code(
        self, code: str, language: str = "auto", optimization_type: str = "performance"
    ) -> Dict[str, Any]:
        """Optimize code using Ollama"""
        try:
            if language == "auto":
                language = self._detect_language(code)

            optimization_focus = {
                "performance": "execution speed and efficiency",
                "readability": "code clarity and maintainability",
                "memory": "memory usage and footprint",
            }.get(optimization_type, "performance")

            system_prompt = f"""You are an expert {language} optimizer. Optimize the code for {optimization_focus}.

Optimization should focus on:
1. {optimization_type.capitalize()} improvements
2. Best practices and patterns
3. Error handling and robustness
4. Code maintainability
5. Documentation

Provide both the optimized code and explanation of changes.
"""

            user_prompt = f"Optimize this {language} code:\n```{language}\n{code}\n```"

            result = self.generate_with_api(
                prompt=user_prompt, system_prompt=system_prompt, temperature=0.3
            )

            if result["success"]:
                content = result["content"]
                optimized_code = self._extract_code_blocks(content)

                return {
                    "success": True,
                    "optimized_code": optimized_code[0] if optimized_code else code,
                    "explanation": content,
                    "improvements": [f"Optimized for {optimization_type}"],
                    "before_metrics": {"lines": len(code.split("\n"))},
                    "after_metrics": {
                        "lines": (
                            len(optimized_code[0].split("\n"))
                            if optimized_code
                            else len(code.split("\n"))
                        )
                    },
                }
            else:
                return {
                    "success": False,
                    "optimized_code": code,
                    "explanation": f"Optimization failed: {result.get('error', 'Unknown error')}",
                    "improvements": [],
                    "before_metrics": {},
                    "after_metrics": {},
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

    def chat(
        self, message: str, history: List[Dict] = None, context: Dict = None
    ) -> Dict[str, Any]:
        """Chat with Ollama"""
        try:
            messages = []
            system_content = """You are an AI coding assistant. You help with:
1. Code generation and completion
2. Code explanation and documentation
3. Debugging and error fixing
4. Code optimization and refactoring
5. Best practices and design patterns
6. Answering programming questions

Be helpful, accurate, and provide code examples when needed.
Format code blocks with proper syntax highlighting.
"""

            if context:
                system_content += f"\nContext:\n{json.dumps(context, indent=2)}"

            if history:
                for msg in history[-6:]:  
                    if msg["role"] == "user":
                        messages.append({"role": "user", "content": msg["content"]})
                    elif msg["role"] == "assistant":
                        messages.append(
                            {"role": "assistant", "content": msg["content"]}
                        )

            messages.append({"role": "user", "content": message})

            payload = {
                "model": self.model,
                "messages": [{"role": "system", "content": system_content}] + messages,
                "stream": False,
                "options": {"temperature": 0.7, "num_predict": 2000},
            }

            response = requests.post(f"{self.api_url}/chat", json=payload, timeout=60)

            if response.status_code == 200:
                result = response.json()

                new_history = (history or []) + [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": result["message"]["content"]},
                ]

                return {
                    "success": True,
                    "response": result["message"]["content"],
                    "history": new_history[-10:],  
                    "model": result["model"],
                }
            else:
                return {
                    "success": False,
                    "error": f"API Error: {response.status_code}",
                    "response": f"Error: Failed to get response from Ollama",
                    "history": history or [],
                }

        except Exception as e:
            logger.error(f"Chat error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "response": f"Error: {str(e)}",
                "history": history or [],
            }

    def write_tests(
        self, code: str, language: str = "auto", test_framework: str = ""
    ) -> Dict[str, Any]:
        """Write tests using Ollama"""
        try:
            if language == "auto":
                language = self._detect_language(code)

            if not test_framework:
                test_framework = self._get_default_test_framework(language)

            system_prompt = f"""You are an expert in writing tests for {language} using {test_framework}.

Write comprehensive tests that cover:
1. Unit tests for individual functions
2. Edge cases and boundary conditions
3. Error cases and exception handling
4. Integration tests if applicable
5. Mocking dependencies if needed

Provide complete test code with setup, teardown, and assertions.
"""

            user_prompt = (
                f"Write tests for this {language} code:\n```{language}\n{code}\n```"
            )

            result = self.generate_with_api(
                prompt=user_prompt, system_prompt=system_prompt, temperature=0.3
            )

            if result["success"]:
                content = result["content"]
                tests = self._extract_code_blocks(content)

                return {
                    "success": True,
                    "tests": tests[0] if tests else "",
                    "test_explanation": content,
                    "coverage": 80.0,  # Estimated
                    "test_cases": ["Unit tests", "Edge cases", "Error handling"],
                }
            else:
                return {
                    "success": False,
                    "tests": "",
                    "test_explanation": f"Test generation failed: {result.get('error', 'Unknown error')}",
                    "coverage": 0.0,
                    "test_cases": [],
                }

        except Exception as e:
            logger.error(f"Error writing tests: {str(e)}")
            return {
                "success": False,
                "tests": "",
                "test_explanation": f"Error: {str(e)}",
                "coverage": 0.0,
                "test_cases": [],
            }

    def analyze_code(
        self, code: str, language: str = "auto", analysis_type: str = "comprehensive"
    ) -> Dict[str, Any]:
        """Analyze code using Ollama"""
        try:
            if language == "auto":
                language = self._detect_language(code)

            system_prompt = f"""You are an expert code analyst. Perform {analysis_type} analysis of this {language} code.

Analysis should include:
1. Code quality assessment
2. Complexity analysis
3. Maintainability score
4. Security vulnerabilities
5. Performance bottlenecks
6. Style and consistency issues
7. Recommendations for improvement

Provide detailed analysis with specific examples.
"""

            user_prompt = f"Analyze this {language} code:\n```{language}\n{code}\n```"

            result = self.generate_with_api(
                prompt=user_prompt, system_prompt=system_prompt, temperature=0.3
            )

            if result["success"]:
                content = result["content"]

                return {
                    "success": True,
                    "analysis": content,
                    "complexity": {"cyclomatic": "Medium", "cognitive": "Medium"},
                    "quality_score": 75.0,
                    "issues": self._extract_issues(content),
                    "recommendations": [
                        "Improve documentation",
                        "Add error handling",
                        "Optimize loops",
                    ],
                }
            else:
                return {
                    "success": False,
                    "analysis": f"Analysis failed: {result.get('error', 'Unknown error')}",
                    "complexity": {},
                    "quality_score": 0.0,
                    "issues": [],
                    "recommendations": [],
                }

        except Exception as e:
            logger.error(f"Error analyzing code: {str(e)}")
            return {
                "success": False,
                "analysis": f"Error: {str(e)}",
                "complexity": {},
                "quality_score": 0.0,
                "issues": [],
                "recommendations": [],
            }

    def convert_code(
        self, code: str, source_language: str, target_language: str
    ) -> Dict[str, Any]:
        """Convert code from one language to another"""
        try:
            if source_language == "auto":
                source_language = self._detect_language(code)

            system_prompt = f"""You are an expert code converter. Convert code from {source_language} to {target_language}.

Conversion should:
1. Preserve functionality exactly
2. Use idiomatic {target_language} patterns
3. Handle language-specific differences
4. Maintain comments and documentation
5. Note any compatibility issues
"""

            user_prompt = f"Convert this {source_language} code to {target_language}:\n```{source_language}\n{code}\n```"

            result = self.generate_with_api(
                prompt=user_prompt, system_prompt=system_prompt, temperature=0.3
            )

            if result["success"]:
                content = result["content"]
                converted_code = self._extract_code_blocks(content)

                return {
                    "success": True,
                    "converted_code": converted_code[0] if converted_code else "",
                    "explanation": content,
                    "compatibility_notes": [
                        "Converted successfully",
                        f"Used {target_language} idioms",
                    ],
                }
            else:
                return {
                    "success": False,
                    "converted_code": "",
                    "explanation": f"Conversion failed: {result.get('error', 'Unknown error')}",
                    "compatibility_notes": [],
                }

        except Exception as e:
            logger.error(f"Error converting code: {str(e)}")
            return {
                "success": False,
                "converted_code": "",
                "explanation": f"Error: {str(e)}",
                "compatibility_notes": [],
            }

    def document_code(
        self,
        code: str,
        language: str = "auto",
        documentation_style: str = "comprehensive",
    ) -> Dict[str, Any]:
        """Add documentation to code"""
        try:
            if language == "auto":
                language = self._detect_language(code)

            system_prompt = f"""You are an expert technical writer. Add {documentation_style} documentation to this {language} code.

Documentation should include:
1. Function/class docstrings
2. Parameter descriptions
3. Return value explanations
4. Usage examples
5. Error handling notes
6. Performance considerations
"""

            user_prompt = f"Add documentation to this {language} code:\n```{language}\n{code}\n```"

            result = self.generate_with_api(
                prompt=user_prompt, system_prompt=system_prompt, temperature=0.3
            )

            if result["success"]:
                content = result["content"]
                documented_code = self._extract_code_blocks(content)

                return {
                    "success": True,
                    "documented_code": documented_code[0] if documented_code else code,
                    "documentation": content,
                    "summary": f"Added {documentation_style} documentation",
                }
            else:
                return {
                    "success": False,
                    "documented_code": code,
                    "documentation": f"Documentation failed: {result.get('error', 'Unknown error')}",
                    "summary": "",
                }

        except Exception as e:
            logger.error(f"Error documenting code: {str(e)}")
            return {
                "success": False,
                "documented_code": code,
                "documentation": f"Error: {str(e)}",
                "summary": "",
            }

    # Helper methods
    def _detect_language(self, code: str) -> str:
        """Detect programming language from code snippet"""
        code_lower = code.lower()

        language_patterns = {
            "python": ["def ", "import ", "from ", "print(", "class "],
            "javascript": ["function ", "const ", "let ", "var ", "console.log", "=>"],
            "java": [
                "public class",
                "public static",
                "System.out.println",
                "import java",
            ],
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
        """Extract code blocks from text"""
        import re

        pattern = r"```(?:\w+)?\n(.*?)\n```"
        matches = re.findall(pattern, text, re.DOTALL)
        return [match.strip() for match in matches]

    def _extract_issues(self, text: str) -> List[str]:
        """Extract issues from text"""
        issues = []
        lines = text.split("\n")
        for line in lines:
            if any(
                keyword in line.lower()
                for keyword in [
                    "error",
                    "bug",
                    "issue",
                    "problem",
                    "warning",
                    "vulnerability",
                ]
            ):
                issues.append(line.strip())
        return issues[:5]

    def _get_default_test_framework(self, language: str) -> str:
        """Get default test framework for language"""
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
