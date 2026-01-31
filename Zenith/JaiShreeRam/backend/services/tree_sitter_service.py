
import os
import logging
from typing import List, Dict, Any, Optional
import tree_sitter
try:
    from tree_sitter import Language, Parser
except ImportError:
    # Handle older versions or specific bindings if needed, but 0.22+ is standard
    pass

# Import language bindings
# Note: In a real environment, we'd need to ensure these are installed and compiled/accessible.
# Since we are adding them to requirements.txt, we assume they are available.
# However, mapping them to Language objects requires pointing to the library file (.so/.dll) 
# OR using the new bindings like `tree_sitter_python.language()` in v0.22+

logger = logging.getLogger(__name__)

class TreeSitterService:
    """
    Service for parsing code using Tree-sitter to extract structure (classes, functions).
    Supports multiple languages beyond just Python.
    """

    def __init__(self):
        self.parsers = {}
        self.languages = {}
        self._initialize_parsers()

    def _initialize_parsers(self):
        """Initialize parsers for supported languages"""
        try:
            # Python
            try:
                import tree_sitter_python
                PY_LANGUAGE = Language(tree_sitter_python.language())
                parser_py = Parser(PY_LANGUAGE)
                self.parsers['python'] = parser_py
                self.languages['python'] = PY_LANGUAGE
            except ImportError:
                logger.warning("tree-sitter-python not installed")
            except Exception as e:
                logger.warning(f"Failed to load python parser: {e}")

            # JavaScript
            try:
                import tree_sitter_javascript
                JS_LANGUAGE = Language(tree_sitter_javascript.language())
                parser_js = Parser(JS_LANGUAGE)
                self.parsers['javascript'] = parser_js
                self.languages['javascript'] = JS_LANGUAGE
            except ImportError:
                logger.warning("tree-sitter-javascript not installed")
            except Exception as e:
                logger.warning(f"Failed to load javascript parser: {e}")

            # TypeScript
            try:
                import tree_sitter_typescript
                TS_LANGUAGE = Language(tree_sitter_typescript.language_typescript())
                parser_ts = Parser(TS_LANGUAGE)
                self.parsers['typescript'] = parser_ts
                self.languages['typescript'] = TS_LANGUAGE
            except ImportError:
                logger.warning("tree-sitter-typescript not installed")
            except Exception as e:
                logger.warning(f"Failed to load typescript parser: {e}")

        except Exception as e:
            logger.error(f"Error initializing Tree-sitter parsers: {e}")

    def get_parser(self, language: str):
        return self.parsers.get(language.lower())

    def parse_file(self, file_path: str, content: str = None) -> Dict[str, Any]:
        """
        Parse a file and return its structure (classes, functions).
        """
        if content is None:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except Exception as e:
                logger.error(f"Failed to read file {file_path}: {e}")
                return {"error": str(e)}

        ext = os.path.splitext(file_path)[1].lower()
        language = self._get_language_from_ext(ext)
        
        if not language or language not in self.parsers:
            return {"language": "unknown", "structure": "Not supported"}

        parser = self.parsers[language]
        
        try:
            tree = parser.parse(bytes(content, "utf8"))
            root_node = tree.root_node
            
            structure = self._extract_structure(root_node, language)
            return {
                "language": language,
                "classes": structure['classes'],
                "functions": structure['functions']
            }
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
            return {"error": str(e)}

    def _get_language_from_ext(self, ext: str) -> str:
        map = {
            '.py': 'python',
            '.js': 'javascript',
            '.jsx': 'javascript',
            '.ts': 'typescript',
            '.tsx': 'typescript'
        }
        return map.get(ext)

    def _extract_structure(self, node, language: str) -> Dict[str, List[str]]:
        classes = []
        functions = []

        # Queries based on language
        if language == 'python':
            # Simple interaction for now, query API is cleaner but traversal is universally understood
            # Function definitions
            # Class definitions
            queries = {
                'class': '(class_definition name: (identifier) @name)',
                'function': '(function_definition name: (identifier) @name)'
            }
        elif language in ['javascript', 'typescript']:
            queries = {
                'class': '(class_declaration name: (identifier) @name)',
                'function': [
                    '(function_declaration name: (identifier) @name)',
                    '(method_definition name: (property_identifier) @name)',
                    '(variable_declarator name: (identifier) @name value: (arrow_function))' 
                ]
            }
        else:
            return {'classes': [], 'functions': []}

        # Use tree-sitter query API
        try:
            lang_obj = self.languages[language]
            
            # Extract Classes
            class_query_str = queries['class']
            if isinstance(class_query_str, list): class_query_str = " ".join(class_query_str) # Incorrect for OR, need separate queries or combining
            
            # Simplified Traversal approach for robustness without perfecting exact query syntax 
            # (since query syntax can be strict and version-dependent)
            # We will use a recursive traversal which is verbose but reliable.
            
            self._traverse_node(node, classes, functions, language)
            
        except Exception as e:
           logger.error(f"Query error: {e}")

        return {'classes': classes, 'functions': functions}

    def _traverse_node(self, node, classes, functions, language):
        type = node.type
        
        if language == 'python':
            if type == 'class_definition':
                name = self._get_node_name(node)
                if name: classes.append(name)
            elif type == 'function_definition':
                name = self._get_node_name(node)
                if name: functions.append(name)
        elif language in ['javascript', 'typescript']:
            if type == 'class_declaration':
                name = self._get_node_name(node)
                if name: classes.append(name)
            elif type in ['function_declaration', 'method_definition']:
                name = self._get_node_name(node)
                if name: functions.append(name)
            elif type == 'variable_declarator':
                # Check for arrow function assignment: const foo = () => {}
                # child_by_field_name('value') -> arrow_function
                value_node = node.child_by_field_name('value')
                if value_node and value_node.type == 'arrow_function':
                    name = self._get_node_name(node)
                    if name: functions.append(name)

        for child in node.children:
            self._traverse_node(child, classes, functions, language)

    def _get_node_name(self, node):
        # Python/JS/TS usually have a 'name' field which is an identifier
        name_node = node.child_by_field_name('name')
        if name_node:
            return name_node.text.decode('utf8')
        return None
        
