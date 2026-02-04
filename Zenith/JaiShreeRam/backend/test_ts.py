import tree_sitter
import tree_sitter_python
from tree_sitter import Language, Parser

try:
    PY_LANGUAGE = Language(tree_sitter_python.language())
    parser = Parser(PY_LANGUAGE)
    print("Success with Parser(PY_LANGUAGE)")
except Exception as e:
    print(f"Failed Parser(PY_LANGUAGE): {e}")

try:
    parser = Parser()
    parser.language = PY_LANGUAGE
    print("Success with parser.language = PY_LANGUAGE")
except Exception as e:
    print(f"Failed parser.language = PY_LANGUAGE: {e}")

try:
    parser = Parser()
    parser.set_language(PY_LANGUAGE)
    print("Success with parser.set_language(PY_LANGUAGE)")
except Exception as e:
    print(f"Failed parser.set_language(PY_LANGUAGE): {e}")
