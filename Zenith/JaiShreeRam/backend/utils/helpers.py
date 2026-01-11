import os
import json
import logging
from typing import Any, Dict, List, Optional
import hashlib
import re

logger = logging.getLogger(__name__)


def validate_api_key(api_key: str) -> bool:
    """Validate OpenAI API key format"""
    if not api_key or api_key == "placeholder-key":
        return False
    # Basic validation - OpenAI keys start with 'sk-'
    return api_key.startswith("sk-")


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage"""
    # Remove any directory path
    filename = os.path.basename(filename)
    # Remove non-alphanumeric characters (except dots, hyphens, underscores)
    filename = re.sub(r"[^a-zA-Z0-9._-]", "_", filename)
    return filename


def extract_code_blocks(text: str) -> List[str]:
    """Extract code blocks from markdown text"""
    pattern = r"```(?:\w+)?\n(.*?)\n```"
    matches = re.findall(pattern, text, re.DOTALL)
    return [match.strip() for match in matches]


def calculate_complexity_metrics(code: str) -> Dict[str, Any]:
    """Calculate basic code complexity metrics"""
    lines = code.split("\n")

    metrics = {
        "lines_of_code": len(lines),
        "non_empty_lines": sum(1 for line in lines if line.strip()),
        "comment_lines": sum(
            1 for line in lines if line.strip().startswith(("#", "//", "/*", "*", "*/"))
        ),
        "function_count": len(re.findall(r"(def|function|func)\s+\w+\s*\(", code)),
        "class_count": len(re.findall(r"class\s+\w+", code)),
        "control_structures": len(
            re.findall(r"\b(if|else|for|while|switch|case|try|catch|finally)\b", code)
        ),
    }

    # Calculate ratios
    if metrics["non_empty_lines"] > 0:
        metrics["comment_ratio"] = metrics["comment_lines"] / metrics["non_empty_lines"]
        metrics["complexity_density"] = (
            metrics["control_structures"] / metrics["non_empty_lines"]
        )
    else:
        metrics["comment_ratio"] = 0
        metrics["complexity_density"] = 0

    return metrics


def format_size(bytes_size: int) -> str:
    """Format bytes to human readable size"""
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} TB"


def generate_file_hash(content: str, algorithm: str = "md5") -> str:
    """Generate hash for file content"""
    hash_func = hashlib.new(algorithm)
    hash_func.update(content.encode("utf-8"))
    return hash_func.hexdigest()


def validate_json_schema(data: Dict, schema: Dict) -> bool:
    """Validate data against JSON schema (simplified)"""
    try:
        for key, value_type in schema.items():
            if key not in data:
                return False
            if not isinstance(data[key], value_type):
                return False
        return True
    except Exception:
        return False


def truncate_text(text: str, max_length: int = 500, suffix: str = "...") -> str:
    """Truncate text to maximum length"""
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def parse_language_from_filename(filename: str) -> str:
    """Parse language from filename extension"""
    ext = os.path.splitext(filename)[1].lower()

    extension_map = {
        ".py": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".java": "java",
        ".cpp": "cpp",
        ".c": "cpp",
        ".h": "cpp",
        ".cs": "csharp",
        ".go": "go",
        ".rs": "rust",
        ".php": "php",
        ".rb": "ruby",
        ".swift": "swift",
        ".kt": "kotlin",
        ".html": "html",
        ".htm": "html",
        ".css": "css",
        ".scss": "css",
        ".sass": "css",
        ".less": "css",
        ".sql": "sql",
        ".json": "json",
        ".yml": "yaml",
        ".yaml": "yaml",
        ".md": "markdown",
        ".txt": "text",
        ".xml": "xml",
        ".csv": "csv",
    }

    return extension_map.get(ext, "unknown")


def safe_json_dumps(data: Any) -> str:
    """Safely convert data to JSON string"""

    def default_serializer(obj):
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        return str(obj)

    return json.dumps(data, default=default_serializer, ensure_ascii=False)


def measure_execution_time(func):
    """Decorator to measure execution time"""
    import time
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        logger.debug(f"{func.__name__} executed in {end_time - start_time:.3f} seconds")

        if hasattr(result, "__dict__"):
            result.execution_time = end_time - start_time
        elif isinstance(result, dict):
            result["execution_time"] = end_time - start_time

        return result

    return wrapper


class Cache:
    """Simple in-memory cache"""

    def __init__(self, max_size: int = 100):
        self.cache = {}
        self.max_size = max_size
        self.access_order = []

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if key in self.cache:
            # Update access order
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None

    def set(self, key: str, value: Any):
        """Set value in cache"""
        if len(self.cache) >= self.max_size:
            # Remove least recently used item
            lru_key = self.access_order.pop(0)
            del self.cache[lru_key]

        self.cache[key] = value
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)

    def clear(self):
        """Clear cache"""
        self.cache.clear()
        self.access_order.clear()
