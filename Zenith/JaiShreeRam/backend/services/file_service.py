import os
import tempfile
import logging
from typing import List, Dict, Any
import hashlib
from werkzeug.utils import secure_filename

logger = logging.getLogger(__name__)


class FileService:
    def __init__(self):
        self.allowed_extensions = {
            ".py",
            ".js",
            ".jsx",
            ".ts",
            ".tsx",
            ".java",
            ".cpp",
            ".c",
            ".h",
            ".hpp",
            ".cs",
            ".go",
            ".rs",
            ".php",
            ".rb",
            ".swift",
            ".kt",
            ".html",
            ".htm",
            ".css",
            ".scss",
            ".sass",
            ".less",
            ".sql",
            ".json",
            ".yml",
            ".yaml",
            ".md",
            ".txt",
            ".xml",
            ".csv",
        }

        self.language_extensions = {
            "python": {".py"},
            "javascript": {".js", ".jsx"},
            "typescript": {".ts", ".tsx"},
            "java": {".java"},
            "cpp": {".cpp", ".c", ".h", ".hpp", ".cc", ".cxx"},
            "csharp": {".cs"},
            "go": {".go"},
            "rust": {".rs"},
            "php": {".php"},
            "ruby": {".rb"},
            "swift": {".swift"},
            "kotlin": {".kt"},
            "html": {".html", ".htm"},
            "css": {".css", ".scss", ".sass", ".less"},
            "sql": {".sql"},
            "json": {".json"},
            "yaml": {".yml", ".yaml"},
            "markdown": {".md"},
            "text": {".txt"},
            "xml": {".xml"},
            "csv": {".csv"},
        }

    def analyze_files(self, files, analysis_type: str = "structure") -> Dict[str, Any]:
        """Analyze uploaded files"""
        try:
            temp_dir = tempfile.mkdtemp()
            file_analysis = []
            language_counts = {}
            total_lines = 0
            total_size = 0

            for file in files:
                if file.filename == "":
                    continue

                filename = secure_filename(file.filename)
                file_path = os.path.join(temp_dir, filename)
                file.save(file_path)

                analysis = self._analyze_single_file(file_path, analysis_type)
                file_analysis.append(analysis)

                lang = analysis.get("language", "unknown")
                language_counts[lang] = language_counts.get(lang, 0) + 1

                total_lines += analysis.get("line_count", 0)
                total_size += analysis.get("size_bytes", 0)

            import shutil

            shutil.rmtree(temp_dir)

            overall_analysis = self._generate_overall_analysis(
                file_analysis, language_counts, total_lines, total_size
            )

            return {
                "analysis": overall_analysis,
                "file_count": len(files),
                "language_distribution": language_counts,
                "total_lines": total_lines,
                "total_size": total_size,
                "file_details": file_analysis[:10], 
            }

        except Exception as e:
            logger.error(f"Error analyzing files: {str(e)}")
            return {
                "analysis": f"Error: {str(e)}",
                "file_count": 0,
                "language_distribution": {},
                "total_lines": 0,
                "total_size": 0,
                "file_details": [],
            }

    def _analyze_single_file(
        self, file_path: str, analysis_type: str
    ) -> Dict[str, Any]:
        """Analyze a single file"""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            filename = os.path.basename(file_path)
            extension = os.path.splitext(filename)[1].lower()

            language = self._detect_language_from_extension(extension)

            line_count = len(content.split("\n"))
            size_bytes = os.path.getsize(file_path)

            file_hash = hashlib.md5(content.encode()).hexdigest()[:8]

            analysis = {
                "filename": filename,
                "path": file_path,
                "language": language,
                "extension": extension,
                "line_count": line_count,
                "size_bytes": size_bytes,
                "hash": file_hash,
                "analysis_type": analysis_type,
            }

            if analysis_type == "structure":
                analysis.update(self._analyze_structure(content, language))
            elif analysis_type == "complexity":
                analysis.update(self._analyze_complexity(content, language))
            elif analysis_type == "security":
                analysis.update(self._analyze_security(content, language))

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing file {file_path}: {str(e)}")
            return {
                "filename": os.path.basename(file_path),
                "error": str(e),
                "language": "unknown",
            }

    def _detect_language_from_extension(self, extension: str) -> str:
        """Detect language from file extension"""
        for lang, extensions in self.language_extensions.items():
            if extension in extensions:
                return lang
        return "unknown"

    def _analyze_structure(self, content: str, language: str) -> Dict[str, Any]:
        """Analyze file structure"""
        lines = content.split("\n")

        code_lines = 0
        comment_lines = 0
        blank_lines = 0
        import_lines = 0
        function_count = 0
        class_count = 0

        for line in lines:
            stripped = line.strip()
            if not stripped:
                blank_lines += 1
            elif (
                stripped.startswith("#")
                or stripped.startswith("//")
                or stripped.startswith("/*")
            ):
                comment_lines += 1
            elif "import" in stripped or "require" in stripped or "using" in stripped:
                import_lines += 1
            elif (
                stripped.startswith("def ")
                or stripped.startswith("function ")
                or stripped.startswith("func ")
            ):
                function_count += 1
            elif stripped.startswith("class "):
                class_count += 1
            else:
                code_lines += 1

        return {
            "structure": {
                "code_lines": code_lines,
                "comment_lines": comment_lines,
                "blank_lines": blank_lines,
                "import_lines": import_lines,
                "function_count": function_count,
                "class_count": class_count,
            },
            "comment_ratio": comment_lines / max(len(lines), 1),
            "complexity_score": function_count + class_count,
        }

    def _analyze_complexity(self, content: str, language: str) -> Dict[str, Any]:
        """Analyze code complexity (simplified)"""
        lines = content.split("\n")

        complexity_indicators = 0
        for line in lines:
            stripped = line.strip()
            if any(
                keyword in stripped
                for keyword in [
                    "if ",
                    "for ",
                    "while ",
                    "switch ",
                    "case ",
                    "try:",
                    "except ",
                    "catch ",
                ]
            ):
                complexity_indicators += 1
            complexity_indicators += stripped.count("{") + stripped.count("}")

        return {
            "complexity": {
                "indicators": complexity_indicators,
                "avg_complexity": complexity_indicators / max(len(lines), 1),
                "level": (
                    "Low"
                    if complexity_indicators < 10
                    else "Medium" if complexity_indicators < 30 else "High"
                ),
            }
        }

    def _analyze_security(self, content: str, language: str) -> Dict[str, Any]:
        """Analyze security issues (simplified)"""
        security_issues = []
        lines = content.split("\n")

        security_patterns = {
            "python": [
                ("exec(", "Dangerous exec usage"),
                ("eval(", "Dangerous eval usage"),
                ("pickle.loads", "Insecure deserialization"),
                ("subprocess.call", "Command injection risk"),
                ("os.system", "Command injection risk"),
            ],
            "javascript": [
                ("eval(", "Dangerous eval usage"),
                ("Function(", "Dynamic code execution"),
                ("innerHTML", "XSS risk"),
                ("document.write", "XSS risk"),
                ("setTimeout(string", "Code injection risk"),
            ],
            "sql": [
                ("SELECT *", "Potential SQL injection"),
                ("INSERT", "SQL injection risk"),
                ("UPDATE", "SQL injection risk"),
                ("DELETE", "SQL injection risk"),
                ("DROP", "Destructive operation"),
            ],
        }

        patterns = security_patterns.get(language, [])

        for i, line in enumerate(lines, 1):
            for pattern, issue in patterns:
                if pattern in line:
                    security_issues.append(
                        {"line": i, "issue": issue, "code": line.strip()}
                    )

        return {
            "security": {
                "issues_found": len(security_issues),
                "issues": security_issues[:5],  
                "risk_level": "High" if security_issues else "Low",
            }
        }

    def _generate_overall_analysis(
        self,
        file_analysis: List[Dict],
        language_counts: Dict,
        total_lines: int,
        total_size: int,
    ) -> str:
        """Generate overall analysis text"""
        if not file_analysis:
            return "No files to analyze"

        avg_lines = total_lines / len(file_analysis) if file_analysis else 0
        avg_size = total_size / len(file_analysis) if file_analysis else 0

        most_common_lang = max(
            language_counts.items(), key=lambda x: x[1], default=("unknown", 0)
        )

        total_security_issues = sum(
            f.get("security", {}).get("issues_found", 0) for f in file_analysis
        )

        analysis = f"""
# Project Analysis Report

## Summary
- **Total Files**: {len(file_analysis)}
- **Total Lines**: {total_lines:,}
- **Total Size**: {total_size:,} bytes
- **Average per File**: {avg_lines:.1f} lines, {avg_size:.1f} bytes
- **Most Common Language**: {most_common_lang[0]} ({most_common_lang[1]} files)

## Language Distribution
{self._format_language_distribution(language_counts)}

## Security Assessment
- **Total Issues Found**: {total_security_issues}
- **Risk Level**: {'⚠️ High' if total_security_issues > 5 else '✅ Low' if total_security_issues == 0 else '⚠️ Medium'}

## Recommendations
1. **Code Quality**: {self._get_code_quality_recommendation(file_analysis)}
2. **Security**: {self._get_security_recommendation(total_security_issues)}
3. **Maintainability**: {self._get_maintainability_recommendation(file_analysis)}
"""

        return analysis

    def _format_language_distribution(self, language_counts: Dict) -> str:
        """Format language distribution for report"""
        if not language_counts:
            return "No languages detected"

        total = sum(language_counts.values())
        lines = []
        for lang, count in sorted(
            language_counts.items(), key=lambda x: x[1], reverse=True
        ):
            percentage = (count / total) * 100
            lines.append(f"- {lang}: {count} files ({percentage:.1f}%)")

        return "\n".join(lines)

    def _get_code_quality_recommendation(self, file_analysis: List[Dict]) -> str:
        """Get code quality recommendation"""
        if not file_analysis:
            return "No files to analyze"

        comment_ratios = [f.get("comment_ratio", 0) for f in file_analysis]
        avg_comment_ratio = (
            sum(comment_ratios) / len(comment_ratios) if comment_ratios else 0
        )

        if avg_comment_ratio < 0.1:
            return "Consider adding more comments to improve code documentation"
        elif avg_comment_ratio > 0.3:
            return "Good comment density, maintain current level"
        else:
            return "Comment ratio is adequate"

    def _get_security_recommendation(self, total_issues: int) -> str:
        """Get security recommendation"""
        if total_issues == 0:
            return "No security issues found, good job!"
        elif total_issues <= 3:
            return f"Found {total_issues} minor security issues, review recommended"
        else:
            return f"Found {total_issues} security issues, immediate review required"

    def _get_maintainability_recommendation(self, file_analysis: List[Dict]) -> str:
        """Get maintainability recommendation"""
        if not file_analysis:
            return "No files to analyze"

        large_files = [f for f in file_analysis if f.get("line_count", 0) > 500]

        if large_files:
            return f"Consider splitting {len(large_files)} large files for better maintainability"
        else:
            return "File sizes are manageable"
