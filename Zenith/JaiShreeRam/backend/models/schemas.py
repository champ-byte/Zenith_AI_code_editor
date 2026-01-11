from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


class Language(str, Enum):
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    CPP = "cpp"
    CSHARP = "csharp"
    GO = "go"
    RUST = "rust"
    PHP = "php"
    RUBY = "ruby"
    SWIFT = "swift"
    KOTLIN = "kotlin"
    HTML = "html"
    CSS = "css"
    SQL = "sql"
    BASH = "bash"
    POWERSHELL = "powershell"


class DetailLevel(str, Enum):
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    COMPREHENSIVE = "comprehensive"


class OptimizationType(str, Enum):
    PERFORMANCE = "performance"
    READABILITY = "readability"
    MEMORY = "memory"
    SECURITY = "security"


class AnalysisType(str, Enum):
    BASIC = "basic"
    COMPREHENSIVE = "comprehensive"
    SECURITY = "security"
    PERFORMANCE = "performance"


class DocumentationStyle(str, Enum):
    BASIC = "basic"
    COMPREHENSIVE = "comprehensive"
    JAVADOC = "javadoc"
    GO_DOC = "go_doc"
    PYTHON_DOC = "python_doc"


class Message(BaseModel):
    role: str
    content: str


class GenerateRequest(BaseModel):
    prompt: str
    language: Language = Field(default=Language.PYTHON)
    context: Optional[str] = None
    requirements: Optional[List[str]] = None
    constraints: Optional[List[str]] = None


class ExplainRequest(BaseModel):
    code: str
    language: Optional[Language] = None
    detail_level: DetailLevel = Field(default=DetailLevel.COMPREHENSIVE)


class DebugRequest(BaseModel):
    code: str
    language: Optional[Language] = None
    error_message: Optional[str] = None
    expected_behavior: Optional[str] = None


class OptimizeRequest(BaseModel):
    code: str
    language: Optional[Language] = None
    optimization_type: OptimizationType = Field(default=OptimizationType.PERFORMANCE)
    constraints: Optional[List[str]] = None


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Message]] = None
    context: Optional[Dict[str, Any]] = None
    language: Optional[Language] = None


class TestRequest(BaseModel):
    code: str
    language: Optional[Language] = None
    test_framework: Optional[str] = None
    test_type: Optional[str] = "unit"


class AnalyzeRequest(BaseModel):
    code: str
    language: Optional[Language] = None
    analysis_type: AnalysisType = Field(default=AnalysisType.COMPREHENSIVE)


class ConvertRequest(BaseModel):
    code: str
    source_language: Optional[Language] = None
    target_language: Language
    preserve_comments: bool = Field(default=True)


class DocumentRequest(BaseModel):
    code: str
    language: Optional[Language] = None
    documentation_style: DocumentationStyle = Field(
        default=DocumentationStyle.COMPREHENSIVE
    )


class FileAnalysisRequest(BaseModel):
    files: List[str]  # Base64 encoded files or file paths
    analysis_type: AnalysisType = Field(default=AnalysisType.BASIC)


# Response models
class BaseResponse(BaseModel):
    success: bool
    timestamp: str
    error: Optional[str] = None


class GenerateResponse(BaseResponse):
    code: str
    explanation: str
    language: str


class ExplainResponse(BaseResponse):
    explanation: str
    key_points: List[str]
    complexity_analysis: Optional[Dict[str, Any]] = None


class DebugResponse(BaseResponse):
    debugged_code: str
    explanation: str
    issues_found: List[str]
    fixes_applied: List[str]


class OptimizeResponse(BaseResponse):
    optimized_code: str
    explanation: str
    improvements: List[str]
    before_metrics: Optional[Dict[str, Any]] = None
    after_metrics: Optional[Dict[str, Any]] = None


class ChatResponse(BaseResponse):
    response: str
    history: List[Message]
    suggestions: Optional[List[str]] = None


class TestResponse(BaseResponse):
    tests: str
    test_explanation: str
    coverage: Optional[float] = None
    test_cases: List[str]


class AnalyzeResponse(BaseResponse):
    analysis: str
    complexity: Dict[str, Any]
    quality_score: float
    issues: List[str]
    recommendations: List[str]


class ConvertResponse(BaseResponse):
    converted_code: str
    explanation: str
    compatibility_notes: List[str]


class DocumentResponse(BaseResponse):
    documented_code: str
    documentation: str
    summary: Optional[str] = None


class FileAnalysisResponse(BaseResponse):
    analysis: str
    file_count: int
    language_distribution: Dict[str, int]
    issues_by_file: Dict[str, List[str]]
    recommendations: List[str]
