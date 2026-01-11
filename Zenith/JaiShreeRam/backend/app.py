from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
import logging
from datetime import datetime
import traceback
# from utils.ollama_helpers import OllamaHelper
import os

# os.environ["OLLAMA_HOST"] = "localhost:11435"
load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

from services.llm_service import GeminiService #, OllamaService
from services.file_service import FileService
from models.schemas import (
    GenerateRequest,
    ExplainRequest,
    DebugRequest,
    OptimizeRequest,
    ChatRequest,
    TestRequest,
    AnalyzeRequest,
)

app = Flask(__name__)

cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
CORS(app, origins=cors_origins)

llm_service = GeminiService() # OllamaService()
file_service = FileService()
from services.rag_service import RAGService
rag_service = RAGService()
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Not found", "message": str(error)}), 404


@app.errorhandler(400)
def bad_request(error):
    return jsonify({"error": "Bad request", "message": str(error)}), 400


@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return (
        jsonify({"error": "Internal server error", "message": "Something went wrong"}),
        500,
    )


#Middleware
@app.before_request
def log_request_info():
    logger.info(f"Request: {request.method} {request.path}")
    if request.is_json:
        logger.debug(f"Request body: {request.get_json()}")


@app.after_request
def log_response_info(response):
    logger.info(f"Response: {response.status}")
    return response

@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify(
        {
            "status": "healthy",
            "service": "AI Code Assistant API",
            "version": "1.0.0",
            "timestamp": datetime.utcnow().isoformat(),
        }
    )

@app.route("/api/generate", methods=["POST"])
def generate_code():
    try:
        data = request.get_json()

        if not data or "prompt" not in data:
            return jsonify({"error": "Missing prompt field"}), 400
        prompt = data["prompt"]
        language = data.get("language", "python")
        context = data.get("context", "")
        logger.info(f"Generating code for prompt: {prompt[:50]}...")
        result = llm_service.generate_code(
            prompt=prompt, language=language, context=context
        )

        return jsonify(
            {
                "success": True,
                "code": result["code"],
                "explanation": result["explanation"],
                "language": language,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

    except Exception as e:
        logger.error(f"Error in generate_code: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/explain", methods=["POST"])
def explain_code():
    try:
        data = request.get_json()
        print(data)

        if not data or "code" not in data:
            return jsonify({"error": "Missing code field"}), 400

        code = data["code"]
        language = data.get("language", "auto")
        detail_level = data.get("detail_level", "comprehensive")

        logger.info(f"Explaining code of length: {len(code)}")

        explanation = llm_service.explain_code(
            code=code, language=language, detail_level=detail_level
        )

        return jsonify(
            {
                "success": True,
                "explanation": explanation,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

    except Exception as e:
        logger.error(f"Error in explain_code: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/debug", methods=["POST"])
def debug_code():
    try:
        data = request.get_json()

        if not data or "code" not in data:
            return jsonify({"error": "Missing code field"}), 400

        code = data["code"]
        language = data.get("language", "auto")
        error_message = data.get("error_message", "")

        logger.info(
            f"Debugging code with error: {error_message[:100] if error_message else 'No error provided'}"
        )

        result = llm_service.debug_code(
            code=code, language=language, error_message=error_message
        )

        return jsonify(
            {
                "success": True,
                "debugged_code": result["debugged_code"],
                "explanation": result["explanation"],
                "issues_found": result["issues_found"],
                "fixes_applied": result["fixes_applied"],
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

    except Exception as e:
        logger.error(f"Error in debug_code: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/optimize", methods=["POST"])
def optimize_code():
    try:
        data = request.get_json()

        if not data or "code" not in data:
            return jsonify({"error": "Missing code field"}), 400

        code = data["code"]
        language = data.get("language", "auto")
        optimization_type = data.get(
            "optimization_type", "performance"
        )  # performance, readability, memory

        logger.info(f"Optimizing code with type: {optimization_type}")

        result = llm_service.optimize_code(
            code=code, language=language, optimization_type=optimization_type
        )

        return jsonify(
            {
                "success": True,
                "optimized_code": result["optimized_code"],
                "explanation": result["explanation"],
                "improvements": result["improvements"],
                "before_metrics": result.get("before_metrics"),
                "after_metrics": result.get("after_metrics"),
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

    except Exception as e:
        logger.error(f"Error in optimize_code: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()

        if not data or "message" not in data:
            return jsonify({"error": "Missing message field"}), 400

        message = data["message"]
        history = data.get("history", [])
        context = data.get("context", {})
        
        # Check for RAG request
        use_rag = data.get("use_rag", False) or context.get("use_rag", False)
        if use_rag:
            logger.info("Using RAG for chat")
            
            # Reframe query for better retrieval
            search_query = llm_service.reframe_query(message, history)
            logger.info(f"Reframed RAG query: {search_query}")
            
            rag_result = rag_service.query_with_context(search_query)
            context["rag_context"] = rag_result["context"]
            context["rag_sources"] = rag_result["sources"]
            context["file_tree"] = rag_result.get("file_tree", "")

        logger.info(f"Chat message: {message[:100]}...")

        response = llm_service.chat(message=message, history=history, context=context)

        return jsonify(
            {
                "success": True,
                "response": response["response"],
                "history": response["history"],
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

    except Exception as e:
        logger.error(f"Error in chat: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/test", methods=["POST"])
def write_tests():
    try:
        data = request.get_json()

        if not data or "code" not in data:
            return jsonify({"error": "Missing code field"}), 400

        code = data["code"]
        language = data.get("language", "auto")
        test_framework = data.get("test_framework", "")

        logger.info(f"Writing tests for {language} code")

        result = llm_service.write_tests(
            code=code, language=language, test_framework=test_framework
        )

        return jsonify(
            {
                "success": True,
                "tests": result["tests"],
                "test_explanation": result["test_explanation"],
                "coverage": result["coverage"],
                "test_cases": result["test_cases"],
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

    except Exception as e:
        logger.error(f"Error in write_tests: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/analyze", methods=["POST"])
def analyze_code():
    try:
        data = request.get_json()

        if not data or "code" not in data:
            return jsonify({"error": "Missing code field"}), 400

        code = data["code"]
        language = data.get("language", "auto")
        analysis_type = data.get("analysis_type", "comprehensive")

        logger.info(f"Analyzing code with type: {analysis_type}")

        result = llm_service.analyze_code(
            code=code, language=language, analysis_type=analysis_type
        )

        return jsonify(
            {
                "success": True,
                "analysis": result["analysis"],
                "complexity": result["complexity"],
                "quality_score": result["quality_score"],
                "issues": result["issues"],
                "recommendations": result["recommendations"],
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

    except Exception as e:
        logger.error(f"Error in analyze_code: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/files/analyze", methods=["POST"])
def analyze_files():
    try:
        if "files" not in request.files:
            return jsonify({"error": "No files provided"}), 400

        files = request.files.getlist("files")
        analysis_type = request.form.get("analysis_type", "structure")

        logger.info(f"Analyzing {len(files)} files")

        result = file_service.analyze_files(files, analysis_type)

        return jsonify(
            {
                "success": True,
                "analysis": result["analysis"],
                "file_count": result["file_count"],
                "language_distribution": result["language_distribution"],
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

    except Exception as e:
        logger.error(f"Error in analyze_files: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/convert", methods=["POST"])
def convert_code():
    try:
        data = request.get_json()

        if not data or "code" not in data or "target_language" not in data:
            return jsonify({"error": "Missing required fields"}), 400

        code = data["code"]
        source_language = data.get("source_language", "auto")
        target_language = data["target_language"]

        logger.info(f"Converting code from {source_language} to {target_language}")

        result = llm_service.convert_code(
            code=code, source_language=source_language, target_language=target_language
        )

        return jsonify(
            {
                "success": True,
                "converted_code": result["converted_code"],
                "explanation": result["explanation"],
                "compatibility_notes": result["compatibility_notes"],
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

    except Exception as e:
        logger.error(f"Error in convert_code: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/document", methods=["POST"])
def document_code():
    try:
        data = request.get_json()

        if not data or "code" not in data:
            return jsonify({"error": "Missing code field"}), 400

        code = data["code"]
        language = data.get("language", "auto")
        documentation_style = data.get("documentation_style", "comprehensive")

        logger.info(f"Documenting code with style: {documentation_style}")

        result = llm_service.document_code(
            code=code, language=language, documentation_style=documentation_style
        )

        return jsonify(
            {
                "success": True,
                "documented_code": result["documented_code"],
                "documentation": result["documentation"],
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

    except Exception as e:
        logger.error(f"Error in document_code: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/rag/index", methods=["POST"])
def index_codebase():
    try:
        data = request.get_json()
        path = data.get("path")
        
        if not path:
            return jsonify({"error": "Missing path field"}), 400
            
        logger.info(f"Indexing codebase at: {path}")
        result = rag_service.index_codebase(path)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in index_codebase: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/rag/reset", methods=["POST"])
def reset_rag_index():
    try:
        logger.info("Resetting RAG index")
        result = rag_service.reset_index()
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in reset_rag_index: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"success": False, "error": str(e)}), 500


# @app.route("/api/ollama/health", methods=["GET"])
# def ollama_health():
#     """Check Ollama health and available models"""
#     try:
#         models = OllamaHelper.list_models()

#         return jsonify(
#             {
#                 "success": True,
#                 "ollama_running": len(models) > 0,
#                 "available_models": [m["name"] for m in models],
#                 "model_count": len(models),
#             }
#         )
#     except Exception as e:
#         return jsonify(
#             {
#                 "success": False,
#                 "ollama_running": False,
#                 "error": str(e),
#                 "message": "Ollama is not running. Please start it with: ollama serve",
#             }
#         )


# @app.route("/api/ollama/pull", methods=["POST"])
# def pull_model():
#     """Pull a model from Ollama"""
#     try:
#         data = request.get_json()
#         model_name = data.get("model", "codellama:7b")

#         success = OllamaHelper.pull_model(model_name)

#         return jsonify(
#             {
#                 "success": success,
#                 "model": model_name,
#                 "message": (
#                     f"Started pulling model: {model_name}"
#                     if success
#                     else "Failed to pull model"
#                 ),
#             }
#         )
#     except Exception as e:
#         logger.error(f"Error pulling model: {str(e)}")
#         return (
#             jsonify(
#                 {"success": False, "error": str(e), "message": "Failed to pull model"}
#             ),
#             500,
#         )


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "1") == "1"

    logger.info(f"Starting AI Code Assistant API on port {port}")
    logger.info(f"Debug mode: {debug}")
    logger.info(f"CORS origins: {cors_origins}")

    app.run(host="0.0.0.0", port=port, debug=debug)
