import pytest
import json
from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_health_check(client):
    """Test health check endpoint"""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["status"] == "healthy"


def test_generate_code(client):
    """Test code generation endpoint"""
    data = {"prompt": "Write a function to calculate factorial", "language": "python"}

    response = client.post(
        "/api/generate", data=json.dumps(data), content_type="application/json"
    )

    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["success"] == True
    assert "code" in data


def test_explain_code(client):
    """Test code explanation endpoint"""
    code = """
def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)
"""

    data = {"code": code, "language": "python"}

    response = client.post(
        "/api/explain", data=json.dumps(data), content_type="application/json"
    )

    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["success"] == True
    assert "explanation" in data


def test_debug_code(client):
    """Test code debugging endpoint"""
    buggy_code = """
def divide(a, b):
    return a / b
"""

    data = {"code": buggy_code, "language": "python"}

    response = client.post(
        "/api/debug", data=json.dumps(data), content_type="application/json"
    )

    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["success"] == True
    assert "debugged_code" in data


def test_optimize_code(client):
    """Test code optimization endpoint"""
    code = """
def sum_numbers(numbers):
    total = 0
    for num in numbers:
        total += num
    return total
"""

    data = {"code": code, "language": "python"}

    response = client.post(
        "/api/optimize", data=json.dumps(data), content_type="application/json"
    )

    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["success"] == True
    assert "optimized_code" in data


def test_chat_endpoint(client):
    """Test chat endpoint"""
    data = {"message": "How do I write a Python function?", "history": []}

    response = client.post(
        "/api/chat", data=json.dumps(data), content_type="application/json"
    )

    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["success"] == True
    assert "response" in data


def test_missing_prompt(client):
    """Test missing required field"""
    data = {"language": "python"}

    response = client.post(
        "/api/generate", data=json.dumps(data), content_type="application/json"
    )

    assert response.status_code == 400
    data = json.loads(response.data)
    assert "error" in data


def test_invalid_json(client):
    """Test invalid JSON"""
    response = client.post(
        "/api/generate", data="invalid json", content_type="application/json"
    )

    assert response.status_code == 400


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
