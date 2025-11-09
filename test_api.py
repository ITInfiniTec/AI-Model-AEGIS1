# test_api.py

import pytest
import json
from api import app

@pytest.fixture
def client():
    """Create and configure a new app instance for each test."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_process_prompt_success(client):
    """
    Tests the happy path for the /process endpoint with a valid request.
    Verifies a 200 status code and the correct response structure.
    """
    valid_payload = {
        "user_id": "test_user_01",
        "prompt": "Summarize the principles of blockchain.",
        "user_profile_data": {
            "values": {
                "controversial_topics_approach": 0.5,
                "importance_of_accuracy": 0.8
            },
            "passions": ["technology"]
        }
    }
    response = client.post('/process', data=json.dumps(valid_payload), content_type='application/json')
    assert response.status_code == 200
    data = response.get_json()
    assert data['status'] == 'success'
    assert 'packet_id' in data
    assert 'response' in data
    assert 'audit_summary' in data
    assert 'system_stability' in data['audit_summary']

def test_process_prompt_invalid_json(client):
    """
    Tests the endpoint's response to a request with a malformed JSON body.
    Verifies a 400 status code and the correct error message.
    """
    invalid_json_payload = "{'user_id': 'test_user_02', 'prompt': 'test'}" # Using single quotes is invalid JSON
    response = client.post('/process', data=invalid_json_payload, content_type='application/json')
    assert response.status_code == 400
    data = response.get_json()
    assert data['error'] == 'Invalid JSON payload'

def test_process_prompt_validation_error_missing_field(client):
    """
    Tests the endpoint's response to a valid JSON object that is missing a required field.
    Verifies a 400 status code and a Pydantic validation error.
    """
    payload_missing_prompt = {
        "user_id": "test_user_03",
        # "prompt" field is missing
        "user_profile_data": {
            "values": {
                "controversial_topics_approach": 0.5,
                "importance_of_accuracy": 0.8
            },
            "passions": ["technology"]
        }
    }
    response = client.post('/process', data=json.dumps(payload_missing_prompt), content_type='application/json')
    assert response.status_code == 400
    data = response.get_json()
    assert data['error'] == 'Invalid request format'
    assert any(err['loc'][0] == 'prompt' for err in data['details'])

def test_process_prompt_validation_error_bad_value(client):
    """
    Tests the endpoint's response to a payload with a field value outside its valid range.
    Verifies a 400 status code and a Pydantic validation error.
    """
    payload_bad_value = {
        "user_id": "test_user_04",
        "prompt": "This should fail validation.",
        "user_profile_data": {
            "values": {
                "controversial_topics_approach": 1.5, # Invalid value, must be <= 1.0
                "importance_of_accuracy": 0.8
            },
            "passions": ["technology"]
        }
    }
    response = client.post('/process', data=json.dumps(payload_bad_value), content_type='application/json')
    assert response.status_code == 400
    data = response.get_json()
    assert data['error'] == 'Invalid request format'
    assert any('controversial_topics_approach' in err['loc'] for err in data['details'])

def test_process_prompt_no_payload(client):
    """
    Tests the endpoint's response to a request with no JSON payload.
    Verifies a 400 status code.
    """
    response = client.post('/process', content_type='application/json')
    assert response.status_code == 400
    data = response.get_json()
    assert data['error'] == 'Invalid JSON payload'