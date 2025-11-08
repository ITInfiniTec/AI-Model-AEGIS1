# api.py

from flask import Flask, request, jsonify
from pydantic import ValidationError
from aegis_core import AEGIS_Core
from data_structures import UserProfile
from api_models import ProcessRequest

app = Flask(__name__)

# Instantiate the AEGIS Core engine once, when the application starts.
# In a production environment, the AEGIS_Core would be stateless,
# connecting to a StateManager for all data persistence.
aegis_engine = AEGIS_Core()

@app.route('/process', methods=['POST'])
def process_prompt():
    """
    The primary API endpoint for interacting with the AEGIS Core.
    Validates the request against the ProcessRequest Pydantic model.
    """
    try:
        # 1. Validate the incoming request data using Pydantic
        request_data = ProcessRequest.model_validate(request.get_json())
    except ValidationError as e:
        return jsonify({"error": "Invalid request format", "details": e.errors()}), 400
    except Exception:
        return jsonify({"error": "Invalid JSON payload"}), 400

    try:
        # 2. Reconstruct the UserProfile object from the validated data
        user_profile = UserProfile(
            user_id=request_data.user_id,
            values=request_data.user_profile_data.values.model_dump()
        )
        user_profile.passions = request_data.user_profile_data.passions

        # 3. Process the prompt through the AEGIS Core
        results = aegis_engine.process_prompt(request_data.user_id, request_data.prompt, user_profile)

        # 4. Return a structured, production-ready response
        return jsonify({
            "status": "success",
            "packet_id": results["cognitive_packet"].packet_id,
            "response": results["final_output"]
        })

    except Exception as e:
        # In a production environment, log the full error.
        return jsonify({"error": "An internal error occurred.", "details": str(e)}), 500

if __name__ == '__main__':
    # For development/testing purposes. Use a production WSGI server for deployment.
    app.run(debug=True, port=5000)