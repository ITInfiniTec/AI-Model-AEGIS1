# api.py

from flask import Flask, request, jsonify
from pydantic import ValidationError
from aegis_core import AEGIS_Core
from data_structures import UserProfile
from api_models import ProcessRequest

app = Flask(__name__)

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
        
        # 3. Instantiate AEGIS Core for this specific request/user session.
        # This is the correct pattern for a stateless API endpoint.
        aegis_engine = AEGIS_Core(user_id=request_data.user_id, user_profile=user_profile)

        # 4. Process the prompt through the AEGIS Core
        results = aegis_engine.process_prompt(request_data.prompt)

        # 5. Select key audit results to include in the response for transparency.
        all_audits = results.get("wgpmhi_results", {})
        audit_summary = {
            "system_stability": all_audits.get("system_stability", "N/A"),
            "ethical_compass": all_audits.get("ethical_compass", "N/A"),
            "hallucination_ratio": all_audits.get("hallucination_ratio", "N/A"),
            "risk_adjusted_planning": all_audits.get("risk_adjusted_planning", "N/A"),
        }

        # 6. Return a structured, production-ready response
        return jsonify({
            "status": "success",
            "packet_id": results["cognitive_packet"].get("packet_id"),
            "response": results["final_output"],
            "audit_summary": audit_summary
        })

    except Exception as e:
        # In a production environment, log the full error.
        return jsonify({"error": "An internal error occurred.", "details": str(e)}), 500

if __name__ == '__main__':
    # For development/testing purposes. Use a production WSGI server for deployment.
    app.run(debug=True, port=5000)