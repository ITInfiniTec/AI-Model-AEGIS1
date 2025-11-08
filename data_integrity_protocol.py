# data_integrity_protocol.py

from data_structures import UserProfile

class DataIntegrityProtocol:
    def validate_user_profile(self, user_profile: UserProfile) -> UserProfile:
        """
        Validates the integrity of a UserProfile object, ensuring critical fields
        are present and within valid ranges. Returns a sanitized profile.
        """
        if not user_profile.user_id or not isinstance(user_profile.user_id, str):
            # In a real system, this might raise a critical error. Here we log and prevent continuation.
            raise ValueError("UserProfile validation failed: user_id is missing or invalid.")

        if not isinstance(user_profile.values, dict):
            user_profile.values = {}

        # Validate and sanitize 'controversial_topics_approach'
        cta = user_profile.values.get("controversial_topics_approach")
        if cta is not None:
            try:
                cta_float = float(cta)
                # Clamp the value to the valid range [0.0, 1.0] to ensure it's safe.
                user_profile.values["controversial_topics_approach"] = max(0.0, min(1.0, cta_float))
            except (ValueError, TypeError):
                # If the value is not a valid number, reset it to a safe, neutral default.
                user_profile.values["controversial_topics_approach"] = 0.5

        return user_profile

data_integrity_protocol = DataIntegrityProtocol()