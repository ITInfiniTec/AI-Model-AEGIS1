# test_utils.py

from data_structures import UserProfile

def create_architect_profile() -> UserProfile:
    """
    Creates and returns the default UserProfile for 'The Architect' (user123).
    This centralizes the test user definition to avoid duplication.
    """
    user_id = "user123"
    user_profile = UserProfile(user_id=user_id, values={"controversial_topics_approach": 0.2, "importance_of_accuracy": 0.9})
    user_profile.passions = ["chess", "poker", "technology", "war tactics"]
    return user_profile