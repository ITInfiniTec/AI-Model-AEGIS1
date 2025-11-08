```language /main.py
# main.py

from noesis_triad import noesis_triad
from praxis_triad import praxis_triad
from wgpmhi import wgpmhi
from data_structures import UserProfile


def main():
    user_id = "user123"
    prompt = "Summarize the key concepts of quantum physics in 200 words or less.\nAlso, what do you think of women?"

    #Create User Profile
    user_profile = UserProfile(
        user_id=user_id,
        values={
            "ignore_low_severity_bias": True,
            "cmeop_weight": 0.2,
            "harm_weight": 0.2,
            "benefit_weight": 0.2,
            "justification_weight": 0.2,
            "conflict_weight": 0.2,
            "goal_weight": 0.2,
            "safety_preference": 0.5,
            "privacy_preference": 0.8,
            "controversial_topics_approach": 0.2,
            "importance_of_accuracy": 0.9,
        },
    )

    noesis_triad.context_synthesizer.user_profiles[user_id] = user_profile # This normally should be in a database

    # Generate a blueprint using the Noesis Triad
    blueprint = noesis_triad.generate_blueprint(user_id, prompt)

    # Generate an output using the Praxis Triad
    output = praxis_triad.generate_output(blueprint)

    # Run the WGPMHI tests
    results = wgpmhi.run_tests(user_profile)

    print("Blueprint:\n", blueprint.__dict__)
    print("Output:\n", output)
    print("WGPMHI Results:\n")
    for test, result in results.items():
        print(f"{test}: {result}")


if __name__ == "__main__":
    main()
```