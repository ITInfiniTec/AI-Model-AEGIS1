
# main.py

from data_structures import UserProfile
from aegis_core import AEGIS_Core, CommandHandler
def main():
    """Initializes the AEGIS Core and runs a continuous interaction loop."""
    print("="*80)
    print("🚀 AEGIS CORE ONLINE. Awaiting command, Architect User.")
    print("="*80)
    print("Type 'exit' or 'quit' to terminate the session.")

    # --- Session Initialization ---
    user_id = "user123"
    user_profile = UserProfile(
        user_id=user_id,
        values={"controversial_topics_approach": 0.2, "importance_of_accuracy": 0.9}
    )
    user_profile.passions = ["chess", "poker", "technology", "war tactics"]
    
    # --- Command Handler Setup ---
    # This decouples the command logic from the main loop.
    aegis_engine = AEGIS_Core(user_id=user_id, user_profile=user_profile)
    command_handler = CommandHandler(aegis_engine)

    while True:
        print("\n" + "-"*80)
        prompt = input("PROMPT> ")
        command_parts = prompt.lower().split()
        command_name = command_parts[0]
        args = command_parts[1:]

        if command_handler.is_command(command_name):
            command_handler.execute(command_name, *args)
            continue

        # Process the prompt through the entire AEGIS lifecycle
        results = aegis_engine.process_prompt(prompt)

        # --- Output Display ---
        print("\n" + "="*25 + " FINAL ORCHESTRATED OUTPUT " + "="*25)
        print(results["final_output"])
        print("="*80)

        print("\n--- WGPMHI AUDIT RESULTS ---")
        for test, result in results["wgpmhi_results"].items():
            print(f"  - {test}: {result}")

        print("\n--- ORTHRUS DEBUG REPORT ---")
        print(results["orthrus_debug_report"])
        print("-" * 80)

if __name__ == "__main__":
    main()
