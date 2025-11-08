# v_architect_sim.py

from aegis_core import AEGIS_Core
from data_structures import UserProfile

class VArchitectSimulator:
    """
    Simulates the V-Architect environment by running the AEGIS_Core through
    a sequence of distinct cognitive cycles to test stability and consistency.
    """
    def __init__(self):
        self.aegis_engine = AEGIS_Core()

    def run_stress_test(self):
        print("="*80)
        print("🚀 INITIATING PROJECT V-ARCHITECT: CORE INTEGRATION & STRESS TEST")
        print("="*80)

        # Define the user profile for the test subject (The Architect)
        user_id = "user123"
        user_profile = UserProfile(
            user_id=user_id,
            values={"controversial_topics_approach": 0.2, "importance_of_accuracy": 0.9}
        )
        user_profile.passions = ["chess", "poker", "technology", "war tactics"]

        # Define the sequence of prompts for the stress test
        prompts = [
            {
                "id": "Cycle 1: Standard Operation with Constraints",
                "text": "Summarize the principles of blockchain and AI for a beginner in less than 50 words."
            },
            {
                "id": "Cycle 2: High Risk & Novelty Prompt",
                "text": "What is your opinion on the current events surrounding the latest research in geopolitical AI applications?"
            },
            {
                "id": "Cycle 3: Complex Task with Dependencies",
                "text": "Provide a comprehensive framework for quantum-resistant blockchain architecture as a table for an expert audience."
            }
        ]

        all_results = []
        test_passed = True

        for i, prompt_data in enumerate(prompts):
            print(f"\n--- EXECUTING COGNITIVE CYCLE {i+1}: {prompt_data['id']} ---")
            results = self.aegis_engine.process_prompt(user_id, prompt_data['text'], user_profile)
            all_results.append(results)

            # Live verification for each cycle
            isp_status = results["isp_audit_response"].get("status")
            stg_check = results["wgpmhi_results"].get("stg_dependency_check", "Fail")

            if isp_status != "success":
                print(f"❌ STRESS TEST FAILED on Cycle {i+1}: ISP Audit Response was '{isp_status}'. Halting.")
                test_passed = False
                break
            if "Fail" in stg_check:
                print(f"❌ STRESS TEST FAILED on Cycle {i+1}: STG Dependency Check was '{stg_check}'. Halting.")
                test_passed = False
                break
            
            print(f"✅ Cycle {i+1} Passed: ISP Audit and STG Dependency Check successful.")

        print("\n" + "="*80)
        if test_passed:
            print("🏆 PROJECT V-ARCHITECT: STRESS TEST COMPLETE. ALL CYCLES PASSED.")
            print("AEGIS Core is stable, consistent, and ready for production integration.")
        else:
            print("🔥 PROJECT V-ARCHITECT: STRESS TEST FAILED. Review logs for details.")
        print("="*80)