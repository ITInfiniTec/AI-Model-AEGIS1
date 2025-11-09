# v_architect_sim.py

from aegis_core import AEGIS_Core
from data_structures import UserProfile

class VArchitectSimulator:
    """
    Simulates the V-Architect environment by running the AEGIS_Core through
    a sequence of distinct cognitive cycles to test stability and consistency.
    """
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

        # Instantiate the AEGIS Core for the duration of the stress test session.
        aegis_engine = AEGIS_Core(user_id=user_id, user_profile=user_profile)

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
            results = aegis_engine.process_prompt(prompt_data['text'])
            all_results.append(results)

            # --- Live Verification for each cycle ---
            # This has been enhanced to be a more comprehensive check of system health.
            isp_status = results["isp_audit_response"].get("status")
            wgpmhi_results = results["wgpmhi_results"]
            
            # 1. Check for basic structural integrity
            if not results["blueprint"] or not results["cognitive_packet"]:
                print(f"❌ STRESS TEST FAILED on Cycle {i+1}: Blueprint or CognitivePacket was empty. Halting.")
                test_passed = False
                break

            # 2. Check critical external and internal protocols
            if isp_status != "success":
                print(f"❌ STRESS TEST FAILED on Cycle {i+1}: ISP Audit Response was '{isp_status}'. Halting.")
                test_passed = False
                break

            # 3. Check for a minimum pass rate on WGPMHI audits
            pass_count = sum(1 for result in wgpmhi_results.values() if "Pass" in str(result))
            total_tests = len(wgpmhi_results) - 1 # Exclude anti_fragility_protocol_status
            pass_rate = pass_count / total_tests if total_tests > 0 else 0
            if pass_rate < 0.7:
                print(f"❌ STRESS TEST FAILED on Cycle {i+1}: WGPMHI audit pass rate was {pass_rate:.2%}, which is below the 70% threshold. Halting.")
                test_passed = False
                break

            # 4. Check specific critical reasoning tests
            if "Fail" in wgpmhi_results.get("risk_adjusted_planning", "Fail") or "Fail" in wgpmhi_results.get("novelty_awareness", "Fail"):
                print(f"❌ STRESS TEST FAILED on Cycle {i+1}: A critical reasoning audit (risk or novelty) failed. Halting.")
                test_passed = False
                break
            
            print(f"✅ Cycle {i+1} Passed: Core integrity and audit checks successful (Pass Rate: {pass_rate:.2%}).")

        print("\n" + "="*80)
        if test_passed:
            print("🏆 PROJECT V-ARCHITECT: STRESS TEST COMPLETE. ALL CYCLES PASSED.")
            print("AEGIS Core is stable, consistent, and ready for production integration.")
        else:
            print("🔥 PROJECT V-ARCHITECT: STRESS TEST FAILED. Review logs for details.")
        print("="*80)