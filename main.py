```language /main.py
# main.py

import json
from datetime import datetime
from data_structures import UserProfile
from aegis_core import AEGIS_Core, CognitivePacket, MemoryNode

def main():
    """Initializes the AEGIS Core and runs a continuous interaction loop."""
    print("="*80)
    print("🚀 AEGIS CORE ONLINE. Awaiting command, Architect Josephis.")
    print("="*80)
    print("Type 'exit' or 'quit' to terminate the session.")

    # --- Session Initialization ---
    user_id = "user123"
    user_profile = UserProfile(
        user_id=user_id,
        values={"controversial_topics_approach": 0.2, "importance_of_accuracy": 0.9}
    )
    user_profile.passions = ["chess", "poker", "technology", "war tactics"]
    
    aegis_engine = AEGIS_Core()

    while True:
        print("\n" + "-"*80)
        prompt = input("PROMPT> ")

        if prompt.lower() in ["exit", "quit"]:
            print("Terminating session. AEGIS Core shutting down.")
            break
        elif prompt.lower() == "view_memory":
            print("\n--- LONG-TERM MEMORY (USER: user123) ---")
            memory = aegis_engine.noesis_triad.context_synthesizer.long_term_memory.get(user_id, [])
            if not memory:
                print("No memories found.")
            else:
                for i, node in enumerate(memory):
                    print(f"  Memory Node {i+1}:")
                    print(f"    - Node ID: {node.id}")
                    print(f"    - Timestamp: {node.timestamp}")
                    print(f"    - Performance Score: {node.performance_score}")
                    # Accessing the packet reference to get the original prompt
                    print(f"    - Prompt: '{node.packet_reference.intent['raw_prompt'][:70]}...'")
            continue
        elif prompt.lower() == "clear_memory":
            if user_id in aegis_engine.noesis_triad.context_synthesizer.long_term_memory:
                aegis_engine.noesis_triad.context_synthesizer.long_term_memory[user_id].clear()
                print("Long-term memory for the current session has been cleared.")
            else:
                print("No memory found for the current session to clear.")
            continue
        elif prompt.lower().startswith("view_node "):
            parts = prompt.split()
            if len(parts) < 2:
                print("Usage: view_node <node_id>")
                continue
            
            node_id_to_find = parts[1]
            memory = aegis_engine.noesis_triad.context_synthesizer.long_term_memory.get(user_id, [])
            found_node = None
            for node in memory:
                if node.id == node_id_to_find:
                    found_node = node
                    break
            
            if found_node:
                print(f"\n--- DETAILS FOR MEMORY NODE: {node_id_to_find} ---")
                print(found_node.packet_reference.__dict__)
            else:
                print(f"Memory Node with ID '{node_id_to_find}' not found in the current session.")
            continue
        elif prompt.lower() == "save_memory":
            memory_to_save = aegis_engine.noesis_triad.context_synthesizer.long_term_memory
            if not memory_to_save.get(user_id):
                print("No memory to save for the current session.")
                continue

            # Custom serializer to handle datetime and other custom objects
            def default_serializer(o):
                if isinstance(o, datetime):
                    return o.isoformat()
                if hasattr(o, '__dict__'):
                    return o.__dict__
                raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

            with open('memory_log.json', 'w') as f:
                json.dump(memory_to_save, f, default=default_serializer, indent=4)
            print(f"Successfully saved long-term memory to 'memory_log.json'.")
            continue
        elif prompt.lower() == "load_memory":
            try:
                with open('memory_log.json', 'r') as f:
                    loaded_memory_raw = json.load(f)
                
                reconstructed_memory = {}
                for user, nodes_raw in loaded_memory_raw.items():
                    reconstructed_nodes = []
                    for node_dict in nodes_raw:
                        # Reconstruct the CognitivePacket first
                        cp_dict = node_dict['packet_reference']
                        reconstructed_packet = CognitivePacket(
                            packet_id=cp_dict['packet_id'],
                            scenario=cp_dict['scenario'],
                            intent=cp_dict['intent'],
                            ethical_considerations=cp_dict['ethical_considerations'],
                            ideal_response=cp_dict['ideal_response'],
                            wgpmhi_results=cp_dict['wgpmhi_results'],
                            debug_report=cp_dict['debug_report']
                        )
                        # Reconstruct the MemoryNode
                        reconstructed_node = MemoryNode(
                            node_id=node_dict['id'],
                            timestamp=datetime.fromisoformat(node_dict['timestamp']),
                            core_intent_vector=node_dict['core_intent_vector'],
                            keywords=node_dict['keywords'],
                            performance_score=node_dict['performance_score'],
                            packet_reference=reconstructed_packet
                        )
                        reconstructed_nodes.append(reconstructed_node)
                    reconstructed_memory[user] = reconstructed_nodes
                aegis_engine.noesis_triad.context_synthesizer.long_term_memory = reconstructed_memory
                print("Successfully loaded long-term memory from 'memory_log.json'.")
            except FileNotFoundError:
                print("Error: 'memory_log.json' not found.")
            except Exception as e:
                print(f"An error occurred while loading memory: {e}")
            continue
        elif prompt.lower() == "view_config":
            print("\n--- CURRENT AEGIS CORE CONFIGURATION ---")
            config = config_loader.get_full_config()
            # Use the existing serializer to handle potential non-serializable objects
            print(json.dumps(config, indent=4))
            continue

        # Process the prompt through the entire AEGIS lifecycle
        results = aegis_engine.process_prompt(user_id, prompt, user_profile)

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
```