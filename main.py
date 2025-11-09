
# main.py

import json
import sys
from datetime import datetime
from data_structures import UserProfile, CognitivePacket, MemoryNode
from aegis_core import AEGIS_Core
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
    aegis_engine = AEGIS_Core(user_id, user_profile)
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


class CommandHandler:
    """Handles the registration and execution of CLI commands."""
    def __init__(self, aegis_engine):
        self.aegis_engine = aegis_engine
        self._commands = {
            "exit": self.exit_session,
            "quit": self.exit_session,
            "view_memory": self.view_memory,
            "clear_memory": self.clear_memory,
            "view_node": self.view_node,
            "save_memory": self.save_memory,
            "load_memory": self.load_memory,
            "view_config": self.view_config,
        }

    def is_command(self, name):
        return name in self._commands

    def execute(self, name, *args):
        """Executes a registered command."""
        try:
            self._commands[name](*args)
        except TypeError:
            print(f"Error: Invalid arguments for command '{name}'.")
        except Exception as e:
            print(f"An error occurred executing command '{name}': {e}")

    def exit_session(self):
        """Terminates the AEGIS Core session."""
        print("Terminating session. AEGIS Core shutting down.")
        sys.exit(0)

    def view_memory(self):
        """Displays a summary of the long-term memory."""
        print(f"\n--- LONG-TERM MEMORY (USER: {self.aegis_engine.user_id}) ---")
        memory = self.aegis_engine.get_memory()
        if not memory:
            print("No memories found.")
        else:
            for i, node in enumerate(memory):
                print(f"  Memory Node {i+1}:")
                print(f"    - Node ID: {node.id}")
                print(f"    - Timestamp: {node.timestamp.isoformat()}")
                print(f"    - Performance Score: {node.performance_score}")
                print(f"    - Prompt: '{node.packet_reference.intent['raw_prompt'][:70]}...'")

    def clear_memory(self):
        """Clears the long-term memory for the current session."""
        self.aegis_engine.clear_memory()
        print("Long-term memory for the current session has been cleared.")

    def view_node(self, node_id=None):
        """Displays the full details of a specific memory node."""
        if not node_id:
            print("Usage: view_node <node_id>")
            return
        
        node = self.aegis_engine.get_memory_node(node_id)
        if node:
            print(f"\n--- DETAILS FOR MEMORY NODE: {node_id} ---")
            # Use model_dump_json for a direct, pretty-printed JSON output.
            print(node.model_dump_json(indent=4))
        else:
            print(f"Memory Node with ID '{node_id}' not found in the current session.")

    def save_memory(self):
        """Saves the current session's memory to a file."""
        try:
            self.aegis_engine.save_memory_to_file('memory_log.json')
            print("Successfully saved long-term memory to 'memory_log.json'.")
        except TypeError as e:
            print(f"Error: Could not serialize memory to JSON. {e}")
        except Exception as e:
            print(f"An error occurred while saving memory: {e}")

    def load_memory(self):
        """Loads a session's memory from a file."""
        try:
            self.aegis_engine.load_memory_from_file('memory_log.json')
            print("Successfully loaded long-term memory from 'memory_log.json'.")
        except FileNotFoundError:
            print("Error: 'memory_log.json' not found.")
        except json.JSONDecodeError:
            print("Error: Could not decode 'memory_log.json'. The file may be corrupt or improperly formatted.")
        except Exception as e:
            print(f"An error occurred while loading memory: {e}")

    def view_config(self):
        """Displays the current system configuration."""
        print("\n--- CURRENT AEGIS CORE CONFIGURATION ---")
        config = self.aegis_engine.get_config()
        print(json.dumps(config, indent=4))

if __name__ == "__main__":
    main()
