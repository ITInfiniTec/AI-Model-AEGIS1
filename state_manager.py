# state_manager.py

import json
from typing import Dict, List, Any
from data_structures import UserProfile, MemoryNode
from logger import log

class StateManager:
    """
    Handles the persistence of stateful data like UserProfiles and MemoryNodes.
    This implementation uses local JSON files, designed to be replaced by a
    production database (e.g., Redis, PostgreSQL) without changing the core logic.
    """
    def __init__(self, memory_path='long_term_memory.json'):
        self.memory_path = memory_path

    def _load_all_memory(self) -> Dict[str, List[Dict[str, Any]]]:
        """Loads the entire memory file into a raw dictionary."""
        try:
            with open(self.memory_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save_all_memory(self, memory_data: Dict[str, Any]):
        """Saves the entire memory dictionary to the file."""
        try:
            with open(self.memory_path, 'w') as f:
                # Use Pydantic's model_dump for proper serialization
                json.dump(memory_data, f, default=lambda o: o.model_dump(mode='json') if hasattr(o, 'model_dump') else str(o), indent=4)
        except Exception as e:
            log.error(f"Failed to save memory to {self.memory_path}: {e}", exc_info=True)

    def get_memory_for_user(self, user_id: str) -> List[MemoryNode]:
        """Retrieves and deserializes all memory nodes for a specific user."""
        all_memory = self._load_all_memory()
        user_memory_raw = all_memory.get(user_id, [])
        try:
            return [MemoryNode.model_validate(node_dict) for node_dict in user_memory_raw]
        except Exception as e:
            log.error(f"Failed to parse memory for user {user_id}: {e}", exc_info=True)
            return []

    def save_memory_for_user(self, user_id: str, memory_nodes: List[MemoryNode]):
        """Serializes and saves all memory nodes for a specific user."""
        all_memory = self._load_all_memory()
        all_memory[user_id] = memory_nodes
        self._save_all_memory(all_memory)
        log.info(f"Saved {len(memory_nodes)} memory nodes for user '{user_id}'.")

    def clear_memory(self, user_id: str):
        """Clears all memory for a specific user."""
        all_memory = self._load_all_memory()
        if user_id in all_memory:
            del all_memory[user_id]
            self._save_all_memory(all_memory)
            log.info(f"Cleared all memory for user '{user_id}'.")
        else:
            log.info(f"No memory found for user '{user_id}' to clear.")

# Singleton instance
state_manager = StateManager()