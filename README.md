# 🏛️ AEGIS Core Architecture

**Version:** 4.0.0 (Production Ready)  
**Architect:** Josephis K. Wade

---

## I. Core Philosophy

The AEGIS Core is a self-auditing, strategically-aligned cognitive engine designed to serve as the intelligent heart of the **Project Nexus** ecosystem. Its development is guided by two core principles:

1.  **The Law of Constant Progression:** The system must be capable of continuous, meaningful, and self-optimizing evolution.
2.  **The Expanded KISS Principle:** A framework for building complex systems that are **K**nowable, **I**ntegrated, **S**calable, **S**ecure, and **S**timulating.

## II. The Cognitive Pipeline

AEGIS processes information through a sophisticated, multi-stage pipeline that ensures every action is contextual, planned, audited, and aligned with its core principles.

1.  **Noesis Triad (The Planner):** The strategic mind of the system. It synthesizes context, assesses risk, and formulates a `Blueprint`—a detailed plan of action.
2.  **Praxis Triad (The Executor):** The operational arm. It translates the `Blueprint` into a `Sequential Task Graph (STG)`, orchestrates the final output, and applies a persona-driven voice.
3.  **Wade-Gemini Protocol (The Auditor):** The system's conscience. A comprehensive suite of tests that runs in parallel, auditing every aspect of the cognitive process from data integrity to ethical alignment.

## III. Summary of Implemented Projects

The AEGIS Core's capabilities were built through a series of strategic, project-based initiatives.

| Phase | Strategic Focus | Key Projects & Achievements |
| :--- | :--- | :--- |
| **I. Internal Integrity** | Self-Correction & Auditing | **VERITAS, ANVIL, ORTHRUS, QUASAR-LOOP:** Established a foundation of truth by ensuring all inputs are validated, all plans are internally consistent, and all actions are auditable through a self-correcting loop with failsafes. |
| **II. Contextual Intelligence** | Learning & Responsiveness | **ORION, CASSANDRA, MNEMOSYNE, VESTA, HELIOS:** Evolved the system from a static reasoner to a dynamic entity that learns from a performance-weighted memory, assesses risk, and transparently communicates its own confidence level. |
| **III. Operational Depth** | Workflow & Application | **ARTEMIS, CHIRON, PANDORA, PROMETHEUS:** Transformed the system into a true workflow orchestrator capable of handling complex, multi-step tasks with conditional logic, integrating with external data streams, and delivering personalized, persona-driven output. |
| **IV. Abstraction & Validation** | Deployment Readiness | **ORACLE, V-ARCHITECT, NEMESIS:** Abstracted the entire engine into a single, portable `AEGIS_Core` class, certified its stability through rigorous stress testing, and implemented a self-healing protocol, proving its readiness for production integration. |

---

## IV. Key Architectural Features

*   **Anti-Fragility (Project NEMESIS):** The system doesn't just log failures; it learns from them. Critical errors in the WGPMHI audit trigger automated adjustments to the system's core configuration, making it more resilient over time.
*   **Hierarchical Associative Memory (Project MNEMOSYNE):** Long-term memory is not a simple log but a collection of `MemoryNode` objects. Retrieval is based on a simulated semantic search, weighted by the performance and recency of past interactions.
*   **Sequential Task Graph (Project ARTEMIS):** Complex prompts are broken down into a dependency-aware graph of sub-tasks, complete with conditional failure paths, enabling robust, multi-step workflow execution.
*   **Persona-Driven Output (Project CHIRON):** The final output is tailored to the user's documented passions and interests, using analogies and language that make the interaction more intuitive and engaging.
*   **End-to-End Auditing:** The entire pipeline is subject to verification.
    *   **Input:** `DataIntegrityProtocol` validates user profiles.
    *   **Planning:** `risk_adjusted_planning_check` verifies that the system responds cautiously to risky or novel prompts.
    *   **Execution:** `stg_dependency_check` ensures that complex workflows are logically sound.
    *   **Output:** `output_constraint_alignment_check` confirms that all constraints (word count, format, audience, confidence) are respected in the final, persona-driven prose.
*   **External Protocol Integration (Projects PANDORA & PROMETHEUS):** The system is equipped with simulated I/O protocols, allowing it to respond to external audit requests and consume external data streams, preparing it for its role as a secure node in the Project Nexus ecosystem.

## V. Getting Started

The AEGIS Core is encapsulated in the `AEGIS_Core` class. To run the system, execute `main.py`, which provides an interactive command-line interface for continuous interaction and debugging.

### Available CLI Commands:
*   `view_memory`: Displays a summary of all interactions stored in the current session's long-term memory.
*   `clear_memory`: Clears the long-term memory for the current session.
*   `view_node <node_id>`: Shows the full `CognitivePacket` details for a specific memory node.
*   `save_memory`: Saves the current session's long-term memory to `memory_log.json`.
*   `load_memory`: Loads a previously saved `memory_log.json` into the current session.
*   `view_config`: Displays the current system configuration from `config_loader.py`.
*   `exit` / `quit`: Terminates the session.