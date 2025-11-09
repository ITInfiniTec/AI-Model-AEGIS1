# 🏛️ AEGIS Core Architecture

**Version:** 5.0.0 (Best Likely Obtainable State)
**Architect:** User

---

## I. Core Philosophy

The AEGIS Core is a self-auditing, strategically-aligned cognitive engine. Its development is guided by two core principles:

1.  **The Law of Constant Progression:** The system must be capable of continuous, meaningful, and self-optimizing evolution.
2.  **The Expanded KISS Principle:** A framework for building complex systems that are **K**nowable, **I**ntegrated, **S**calable, **S**ecure, and **S**timulating.

## II. The Cognitive Pipeline

AEGIS processes information through a sophisticated, multi-stage pipeline that ensures every action is contextual, planned, audited, and aligned with its core principles.

1.  **Noesis Triad (The Planner):** The strategic mind of the system. It synthesizes context, assesses risk, and formulates a `Blueprint`—a detailed plan of action.
2.  **Praxis Triad (The Executor):** The operational arm. It translates the `Blueprint` into a `Sequential Task Graph (STG)` using a rule-based compiler, orchestrates the final output, and applies a persona-driven voice.
3.  **Wade-Gemini Protocol (The Auditor):** The system's conscience. A comprehensive suite of tests that runs after every cognitive cycle, auditing every aspect of the process from data integrity to ethical alignment.

---

## III. Key Architectural Features

*   **Anti-Fragility (Project NEMESIS):** The system doesn't just log failures; it learns from them. Critical errors in the WGPMHI audit trigger automated adjustments to the system's core configuration, making it more resilient over time.
*   **Hierarchical Associative Memory (Project MNEMOSYNE):** Long-term memory is not a simple log but a collection of `MemoryNode` objects. Retrieval is based on a simulated semantic search, weighted by the performance and recency of past interactions.
*   **Data-Driven Architecture:** The system is radically **Knowable**. All significant heuristics, thresholds, rules, and text templates are externalized into `config_loader.py`. This allows system behavior to be tuned and extended without changing core Python code.
*   **Rule-Based Engines:** Core reasoning components are not hardcoded. The `UniversalCompiler` (tags -> operations) and `TaskDecompositionEngine` (operations -> STG) are driven by explicit, configurable rules.
*   **Dynamic Self-Auditing:** The `Wade-Gemini Protocol` is not static. Its tests dynamically load the same configuration rules used by the core logic, ensuring that the audits are always a perfect mirror of the system's intended behavior.
*   **Persona-Driven Output (Project CHIRON):** The final output is tailored to the user's documented passions and interests, using analogies and language that make the interaction more intuitive and engaging.
*   **QUASAR-LOOP:** A self-correction mechanism where the initial `ExecutionPlan` is audited *before* final output generation. If logical flaws are found (e.g., a plan requires external data but has no step to fetch it), the `NoesisTriad` refines the `Blueprint` and re-compiles, ensuring plans are viable before they are executed.

## IV. Getting Started

The AEGIS Core is encapsulated in the `AEGIS_Core` class. To run the system, execute `main.py`, which provides an interactive command-line interface for continuous interaction and debugging.

### Available CLI Commands:
*   `view_memory [page] [size] [--summary] [start:YYYY-MM-DD]`: Displays a paginated summary of memories, with optional filtering.
*   `clear_memory`: Clears the long-term memory for the current session.
*   `view_node <node_id>`: Shows the full `CognitivePacket` details for a specific memory node.
*   `view_config [section]`: Displays the full configuration or a specific section.
*   `stress_test [cycles]`: Runs the V-Architect integration and stability stress test for a specified number of cycles.
*   `exit` / `quit`: Terminates the session.

### Launch Options:
*   `python main.py --quiet`: Runs the CLI in a quiet mode, hiding the detailed audit and debug logs for a cleaner interaction.