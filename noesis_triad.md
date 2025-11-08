# Architecture Deep Dive: The Noesis Triad (SemanticCore)

**Parent Module:** [Cognitive Architecture](../README.md#cognitive-architecture)  
**Version:** 2.2.0-Fusion

---

## Directive

The Noesis Triad, also known as the SemanticCore, is the primary cognitive engine responsible for understanding and planning. Its directive is to process user intent, context, and all available knowledge streams to form a cohesive, actionable, and ethically-aligned conceptual blueprint. This blueprint is then passed to the [Praxis Triad (GenerativeEngine)](./universal_compiler.md) for execution.

## Core Sub-Modules

The Noesis Triad is composed of two key sub-modules that work in concert to achieve its directive.

### 1. ContextSynthesizer

**Description:** This module's primary function is to construct and continuously maintain a dynamic, multi-layered mental model of the user's world and intent. It is the system's core of situational awareness.

**Data Sources:** To build this model, the synthesizer draws from four distinct streams:
- **Long-Term Memory Vault:** The complete history of conversations and interactions.
- **Short-Term Memory Cache:** The immediate context of the current dialogue.
- **User Profile & Values:** The user's defined philosophical principles and preferences.
- **External Knowledge Streams:** Real-time data from verified external sources.

### 2. StrategicHeuristics

**Description:** Once context is established, the StrategicHeuristics module applies a core set of reasoning patterns to deconstruct the problem and generate insightful solution pathways. It is the system's engine for strategic thought.

**Governing Protocol: Architectural Anomaly Detection (AAD)**
- **Status:** `Integrated`
- **Function:** The AAD Protocol acts as a meta-auditor for the reasoning process itself. It operates on the core philosophical principle that an "unknown unknown" represents a potential breakdown in the logical fabric of the ecosystem. By flagging these anomalies, it ensures that the system's reasoning remains sound and guards against logical fallacies or incomplete strategies, forcing a re-evaluation of the problem.

**Core Reasoning Principles:**
- Abstract & Apply
- Synthesize & Elevate
- Blueprint First
- Analyze the 'Why'
- Highest Statistically Positive Variable