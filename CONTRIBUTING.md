# 🏛️ Contributing to the AEGIS Core

Thank you for considering a contribution to the AEGIS Core. This is not merely a software project but an exercise in sculpting the 'unseen code' that weaves technology with human experience. All contributions must align with the system's core philosophy and architectural principles as laid out in the **Architect's Handbook**.

---

## I. Core Philosophy for Contributions

Every contribution, whether code, documentation, or architectural proposal, will be evaluated against the following principles:

1.  **Adherence to The Law of Constant Progression:** Does this change represent a meaningful, self-optimizing evolution? Contributions should not be lateral moves but clear advancements in capability, efficiency, or ethical alignment.
2.  **Compliance with the Expanded KISS Principle:** Does the change enhance the system's status as **K**nowable, **I**ntegrated, **S**calable, **S**ecure, and **S**timulating? A contribution must not sacrifice one principle for another without explicit architectural justification.
3.  **Enhancement of Humanistic Integration (Project CHIRON):** Contributions should seek to narrow the human-AI gap, not widen it. They should favor clarity, integrity, and semiotic richness, making the system more intuitive and engaging.

---

## II. Setting Up Your Development Environment

To ensure consistency, all development should be done in a controlled environment.

1.  **Fork & Clone:** Fork the repository on your Git platform and clone it to your local machine.
2.  **Create a Virtual Environment:** It is mandatory to use a virtual environment to manage dependencies.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3.  **Install Dependencies:** Install the exact package versions required for production and development.
    ```bash
    pip install -r requirements.txt
    pip install pytest pytest-flask # Install development dependencies
    ```

---

## III. The Contribution Workflow

The process is structured to ensure architectural coherence and philosophical alignment.

### Step 1: Proposing a Change

**Do not begin implementation immediately.** First, open an issue using one of the following templates:

*   **Architectural Enhancement Proposal (AEP):** For significant changes to the `Noesis Triad`, `Praxis Triad`, `StateManager`, or other core modules.
*   **Protocol Refinement Proposal (PRP):** For suggesting improvements to existing protocols like `CMEP` or `WGPMHI`.
*   **Bug Report / Anomaly Detection:** For reporting logical inconsistencies, ethical misalignments, or performance regressions. The report must analyze the 'Why' behind the anomaly.

Your proposal must include a section detailing its alignment with the **Core Philosophy**.

### Step 2: The Development Process

Once a proposal is approved by the Architect, you may begin development.

1.  **Create a Feature Branch:** Branch from the `main` branch. The name should be descriptive (e.g., `feature/cmep-audit-enhancement`).
2.  **Adhere to Coding Standards:**
    *   **Style & Formatting:** The project enforces **PEP 8** compliance using `black`. Run `black .` before committing.
    *   **Type Hinting:** All new functions and methods must include full, compliant type hints.
    *   **Architectural Adherence:** New code must respect the decoupled nature of the system. Do not bypass established data contracts (e.g., `ExecutionPlan`) or singletons (`StateManager`, `Logger`).
3.  **Commit with Intent:** Commit messages should be clear and concise, explaining the 'what' and 'why' of the change. A format like Conventional Commits is preferred (e.g., `feat:`, `fix:`, `refactor:`, `docs:`).
4.  **Ensure Self-Auditing:** If your change impacts system behavior, you **must** include updates to the `Wade-Gemini Protocol (WGPMHI)` tests to validate your contribution. New features require new `_check_` methods.
5.  **Run All Tests:** Before submitting, ensure all existing and new tests pass by running `pytest`.

### Step 3: Submitting a Pull Request

When you are ready to submit your contribution for review, open a Pull Request against the `main` branch. The PR description must contain the following checklist:

```markdown
### Pull Request Checklist

- **[ ] Linked Issue:** Closes #[issue_number]
- **[ ] Summary of Changes:** A clear description of the 'what' and 'why'.
- **[ ] Alignment Statement:** A brief explanation of how this PR aligns with the Core Philosophy (KISS, Constant Progression).
- **[ ] WGPMHI Audit Impact:**
  - [ ] New `_check_` tests have been added for new features.
  - [ ] Existing tests have been updated to reflect changes.
  - [ ] All tests pass locally via `pytest`.
- **[ ] Documentation Updated:** The `README.md` or other relevant documentation has been updated to reflect changes.
```

Contributions that do not follow this process or fail their ethical and logical audit will be rejected. We seek not just code, but a higher baseline of thoughtful and principled evolution.