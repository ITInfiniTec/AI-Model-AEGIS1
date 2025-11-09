# 🛡️ AEGIS Core Security Policy

The integrity and security of the AEGIS Core and the broader Project Nexus ecosystem are of paramount importance. We are committed to ensuring our systems are secure and that any potential vulnerabilities are addressed responsibly.

---

## I. Supported Versions

Security updates and patches are only applied to the most recent major version of the AEGIS Core. Please ensure you are running the latest stable version to benefit from all security enhancements.

| Version | Supported          |
| :------ | :----------------- |
| 4.x.x   | :white_check_mark: |
| < 4.0.0 | :x:                |

---

## II. Reporting a Vulnerability

We take all security reports seriously. If you believe you have discovered a security vulnerability, we ask that you help us by reporting it responsibly. **Please do not disclose any vulnerability publicly until a resolution has been implemented.**

### How to Report

1.  **Private Disclosure:** To report a vulnerability, please send a detailed email to `security@project-aegis-core.sim`. (This is a simulated address for the purpose of this project).
2.  **Provide Detailed Information:** Your report should be as detailed as possible. Please include:
    *   A clear description of the vulnerability and its potential impact.
    *   Step-by-step instructions to reproduce the issue.
    *   The version of the AEGIS Core you are using.
    *   Any relevant code snippets, logs, or screenshots.

### What to Expect

After you submit a report, you can expect the following process:

1.  **Acknowledgment:** We will acknowledge receipt of your report within 48 hours.
2.  **Initial Assessment:** Our team will work to validate the vulnerability. We may contact you for additional information during this phase.
3.  **Resolution:** Once confirmed, we will work on a patch. We aim to release a fix for critical vulnerabilities within a timeframe appropriate to the severity.
4.  **Public Disclosure:** After a patch has been released, we may publicly disclose the vulnerability (often with credit to the reporter, with your permission) to ensure all users are aware of the need to update.

---

## III. Security Philosophy

The AEGIS Core's security posture is guided by the **Architect's Handbook**. Our approach is proactive, not reactive.

*   **Secure by Design:** Security is not an afterthought. Principles like input validation (`DataIntegrityProtocol`), explicit data contracts (`ExecutionPlan`), and self-auditing (`WGPMHI`) are built into the core architecture.
*   **Differential Tainting (M8):** We assume all external inputs are potentially hostile. Data from external sources is "tainted" and tracked to ensure it does not corrupt critical sinks.
*   **Least Privilege:** Components are designed with the minimum necessary permissions and access to data, as seen with the `StateManager` acting as a gatekeeper to persistent memory.