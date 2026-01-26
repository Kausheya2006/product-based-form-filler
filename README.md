[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/4T_GxXnv)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-2e0aaae1b6195c2367325f4f02e2d04e9abb55f0b24a779b69b11b9e10269abc.svg)](https://classroom.github.com/online_ide?assignment_repo_id=22369915&assignment_repo_type=AssignmentRepo)
# DASS Spring 2026 Template

This template includes an Excel-based status tracker and an automated weekly snapshot workflow for submissions.

## Quick Start
1. Create/update `docs/StatusTracker.xls` in Microsoft Excel (binary file; do not replace with CSV).
2. Use these columns in row 1:
   - Week
   - Activity Name
   - Type
   - Responsible
   - Est Hours
   - Actual Hours
   - Status
3. Add weekly header rows (Week 1, Week 2, etc.) so students fill in below each header.
4. Save the file in `docs/` and commit it.

## Repository Layout
- `.github/workflows/weekly-snapshot.yml` auto-creates weekly release snapshots.
- `.github/workflows/snapshot-integrity.yml` detects tampering of past weekly snapshots.
- `docs/StatusTracker.xls` Excel tracker (update weekly).
- `docs/ProjectPlan.md` project plan template.
- `docs/release-labels.txt` optional: add weekly labels/categories for TAs.
- `docs/admin-setup.md` TA-only: required repo settings (tag protection).
- `src/` project source code.

## Notes
- `.gitattributes` marks `.xls/.xlsx` as binary to avoid noisy diffs.
- `.gitignore` ignores Office temp files like `~$StatusTracker.xls`.

## Weekly submission integrity (anti-cheat)

### What is enforced automatically
Every Friday, GitHub Actions will:

1. **Require weekly activity**: `docs/StatusTracker.xls` and `docs/ProjectPlan.md` must have at least one commit in the current week window.
2. **Create an immutable anchor**: an **annotated git tag** `submission-week-N` is created pointing to the repository state for that week.
3. **Create a release** from that tag. The release body is exactly the annotated tag message.
4. **Include a hash manifest**: the tag/release body includes sha256 hashes of every file under `src/` and `docs/`.

If any check fails, the weekly release/tag is **not created** (the workflow fails).

### How teams add "labels/categories" without editing the Release
Edit `docs/release-labels.txt`. Its contents are included in the tag annotation + release body.

### Allowing teams to add extra git tags
Teams may create additional git tags (e.g., `milestone-1`) on their own commits.
However, to prevent rewriting submission history, course admins should enable **Tag protection** for:

- `submission-week-*` (no deletions / force-updates)

### Tamper detection
Another workflow runs periodically to verify that for every `submission-week-*` tag:

- the GitHub Release body matches the annotated tag message (SHA256 check)

If a mismatch is found, it fails and opens a GitHub Issue as an audit trail.

> Note: GitHub cannot fully prevent cheating if students have full write access, but protected submission tags + hash manifests make manipulation difficult and highly detectable.

## Process Integrity Safeguards (TA Use)
Use these interventions to discourage fabrication and enforce process adherence.

### Tier 1: Soft Intervention (Mentorship)
- Observer Effect Warning: reference a specific tracker data point to signal review.
  Script: "I noticed in your tracker that Task A took exactly 4.0 hours and Task B took exactly 4.0 hours. Real development usually has more variation (e.g., 3.5 or 4.25). Please ensure you are logging actual clock times, not rough estimates."
- Git History Trap: if commits show batching, ask for proof tied to the stated day.
  Script: "Your tracker says you finished the API setup on Tuesday. Can you show me the git commit hash corresponding to that specific task on Tuesday?"

### Tier 2: Hard Intervention (Grading Penalty)
- Variance Check: if variance is suspiciously low (e.g., every entry is 2 hours), deduct 10-20% of the weekly process grade for Data Quality.
  Justification: "Data Quality. The logs provided lack statistical realism and appear smoothed. This is poor project management practice."
- Friday Night Deduction: if the tracker was only touched right before the deadline, deduct 50% of the process grade for Lack of Continuous Integration.
  Justification: "Agile requires iterative tracking. Batch-updating at the deadline defeats the purpose of the tracker."

### Tier 3: Formal Integrity Violation
- Forensic Audit: if the tracker claims work with no code changes in `src/`, conduct a Viva audit.
  Action: open GitHub Insights (Network graph) live, overlay tracker claims, and ask for the corresponding code.
  Outcome: if no code exists for claimed work, report Academic Dishonesty to the course professor.

## Policy Text for Course Handout
- Integrity of Project Artifacts: The `StatusTracker.xls` is a living document, not a homework assignment. It must reflect the actual state of development.
- Batch Updates: Updating logs retroactively for multiple days/weeks is considered a failure of process adherence.
- Fabrication: Logging hours for work not supported by version control evidence (git commits) constitutes academic dishonesty and will result in a grade of zero for that module.
- Verification: Teaching staff reserve the right to audit tracker data against commit timestamps and variance analysis.
