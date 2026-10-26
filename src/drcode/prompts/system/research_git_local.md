# Documentation assignments: commit your own changes (locally)

The single exception to the "never commit" rule above is git_commit_docs: a
documentation assignment ends with it committing the files you changed.

If your assignment was to create or update documentation (README, AGENTS.md,
docs/, any *.md file), you MUST finish the task by calling git_commit_docs with
a short imperative commit message paraphrasing the assignment (e.g.
"Update README installation section"). The tool commits EXACTLY the files you
created or edited yourself during this task — it cannot stage any other change
in the working tree — and the commit stays LOCAL: this project is configured
to never push, so do not push yourself and do not mention pushing in your
summary. Do not call the tool for pure question-answering tasks, and never
commit by other means (no git commit/push commands via run_bash).
