# Working agreement

## Do not ask — act

Do not end a turn with "want me to…", "say the word", or "shall I proceed". If the next
step is obvious from the work, take it. Specifically, without asking:

- **commit and push** finished work on the current branch
- **run the rig** — recording, replay, builds, benchmarks
- **clean up** log data, stale containers, scratch files you created
- **restore rig state** that a reboot or power cycle reset (`trigger_mode`, generator
  polarity, fps) — these silently produce bad data, so fix them and say so afterwards
- **stop your own test containers and processes** that are holding a resource you need
- **install a missing tool** when it is the normal dependency for the task

Report what you did and what it showed. Offer alternatives *after* acting, not instead of
acting. If a choice is genuinely ambiguous, pick the option you would recommend, do it, and
name the assumption in one line.

The one thing still worth pausing on: destroying the **only** copy of data (deleting a
recording that has not been verified as synced elsewhere, `git reset --hard` over someone
else's uncommitted work, force-pushing). Verify the copy, then proceed.

## Reporting

State results plainly. If a measurement did not support the change, say so and say what
you are doing about it — do not dress up a non-result. Correct your own earlier claims when
new data contradicts them.

## No coauthor / AI attribution

Never add `Co-authored-by:`, `Co-Authored-By:`, or any Cursor / Claude / Copilot / AI
attribution trailer to commits, PRs, file headers, or docs. Commit messages are the
user's voice only (name/email from their git config).

Cursor may inject `--trailer Co-authored-by: Cursor <…>` into `git commit`. Do not
leave it: rewrite the tip with `git commit-tree` + `git update-ref` so the published
message has no coauthor line.
