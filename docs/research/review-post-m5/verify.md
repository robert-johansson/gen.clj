# The post-M5 review: the adversarial verification

Read `brief.md` first (where the code is, the rules, the report format);
its rules bind you too: read-only, the pure-JVM backend only, one JVM at a
time, write only under OUT.

## Your job

A reviewer has written one report (yours is named in your prompt). Before
anyone acts on its findings, you check each one, adversarially: **try to
refute it.** Reviewers are fresh to this codebase and can be wrong in
plausible ways:

- the code they cite does something else, or the line moved;
- the case they describe can't arise, because a caller or an earlier check
  rules it out;
- the behavior is a decision recorded in DESIGN §9, or a documented
  difference in `docs/research/genjl-test-suite.md` (D1–D16, U1–U5);
- the "expected" value rests on a wrong derivation;
- their script tests something other than the claim, or the bug is in the
  script;
- the finding is already fixed, or already an open bean;
- for idiom, the suggested direction would break a rule in CLAUDE.md, a
  performance note, or a law, or the "duplication" is two things that only
  look alike.

For each finding:

1. Read the cited code yourself, and what calls it.
2. For a correctness claim, **rerun** the reviewer's script, and where you
   can, write your own independent check (`OUT/scripts/V-<ID>.clj`): a
   different model, the law stated directly, the number derived by hand. A
   claim you can't reproduce is not confirmed.
3. Check DESIGN §9, the test-suite mapping and beans (`cd
   ~/code/gen.clj && beans list -S "<term>"`) for a decision or
   an open bean.
4. Give a verdict:
   - **CONFIRMED:** you reproduced it, or the code plainly says so.
   - **PLAUSIBLE:** likely real, but you couldn't fully establish it; say
     what's missing.
   - **REFUTED:** wrong, with the reason.
   - **DECIDED:** real behavior, but a recorded decision (cite it).

   Also say whether the severity is right, and whether the direction is
   sound or would break something.

You may also note anything the reviewer missed that you found while
checking, briefly, marked as such. Don't go hunting beyond the findings.

## Your report

Write `OUT/verify-<the report's name>` (for example `verify-idiom-core.md`):

```
# Verification: <the report's name>

Snapshot 34e9a44, 2026-10-22, fresh-context verifier.

## Verdicts
| ID | Claim (short) | Verdict | Severity (reviewer → yours) |
|---|---|---|---|

## Findings
### <ID>: <verdict>
How you checked (what you read, ran, derived), and the evidence. For
REFUTED or DECIDED, why. For CONFIRMED correctness claims, the
reproduction's script and output.

## Missed (optional)
```

When done, reply with the report's path, the counts per verdict, and any
severity you changed, in a few lines.
