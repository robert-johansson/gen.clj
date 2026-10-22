# Verification: ppl-llm.md

Snapshot 34e9a44, 2026-10-22, fresh-context verifier.

I reran all six of the reviewer's scripts on the JVM backend and got the
reviewer's output exactly. I wrote an independent check for five findings
(`OUT/scripts/V-P4-01.clj`, `-02`, `-03`, `-04` and `-06`, each with its
`.out` file). For P4-05 I read the code and reran the script. Every run used
the toy engines, one JVM at a time. In beans, `beans list -S` finds no open
bean for any of these findings, under the terms smc, constrained,
conditional-weight, end-tokens, potential, mismatched, token/dist and
context-full.

## Verdicts
| ID | Claim (short) | Verdict | Severity (reviewer → yours) |
|---|---|---|---|
| P4-01 | Token SMC ignores the completion's own `:potential` | CONFIRMED (the claim about which target the docs promise needs care, below) | high → high |
| P4-02 | The end-token agreement is checked only for a completion's top-level potential; SMC and `product` skip it | CONFIRMED | medium → medium |
| P4-03 | `conditional-weight` returns 0.0, silently, for a trace that is not a completion's | CONFIRMED | medium → medium |
| P4-04 | No law or test covers completions under a soft potential or a product | CONFIRMED (a planted mutant passes the suite) | medium → medium |
| P4-05 | SMC does not check prompt + max-tokens against the `:limit` | CONFIRMED | low → low |
| P4-06 | `token/dist` weighs 3.0 as outside its support, silently | CONFIRMED | low → low |

## Findings

### P4-01: CONFIRMED

**The code.** I read `src/gen/llm/smc.clj` in full. Every use of
"potential" in it is SMC's own `:potential` option, and
`(:potential completion)` is never read:
- `:lm` proposes from the row of `lm/batch-next-logprobs` (`:213-216`);
- `:masked` proposes from `constrained/weighted` over that row and SMC's
  twist (`:215`);
- log p is `lm/score`, the LM's ledger (`:228`);
- `check-args` accepts any `Completion` (`:85-88`);
- `traces` rebuilds each particle with the completion's own `generate`
  (`:374`). For a constrained completion that goes through
  `constrained/run-scores` (`token.clj:253-259`), so a trace's score is
  log q_g.

**Which target the docs promise.** The lead asked me to derive this
myself. The docs make two promises, and for a plain completion they say the
same thing:
1. The target is "the completion's distribution weighed by the potential,
   π(x) ∝ p(x)·exp(complete(x)), p the completion's density" (smc's
   docstring at `:15-17`). DESIGN.md:1285-1286 says "the token generative
   function's distribution times exp(complete)". `llm-token-smc.md`
   Decision 2 says "the target is then exactly the token generative
   function's distribution … and the particles are its traces".
2. The particles are the completion's traces. "p the LM's score for it (its
   ledger's, as the completion's, so a particle's log p is its trace's
   score)" (`:37-39`). `traces`: "their scores are the particles' log p"
   (`:365`). DESIGN.md:1290: "p the LM's ledger's score, so particles
   agree with the completion's traces exactly".

`git blame` shows why the two can come apart. Both sentences date from
10bec81 (2026-10-21 00:32), and the completion's `:potential` arrived 44
minutes later (ace13df, 01:16, on m3-grammar). When the text was written,
"the completion's density" and "the LM's ledger" were the same thing. No
document has considered SMC over a constrained completion. So "the
documented target is q_g·exp(complete)" is the literal reading of promise 1,
not a decision anyone made. There are three readings, each with its own Z:

- **A (literal):** the completion's density, Z_A = Σ q_g(x)·exp(h(x)).
- **B (what the code does):** the LM's rows with the twist, the
  completion's potential ignored, Z_B = Σ p(x)·exp(h(x)).
- **C (genlm-control's shape):** there the target is the sampler's target
  times the critic, so a constraint built into the sampler multiplies into
  the target: Z_C = Σ p(x)·exp(g(x) + h(x)). (This reading is from what
  genlm-control's sampler/critic split does. The repository's docs say only
  that potentials and SMC follow genlm-control's.)

Under no reading is the current behavior consistent with the docs. Under B,
promise 2 fails: particles leave g's support, and their traces score -Inf
or differ from `:log-p`. Under A or C, Ẑ is wrong.

The reviewer's case 3 sets g = h (a 0/-Inf constraint, so exp(g + h) =
exp(h)). There Z_C = Z_B = 0.8909, so case 3 shows a wrong Ẑ only under
reading A. I therefore built a case where all three readings differ.

**Independent check (`V-P4-01.clj`).**
- **Setup:** the toy LM, prompt [3 0], max-tokens 3. g is "no token 1",
  the completion's constraint. h is a soft SMC twist: each 2 costs 0.9, and
  a sequence ending in 0 gains 0.4.
- **The three Z's by enumeration** (85 completions; Σ q_g = 1.0000000):
  - Z_A = 0.45200
  - Z_B = 0.48897
  - Z_C = 0.39155
- **SMC over the constrained completion** (60 runs of 50 particles), with
  the distance from each Z in standard errors (A / B / C):

  | proposal | mean Ẑ | A | B | C |
  |---|---|---|---|---|
  | `:lm` | 0.48562 | 5.7 | -0.57 | 16.0 |
  | `:masked` | 0.48906 | 25.0 | 0.06 | 65.8 |

  Ẑ is Z_B. A and C are rejected decisively under `:masked`.
- **The same keys over the plain completion** give bit-identical estimates,
  under both proposals. The completion's potential has no effect at all.
- **First tokens of 4,000 particles, no SMC potential:** token 1 appears
  5.8% of the time. p_LM gives 6.1%; q_g gives 0. The other tokens also
  follow p_LM, not q_g.
- **`smc/traces` after a 500-particle run:** 83 particles contain a 1, and
  those 83 traces score -Inf. 0 of 500 scores equal `:log-p`. Over the
  plain completion, 500 of 500 do.

**The reviewer's script (`P4-01.clj`)** reran identically:
- 230 of 2,000 particles violate g, log Ẑ = 0.0, 2,000 mismatched scores,
  230 at -Inf;
- Z_LM = 0.89088, Z_doc = 0.99999999, and the mean Ẑ = 0.89146, the same
  to the last digit as the plain completion's.

**Severity:** high is right. The failure is silent, and the use is legal
but unusual: `initialize` accepts the completion, and every SMC example in
the docs uses a plain one. A small correction to the evidence: the reviewer
says `llm-grammar.md:260` "invites exactly this pairing". It does not. That
line describes SMC's `:masked` proposal with SMC's own `:potential`, the
documented pairing with a plain completion. What does invite the pairing is
that the grammar doc's `answer` (line 31) is a constrained completion,
while the SMC doc's `answer` (line 43) is a plain one.

**Direction:** sound.
- Refusing a completion with a `:potential` in `initialize` is the
  simplest fix. It loses nothing, since π stays reachable through SMC's own
  `:potential` with `:masked`.
- "Take p from the completion's sites" implements reading A, and it is the
  only fix that keeps promise 2 as written.
- A third option for the owner is reading C (genlm-control's): treat the
  completion's potential as a factor of the target, the same as SMC over
  the plain completion with `(product g h)`. That is probably what a user
  who passes a constrained completion wants, π conditioned on g. But
  `traces` would still score q_g ≠ p, so promise 2 would have to be
  restated.

### P4-02: CONFIRMED

**The code.** I read the following:
- `constrained/check-potential` (`constrained.clj:197-218`): the agreement
  check is `(grammar/grammar? potential)` at the top level, called only
  from `token/completion` (`token.clj:520`);
- `potential/product` is a bare `reify` (`potential.clj:85-103`), so it
  hides the grammar;
- `smc/initialize` (`:150-174`) and `twist-row` (`:186-195`) check no end
  tokens;
- the grammar's own contract (`grammar.clj:29-30`, `:131-133`): "its end
  tokens must be the completion's (gen.llm.constrained checks them)" and
  "a completion under the grammar must have these end tokens";
- DESIGN.md:1393-1407, the R2 decision.

The pair is forbidden by the grammar's contract, so this is an unchecked
invariant, not a decided behavior. The reviewer is right that SMC is
unbiased for its literal target. In a mismatched pair, the grammar reads as
-Inf any token it treats as an end token.

**The reviewer's script (`P4-02.clj`)** reran identically:
`:mismatched-end-tokens` at construction, `:no-error` through `product`,
and the following mean Ẑ values:

| grammar | proposal | mean Ẑ |
|---|---|---|
| matched | `:masked` | 1.0 |
| mismatched | `:masked` | 0.37196 |
| mismatched | `:lm` | 0.377 |

The exact P(no EOS in 3 tokens) = 0.37513.

**Independent check (`V-P4-02.clj`).** This is R2's own case, the opposite
direction to the reviewer's. A newline ends the completion
(`:end-tokens #{2}`), but the grammar treats it as text
(`(grammar m "[ab]{3}\n" {:end-tokens #{}})`).
- The constructor refuses it (`:mismatched-end-tokens`).
- Wrapped in `product`, construction succeeds and `simulate` fails at the
  draw with `:zero-weight`: R2's stranding, loud but late.
- The pair means Z = P(three of a/b, then the newline) = 0.0051304 by
  enumeration. The matched pair (`"[ab]{3}"` with `{:end-tokens #{2}}`)
  gives mean Ẑ = 0.0051594 with `:masked`.
- R2's pair in SMC gives Ẑ = 0.0 under both proposals, with no error. So
  SMC reproduces the exact bug the R2 decision made loud, silently. Every
  particle dies, which gives log Ẑ = -Inf, a visible symptom. In the
  reviewer's direction the wrong Ẑ (0.37 against 1) has no symptom at all.

(The idiom review's I4-02 reached the same R2 case independently. Its
cross-reference "the PPL review reports it as P4-03" should read P4-02.)

**Severity:** medium is right. The misuse is forbidden by the contract
rather than legal, so medium's "an invariant nothing checks" fits better
than high.

**Direction:** sound. One caveat: a product is a bare `reify`, so
"`check-potential` could walk them" needs `product` to expose its factors,
or to carry the end tokens, as the reviewer also suggests. DESIGN's listed
alternative, passing the end tokens on every call, makes the agreement
structural, at the cost of a protocol change.

### P4-03: CONFIRMED

**The code.** I read `constrained.clj:224-243`. `(:gen-fn tr)` of a
non-completion trace has no `:potential`, so the `(nil? potential)` branch
returns 0.0. The docstring limits the argument to "a trace of a completion
under a potential". DESIGN.md:395-396 ("a trace's importance weight for
the LM conditioned on the potential") and `llm-grammar.md:33` describe it
more loosely. Rule 6 forbids silent defaults. No caller in `src/` passes it
anything else; the only callers are in tests.

**Runs.**
- The reviewer's script reran identically: the subtrace gives -9.9e-4,
  -0.562, -9.9e-4 and -0.0907, while the model's trace and the Map's trace
  give 0.0 each time.
- `V-P4-03.clj` widens it. A normal model's trace gives 0.0, and so does a
  distribution's trace. A `call-at` around a constrained completion gives
  0.0 for every key, where the completion's own traces give -0.247, -3.55
  and -0.593.

**Severity:** medium is right. The silent default would turn importance
weights into the q-prior's without any error.

**Direction:** sound. Being loud unless `(:gen-fn tr)` is a `Completion` is
the minimal fix. The compositional form, the sum over completion subtraces,
is right only for a model whose sole change is conditioning each
completion on its potential, and its docstring would have to say so, as
the reviewer notes.

### P4-04: CONFIRMED

**Coverage.** `grep` over `test/` finds 11 valid completions constructed
with a `:potential` (`constrained_test.clj:74-77, 111, 119, 196, 324, 332`,
and `llama_grammar_test.clj:82, 97`, whose `phone`, `answer` and `json`
come from `setup`). Every one is a grammar. The misuse tests
build completions with bad potentials, but none is a valid soft potential.
smc_test's `soft` is used only as SMC's twist. The
`weighted-is-the-renormalized-product` property covers soft weights in
`weighted` itself, so a bug in the normalizer would be caught. What is
uncovered is everything soft past `weighted`, above all
`conditional-weight` with w(x) ∉ {0, -Inf}.

**The reviewer's script (`P4-04.clj`)** reran identically:

| potential | Σ q | Z = E_q[exp cw] | laws |
|---|---|---|---|
| soft | 1.000000009 | 0.840965 | 22 laws × 100 trials, no failure |
| product | 0.999999992 | 0.815872 | 22 laws × 100 trials, no failure |

Note that E_q[exp cw] = Z holds as an identity once cw's p and w match the
enumeration's. So it checks those two inputs, not q itself; the laws and
Σ q = 1 check q.

**Independent check (`V-P4-04.clj`).** I planted mutants at run time with
`alter-var-root`, so no file changed. The planted mutant reads any finite
w(x) as 0 and keeps -Inf. That is right for every grammar and wrong for any
soft potential. I ran the only tests that call `conditional-weight`
(`local-is-not-global` and `an-unconstrained-completion-has-weight-0`;
`grep` finds no other caller):

| conditional-weight | pass | fail | error |
|---|---|---|---|
| original | 4 | 0 | 0 |
| clamped mutant | 4 | 0 | 0 |
| dropped mutant (control: no w(x) at all, caught per `llm-grammar.md:601`) | 2 | 2 | 0 |

The control shows the method can fail. Under a soft potential (each 0
costs 0.6), the clamped mutant's E_q[exp cw] is 1.0000, where Z = 0.84669.
So a regression on this path passes the suite, as the reviewer says.

**Severity:** medium, by the scale's own wording ("a path with no law or
test").

**Direction:** sound. The direction should name a soft
`conditional-weight` case explicitly, since that is where the gap is.

### P4-05: CONFIRMED

**The code.** I read `smc.clj:85-93` against `token.clj:329-354` and
`cache.clj:359-375`. `:context-full` is a per-context check, so a run fails
exactly when some particle's context plus its token exceeds `:limit`.
That depends on the draws.

**The reviewer's script** reran identically:
- `:limit` 8;
- `simulate` gives `:context-full` for every key;
- `smc/run` over keys 0 to 29: 23 succeed and 7 fail with
  `:context-full`;
- key 0's particles are [[4] [4] [4] [4]], and `smc/traces` of that run
  gives `:context-full`.

This is review-lm's D-M2, fixed for the completion, reappearing on SMC's
path. **Severity:** low is right: it is always loud, only at a
key-dependent time. **Direction:** sound. Sharing the completion's
`check-args` also covers the other half of I4-01, the prompt's tokens.

### P4-06: CONFIRMED

**The code.** I read `token.clj:148-170`: `token-of` returns nil for any
non-integer number, so the log density is -Inf. Every discrete
distribution in `gen.dist` accepts integral floats through `integral?`
(`dist.clj:34`, `496`, `525`, `539`, `628`, `637`, `653`). That makes the
inconsistency layer-wide, not a matter of categorical alone. No DESIGN §9
entry decides it. The docstring says only "-Inf for an integer outside the
vocabulary".

**The reviewer's script** reran identically: -6.9207 for both
distributions at 3, -Inf against -6.9207 at 3.0, and `[:error
:invalid-token]` for a completion constrained to 3.0. `V-P4-06.clj` adds
to it:

| value | token/dist | categorical |
|---|---|---|
| 1 (Long) | -3.864 | -3.864 |
| 1 (Integer) | -3.864 | -3.864 |
| 1.0 (Double) | -Inf | -3.864 |
| 1.0 (Float) | -Inf | -3.864 |
| 1.5 | -Inf | -Inf |
| 7 | -Inf | -Inf |
| -1 | -Inf | -Inf |

The MLX remark (an `array/item` of a 0-d float32 array is a double) is
right by reading the code. I did not run it, because MLX is not allowed
here.

**Severity:** low is right. **Direction:** sound. Either rule works if the
layer adopts one; following `gen.dist` (integral means in the support)
matches rule 5's spirit.

## Missed (optional)

- **The reviewer's P4-01 case 3 cannot tell readings A and C apart.** With
  g = h, the code's Ẑ agrees with genlm-control's reading. The finding
  needs a case with g ≠ h, which `V-P4-01.clj` supplies. This changes how
  the evidence should be presented, not the verdict.
- **The idiom review's numbering:** I4-02 says P4-03 where it means P4-02.

Nothing else turned up while checking.
