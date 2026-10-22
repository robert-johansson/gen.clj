# Post-M5 review (PPL soundness): P4, the LLM layer

Snapshot 34e9a44, 2026-10-22, fresh-context reviewer.

## Summary

The heart of the layer is sound. I derived every GFI weight of the token
generative function (generate, update, regenerate, project, assess,
propose), its retdiffs and discards, and the way a revision reuses kept
scores, and found them right. The same holds for the constrained
completion's local distribution and `conditional-weight`, the twist
telescoping and unbiasedness of token SMC (adaptive resampling included),
grammar masks and liveness, the byte-level marginals, and the LM cache's
first-read-wins ledger. Under a soft potential and a product, which the
suite never runs, I checked the density by enumeration, the conditional
weight's mean exactly, and every law. The defects are at the joins between
pieces:

- **The most serious (P4-01, high):** token SMC ignores the completion's own
  `:potential`. Given `(token/completion m {:potential g})`, it draws and
  weighs by the LM's rows. Its particles leave the completion's support,
  `smc/traces` gives them scores other than their `:log-p` (some -Inf), and
  Ẑ estimates the LM's Z rather than the documented one.
- **P4-02 (medium):** the end-token agreement DESIGN §9 requires between a
  grammar and a completion is checked only for a completion's own top-level
  potential. In SMC, or inside a product, a mismatch goes through, and SMC
  silently estimates another Z (0.37 where the matched pair gives 1).
- **P4-03 (medium):** `conditional-weight` returns 0.0, silently, for any
  trace that is not a completion's, such as a model's trace or a Map's.

The other three are a coverage gap (P4-04) and two edge contracts (P4-05,
P4-06).

## Findings

### P4-01: Token SMC ignores the completion's own `:potential`, so a constrained completion's SMC targets the LM, not the completion
- **Severity:** high (silent; confined to SMC over a constrained completion, which `smc/initialize` accepts. It could be argued critical: nothing warns.)
- **Where:** `src/gen/llm/smc.clj:15-17` (the target: "p the completion's density"), `:37-39`, `:85-93` (check-args accepts any `Completion`), `:213-217` (`:lm` proposes from the LM's row, `:masked` from the row times the SMC potential only), `:228` (`log-p` is `lm/score`, the LM's ledger), `:362-375` (`traces`: "their scores are the particles' log p"). Contrast `src/gen/llm/token.clj:253-268`, where the completion scores and draws by `gen.llm.constrained` when it has a potential. DESIGN.md:1285-1286.
- **What:** smc's docstring and DESIGN §9 define the target as "the token generative function's distribution times exp(complete)". For a completion with a `:potential g`, that distribution is the local q_g (gen.llm.constrained). But `weighed` takes `p` from the LM's ledger and the `:lm` proposal draws from the LM's row: neither looks at `(:potential completion)`. So:
  - the particles follow p_LM, not q_g, and violate g;
  - Ẑ estimates Σ p_LM(x)·exp(complete(x)), not Σ q_g(x)·exp(complete(x));
  - `smc/traces` rebuilds each particle with the constrained completion's `generate`, whose score is log q_g. That score differs from the particle's `:log-p`, and it is -Inf for every particle outside g's support.

  The grammar doc's recipe (`llm-grammar.md:260`: "token SMC's `:masked` proposal … targets π") invites exactly this pairing of a constrained completion with SMC.
- **Evidence:** `OUT/scripts/P4-01.clj` (output `P4-01.out`). Toy engine, 5 tokens, prompt [0 1], max-tokens 4. g is "every token is 2" (an end token may end it).
  - The constrained completion's 2,000 simulations all satisfy g.
  - `smc/run` over that completion with no SMC potential: 230 of 2,000 particles violate g, and log Ẑ is 0.0. All 2,000 traces have a score other than the particle's `:log-p`; 230 traces score -Inf.
  - With g also as the SMC potential and `:masked`: the documented Z is Σ q_g(x)·exp(complete(x)) = 1.0000 (by enumeration through assess), and Σ p_LM(x)·exp(complete(x)) = 0.8909. The mean of 40 runs' Ẑ (50 particles each) is 0.8915, identical to the plain completion's 0.8915. That is the LM's Z, not the completion's.
- **Direction:** Either refuse a completion with a `:potential` in `smc/initialize` (loud, pointing to SMC's own `:potential` with `:masked`, which is the documented way to target π), or take p and the `:lm` proposal from the completion's sites (`constrained/weighted` with its potential). The second would make the target q_g·exp(complete), and `traces` would then agree. The owner's call (see Decisions).
- **Known?** new.

### P4-02: The grammar–completion end-token agreement is checked only for a completion's top-level `:potential`; SMC and a product accept a mismatch, and SMC silently estimates another Z
- **Severity:** medium (an invariant checked on one path only; the result on the other is a silent, different estimate)
- **Where:** `src/gen/llm/constrained.clj:213-217` (`grammar?` only at the top level), called only from `src/gen/llm/token.clj:520`. `src/gen/llm/smc.clj:150-174` (initialize never checks its potential) and `:186-195` (twist-row checks only the row's size). `src/gen/llm/potential.clj:85-103` (`product` hides a grammar). DESIGN.md:1393-1400 ("a completion under a grammar must have the same end tokens (`:mismatched-end-tokens` otherwise)").
- **What:** A grammar leaves its end tokens out of its text, and DESIGN §9 makes a completion under a grammar with other end tokens a loud error: R2's stranding bug came from exactly that disagreement. The check runs only where `token/completion` receives a grammar directly.
  - Wrapped in `potential/product`, the same grammar is accepted.
  - `smc/run` never checks at all. The grammar then reads a text the completion sees differently: it refuses a token the completion treats as text (its own end token), or it treats as text a token the completion treats as the end.
  - Both runs are unbiased for SMC's literal target, complete(x) of the grammar's view. But that is not the language the pair means, and the design forbids the pair.
- **Evidence:** `OUT/scripts/P4-02.clj` (output `P4-02.out`). Byte tokenizer: a b c, BOS, EOS (`:eog #{EOS}`). Completions of exactly 3 tokens (`:end-tokens #{}`), grammar `.*`, so every completion's text is in the language and Z = 1.
  - `(token/completion m {:end-tokens #{} :potential (grammar/grammar m ".*")})` gives `:mismatched-end-tokens`, as designed. The same grammar inside `(potential/product …)` raises no error.
  - SMC with the matched grammar `(grammar/grammar m ".*" {:end-tokens #{}})`: mean Ẑ = 1.0.
  - SMC with the default-end-token grammar: mean Ẑ = 0.372 (`:masked`) and 0.377 (`:lm`), with no error. That is P(no EOS among 3 tokens) = 0.375, because EOS's "</s>" is never allowed as text.
- **Direction:** Check agreement wherever a potential meets a completion's end tokens: in `smc/initialize`, and through products (a product could carry its factors' end tokens, or `check-potential` could walk them). Alternatively, take DESIGN's listed alternative and pass the end tokens to the potential on each call (see Decisions).
- **Known?** new. R2's fix (DESIGN §9) covers the completion path only.

### P4-03: `conditional-weight` returns 0.0, silently, for any trace that is not a completion's
- **Severity:** medium (a silent default on misuse, against rule 6; the misuse is a natural one)
- **Where:** `src/gen/llm/constrained.clj:234-243`, the branch `(nil? potential) 0.0` at `:239`.
- **What:** `conditional-weight` reads `(:potential (:gen-fn tr))`. For a gen model's trace, a Map's or any other non-completion trace, that field does not exist. So the result is 0.0, the value meant for a plain completion, where q = p. Two documents call it "a trace's importance weight for the LM conditioned on the potential": DESIGN §6 "As built", and `llm-grammar.md:33`. A user weighing traces of a model that splices a constrained completion would get 0 for every one, so the importance weights would be the q-prior's without any error.
- **Evidence:** `OUT/scripts/P4-03.clj` (output `P4-03.out`). A gen model splices `(token/completion m {:potential no3})` at `:answer`. Over keys 0 to 3:
  - the subtrace's conditional weight is -9.9e-4, -0.562, -9.9e-4 and -0.0907;
  - the model's trace gives 0.0 every time, and so does a Map of the completion's trace.
- **Direction:** Be loud unless `(:gen-fn tr)` is a `Completion`. Or define it compositionally: the sum over the trace's completion subtraces, which is the right importance weight for a model whose only change is conditioning each completion on its potential. Document which.
- **Known?** new.

### P4-04: Completions under a soft potential or a product have no law or test (verified sound here)
- **Severity:** medium (a path with no law or test)
- **Where:** `test/gen/llm/constrained_test.clj:69-77, 111, 119, 196, 324` (every completion under a potential has a grammar). The claim of generality is in `src/gen/llm/token.clj:7-8, 505-507`, `src/gen/llm/constrained.clj:1-38` and `docs/research/llm-grammar.md:50, 431` ("any potential: a grammar, a product, a weighted one").
- **What:** A soft potential is the only route to `weighted`'s general log-sum-exp normalizer at a completion's site, and to `conditional-weight` with w(x) ∉ {0, -Inf}; `allowed-count` returns -1 there. The `weighted-is-the-renormalized-product` property tests that normalizer on random rows, but no GFI law or density test covers a completion under a soft potential or a product. smc_test's soft twist exercises SMC only. A regression there would pass the suite.
- **Evidence:** `OUT/scripts/P4-04.clj` (output `P4-04.out`). The toy engine, 5 tokens, max-tokens 4, completions enumerated. The soft potential is smc_test's `soft`; the product is `(product soft no3)`.
  - Σ q over every completion: 1.000000009 (soft) and 0.999999992 (product).
  - E_q[exp(conditional-weight)], computed exactly: equal to Z = Σ p·exp(complete) to every digit, 0.8410 and 0.8159.
  - Every law in `gen.laws/all` (22 laws, 100 trials each, seed 11), over the completion and a Map of it: no failure.
  So the code is right today; only the guard is missing.
- **Direction:** Add the soft and product models to `constrained_test`'s `deflaws`, and the enumeration and conditional-weight identities for a soft potential.
- **Known?** new.

### P4-05: Token SMC does not check prompt + max-tokens against the LM's `:limit`, so failure depends on the draws, and a successful run's `smc/traces` throws
- **Severity:** low (loud, but whether it is loud depends on the key)
- **Where:** `src/gen/llm/smc.clj:85-93` against `src/gen/llm/token.clj:348-353`.
- **What:** The completion refuses `[prompt max-tokens]` past the LM's limit before any read, "so that their validity never depends on the draws" (review-lm D-M2). smc's own `check-args` skips that check. A run whose particles all end early succeeds; one that does not fails deep in a step with the LM's `:context-full`. Converting a successful run (`smc/traces` → `generate`) then fails the completion's check.
- **Evidence:** `OUT/scripts/P4-05.clj` (output `P4-05.out`). With `:max-cells 8` (so the limit is 8), args [[0 1] 10]:
  - `simulate` gives `:context-full` before any read, for every key;
  - `smc/run` with 4 particles over keys 0 to 29 succeeds 23 times and fails 7 times (`:context-full`);
  - a successful run (key 0, particles [[4] [4] [4] [4]]) has `smc/traces` → `:context-full`.
- **Direction:** Have smc's `check-args` apply the completion's check, ideally by sharing it with gen.llm.token rather than copying it.
- **Known?** new. It is the completion's D-M2 (fixed) on SMC's path.

### P4-06: `token/dist` weighs an integral non-integer number (3.0) as outside its support, silently
- **Severity:** low (an inconsistent contract in an edge case: a silent -Inf where the house convention gives the density and the completion is loud)
- **Where:** `src/gen/llm/token.clj:148-157` (`token-of` returns nil for any non-integer number) and `:165-170` (-Inf). Contrast `src/gen/dist.clj:34, 492-506`, where categorical takes an integral value (`integral?`), and `token.clj:374-378`, where a completion's constraint 3.0 is loud (`:invalid-token`).
- **What:** At a `trace` site, `(token/dist row)` constrained to 3.0 weighs -Inf, while `(d/categorical probs)` constrained to 3.0 weighs log p[3]. On MLX, a 0-d float32 array's `array/item` is such a double, and `token-of` reads concrete 0-d arrays that way. So a token held in a float array scores -Inf there, with no error.
- **Evidence:** `OUT/scripts/P4-06.clj` (output `P4-06.out`). `generate` with `{:t 3}` gives -6.9207 for both distributions; with `{:t 3.0}`, token/dist gives -Inf and categorical -6.9207. A completion constrained to 3.0 gives `[:error :invalid-token]`.
- **Direction:** Take an integral value as its integer, as gen.dist does, or be loud for a non-integer as the completion is. Either way, one rule for the layer.
- **Known?** new.

## Decisions worth revisiting

- **Token SMC's p (DESIGN.md:1285-1290, "Token SMC and potentials").** Two sentences of the entry agree only for a plain completion: "its target is the token generative function's distribution times exp(complete)", and "p the LM's ledger's score". P4-01 is where they part. The owner should choose:
  - refuse constrained completions (the simplest; π stays reachable through SMC's own `:potential` with `:masked`);
  - or make p the completion's site distribution, so that SMC composes with a constrained completion and its traces agree.

  I'd recommend refusing, since the second adds a second way to say the same π.
- **"A grammar knows its end tokens" (DESIGN.md:1393-1407).** The decision put the agreement check in the completion's constructor. One alternative it listed was the completion passing its end tokens to the potential on every call, a protocol change. P4-02 shows the cost of the chosen form: every other place a potential meets end tokens needs its own check (SMC, products, any future wrapper), and today none has one. The listed alternative makes the agreement structural. If the owner keeps the constructor check, it should at least move to the two points where end tokens meet a potential: `constrained/check-potential`, made recursive through products, and `smc/initialize`.

## Checked and sound

- **Token GF weights** (read and derived, `token.clj:396-481`):
  - generate's weight is Σ of the constrained tokens' scores;
  - update's is new score − old − the fresh tokens'. Its discard (the constrained old positions and those past the new end) makes it reversible;
  - regenerate's is Σ over retained unselected tokens of (new − old). Selected and newly drawn choices cancel against the internal proposal, and dropped unselected ones against the reverse move's fresh draw (Eq. 4.1);
  - project, assess and propose are right.
  - The reuse of kept scores (`agreeing`/`scored`) keeps a score only while the prompt and every earlier token are the trace's. A redraw equal to the old token keeps it; anything after the first change is rescored.
  - Retdiffs name exactly the positions and lengths that changed.
  - Keys are `(fold-in key i)` by position, so constraining a token never moves another's randomness.
  - The no-change shortcuts compare normalized args.
  - `same-gen-fn?` compares the LM, potential and grammar by identity, so kept scores are never reused across different completions.
- **The constrained completion** (`constrained.clj`):
  - q_i = p·exp(w_i)/Z_i. Both normalizer branches are right: `log1p(-d)` under a permissive constraint, and log-sum-exp otherwise.
  - The end tokens take the end entry. The dead state stays consistent across draw, run scores and kept old scores.
  - The identity log p + complete − log q = prefix([]) + Σ log Z_i (+ complete − prefix at max-tokens) holds by telescoping, and `conditional-weight` computes it directly, in double.
  - Under a soft potential and a product: ran `P4-04.clj` (Σq = 1, E_q[exp cw] = Z exactly, every law).
- **Token SMC** (`smc.clj`):
  - The twist telescopes: prefix([]), the next-token entries, the end entry, and complete − prefix at the cap give complete(x).
  - Unbiasedness with ESS-adaptive resampling: the resampling decision is measurable on the current weights, and systematic, multinomial and the other resamplers keep each particle's expected offspring proportional to its weight. So the unnormalized particle measure is a martingale and E[Ẑ] = Z.
  - Dead particles, max-tokens 0, and the "all dead" estimate of 0 are right.
  - Keys are distinct by step, purpose and particle.
  - Float32 twists: `check-float-drift.clj` shows a particle's final log weight drifts from complete(x) by 1.7e-4 nats after 10 tokens and 1.0e-3 after 100, with weights near 1e4 and 1e5. That is negligible, but it is not exactly complete(x).
- **Grammars** (`grammar.clj`, `bytes.clj`):
  - `explore`, `token-live` and the exact masks make prefix 0 exactly on token-live states, so logw-next is consistent with prefix and complete.
  - End tokens are kept out of the trie and the liveness.
  - The sweep's depth stack is valid in preorder and dead subtrees are skipped.
  - In the marginals, each tokenization of a text with prefix b is counted once (by the token covering b's last byte), as are the next-byte terms and the end. A beam gives a lower bound.
  - Bare tokens are excluded, as documented.
- **Regex and DFA** (`regex.clj`, `dfa.clj`):
  - The derivative rules, `nullable?` and the normal forms (the repeat, star, alt and intersect identities) are right.
  - Hopcroft's split keeps the smaller block waiting, the sink absorbs every empty-language state, and the numbering is breadth-first canonical.
- **The LM cache's ledger** (`cache.clj`, `trie.clj`, `cells.clj`, `slots.clj`):
  - Every score a step or an extend records for the token after an entry is that entry's row entry (`slots.clj:359`, `cache.clj:249`), which is what `known-scores` hands out. So first read wins while entries are held.
  - Paths live only while they lead to an entry.
  - A read's sources are pinned, and a consumed parent is dropped after its child's entry is added.
  - The segment refcounts free cells only when no branch descends from them.
  - A hit's branch value stays readable after a later eviction in the same call.
- **The llama.cpp engine** (`llama.clj`, `llama/recurrent.clj`), by reading:
  - In a chunked run, row k scores token from+k+1, and the last row is the next-token distribution.
  - Rows are log-softmax with non-finite normalizers loud.
  - `restore` frees a slot it cannot roll back rather than leave a wrong state.
  - The recurrent replay changes only cost, and it splits deterministically.

## Coverage

Every line of each file in the layer, at the snapshot:

| File | Lines read |
|---|---|
| `src/gen/llm/branching.clj` | 1–254 |
| `src/gen/llm/bytes.clj` | 1–333 |
| `src/gen/llm/cache.clj` | 1–447 |
| `src/gen/llm/cells.clj` | 1–80 |
| `src/gen/llm/constrained.clj` | 1–243 |
| `src/gen/llm/dfa.clj` | 1–290 |
| `src/gen/llm/grammar.clj` | 1–250 |
| `src/gen/llm/llama.clj` | 1–642 |
| `src/gen/llm/llama/recurrent.clj` | 1–234 |
| `src/gen/llm/lm.clj` | 1–187 |
| `src/gen/llm/potential.clj` | 1–103 |
| `src/gen/llm/regex.clj` | 1–611 |
| `src/gen/llm/slots.clj` | 1–480 |
| `src/gen/llm/smc.clj` | 1–375 |
| `src/gen/llm/token.clj` | 1–520 |
| `src/gen/llm/trie.clj` | 1–176 |

Context read for the derivations:
- `src/gen/gfi.clj:100-600` (IRevisable, `same-value?`, `generate-again`, `run-op`, `defgf`);
- `src/gen/diff.clj:35-40`, `src/gen/dist.clj:492-512`, and the head of `src/gen/inference/weights.clj`;
- DESIGN §6 and the LLM entries of §9;
- `docs/research/llm-token-gf.md`, `llm-token-smc.md`, `llm-grammar.md`, and the earlier LLM reviews (`scratchpad/review-lm/lm-token-review.md`, `review-m3/m3-quality-review.md`);
- tests: `test/gen/llm/toy_engine.clj`, `toy_tokenizer.clj`, `byte_tokenizer.clj`, `smc_test.clj:1-200`, `constrained_test.clj:1-125`, the law list of `test/gen/laws.clj`.

Scripts and their outputs: `OUT/scripts/P4-01.clj` … `P4-06.clj` with `.out`, and `check-float-drift.clj` with its output. All were run on the JVM backend, one JVM at a time, on the toy engines.
