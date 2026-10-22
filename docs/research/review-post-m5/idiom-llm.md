# Post-M5 review (idiom): the LLM layer (I4)

Snapshot 34e9a44, 2026-10-22, fresh-context reviewer.

## Summary

The LLM layer is in good shape: 16 namespaces of one idea each, data-oriented
where it can be (the persistent trie, the cell ledger and the recurrent
replay are pure maps and functions; the books of `gen.llm.slots` are pure
transitions with one effectful `forget!`), protocols only where engines or
potentials vary, primitive hot loops, and no reflection or boxing on a hot
path. I found nothing major. What matters most is checks written several
times that have since drifted apart. Token SMC's copy of the completion's
argument check is weaker than the original, so a run can succeed and its
`traces` then throw, and a completion's own `:potential` is silently ignored
(I4-01). The grammar/completion end-token agreement decided in §9 is enforced
as a type check on `Grammar`, so a `product` or token SMC skips it (I4-02).
The rule "a token of the vocabulary" is written in eight places, and two of
them bring back the big-integer bug that the Branching quality review fixed
(M6) (I4-03). After those come a host thread pool written twice (I4-04) and
a private run-state contract between `gen.llm.token` and
`gen.llm.constrained` (I4-05).

Counts: 0 major, 6 moderate, 5 minor.

## Findings

### I4-01: gen.llm.smc re-implements the completion's argument check, and the copy is weaker
- **Severity:** moderate (correctness; the PPL review reports the constrained-completion part as P4-01)
- **Where:** `src/gen/llm/smc.clj:85-93` (`check-args`) against `src/gen/llm/token.clj:329-354` (`check-args`); `smc.clj:150-174` (`initialize`), `smc.clj:258-286` (`step`), `smc.clj:362-375` (`traces`)
- **What:** `smc/initialize` validates `[prompt max-tokens]` with its own copy of the completion's rule. The copy checks only the shape (two arguments, a non-empty sequential prompt, a non-negative max-tokens). It does not check that the prompt's tokens are in the vocabulary, or that prompt plus max-tokens fit the LM's `:limit`. `token/check-args` checks both "so that their validity never depends on the draws". The copy also accepts any `Completion`, including one with its own `:potential`, and `step` never consults that potential: rows come from `lm/batch-next-logprobs`, log p from `lm/score`. As a result, a run's validity depends on its draws: it fails at whichever step reaches the limit, or succeeds if every particle ends first. `smc/traces` of a run that succeeded then throws `:context-full`, because it calls `gfi/generate`, which runs the real check. And over a constrained completion the particles break its grammar, and `smc/traces` scores them -Inf, where its docstring promises "their scores are the particles' log p".
- **Evidence:** `OUT/scripts/I4-01.clj`, on the JVM backend:
  ```
  gfi/simulate   [[95 500] 3]   :invalid-token
  smc/initialize [[95 500] 3]   :no-error
  gfi/simulate   [[95] 100]     :context-full        (:limit 64)
  smc/initialize [[95] 100]     :no-error
  smc/run [[95] 100], ending early: [[57] [6] [69] [28]] log-ml 0.0
  smc/traces of that run        :context-full
  particles   [[57 27 86] [6 36 21] [69 15 89] [28 63 17]]   (the grammar [a-c]+ allows only tokens 65 to 67)
  particles' log p    [-9.40 -12.10 -12.12 -8.75]
  smc/traces' scores  [##-Inf ##-Inf ##-Inf ##-Inf]
  ```
- **Direction:** Keep one check, the completion's. Make token's argument check public (for example `token/check-args`) and call it from `initialize`. Refuse a completion that has a `:potential`, or define the target for one (that is the PPL review's call). Check the twist once at `initialize` with `constrained/check-potential` (see I4-02), rather than its size per particle per step in `twist-row`.
- **Known?** new. P4-01 (the PPL review of this layer) reports the constrained-completion consequence.

### I4-02: The end-token agreement is a type check on `Grammar` in gen.llm.constrained, so a product or token SMC skips it
- **Severity:** moderate (correctness; the PPL review reports it as P4-03)
- **Where:** `src/gen/llm/constrained.clj:213-217` (`(grammar/grammar? potential)` in `check-potential`), `constrained.clj:39` (the require of `gen.llm.grammar`, only for this check); `src/gen/llm/potential.clj:85-103` (`product`); `src/gen/llm/smc.clj:150-174` (`initialize` runs no potential check)
- **What:** DESIGN §9 ("A grammar knows its end tokens") decides that a completion under a grammar must have the grammar's end tokens, and `:mismatched-end-tokens` otherwise. The check is `(grammar/grammar? potential)`, so the generic potential middleware special-cases one implementation and requires `gen.llm.grammar` to do it. A grammar wrapped in `potential/product` is not a `Grammar`, so the completion accepts the mismatch and half its runs then fail mid-run with `:zero-weight`. Token SMC takes grammars as twists (§9: grammars stay potentials shared with token SMC) and runs no check at all. There, the DESIGN's own failing example, a language that needs the end token's bytes, strands every particle and returns log Ẑ = -Inf. A matched grammar for the same language is `:inexpressible` when it is made.
- **Evidence:** `OUT/scripts/I4-02.clj`:
  ```
  bare grammar, completion :end-tokens #{}:    :mismatched-end-tokens
  (product g), completion :end-tokens #{}:      :no-error
  simulate [prompt 8] over keys 0..19, wrapped (mismatched): {:zero-weight 10, :no-error 10}
  grammar [ab]{3}\. with :end-tokens #{}; under completion end tokens #{14}: :mismatched-end-tokens
    as smc's :potential, completion end tokens #{14}: :no-error
    smc/run :masked, log-ml-estimate ##-Inf
  the same language, matched end tokens #{14}: :inexpressible
  ```
- **Direction:** Let a potential declare its end tokens. That means an `-end-tokens` method on `Potential` (nil for no opinion) that `Grammar` answers, `product` forwards (failing loudly when two members disagree), and `check-potential` compares without knowing about grammars. Call `check-potential` from `smc/initialize` too. This keeps §9's decision that a grammar knows its end tokens. It is not the alternative §9 rejected, which was passing the end tokens to the potential on every call.
- **Known?** new. P4-03 reports the same consequences from the PPL side.

### I4-03: "A token of the vocabulary" is checked in eight places, and two copies regress the Branching review's M6
- **Severity:** moderate
- **Where:** `src/gen/llm/branching.clj:74-84, 131-142` (`index?`, `vocabulary-tokens`); `src/gen/llm/lm.clj:77` (the tokenizer's `:eog`), `lm.clj:115-120` (`check-token`); `src/gen/llm/cache.clj:360-374` (`check-context`); `src/gen/llm/token.clj:340, 366, 375, 514-518`; `src/gen/llm/bytes.clj:236-240`; `src/gen/llm/grammar.clj:115-125`
- **What:** `br/index?` is the rule. It is quick for a long, and it treats a big integer as out of range rather than as an error, which was M6's fix in the quality review (`docs/research/llm-branching.md`). `lm/check-token` and `cache/check-context` restate it, each with its own message. The end-token option (a set of tokens, defaulting to the LM's `:eog`, stored as longs) is read and checked three times: in `token/completion`, `bytes/byte-lm` and `grammar/end-tokens-of`. The last two coerce with `(long %)`: `(< -1 (long %) (long n-vocab))`. So a token beyond long's range throws `IllegalArgumentException`, not an ex-info (rule 6), which is the class of bug M6 fixed.
- **Evidence:** `OUT/scripts/I4-03.clj`: `token/completion :end-tokens #{big}` gives `:invalid-end-tokens`; `bytes/byte-lm` and `grammar/grammar` give `java.lang.IllegalArgumentException`.
- **Direction:** Use one predicate, `br/index?`, and one reader for the end-token option, taking its default (for example `lm/end-tokens m opts default`), in the completion, `byte-lm` and `grammar`. Have the cache reuse branching's vocabulary check, so the messages come from one place.
- **Known?** new. M6 of the Branching quality review fixed the token checks that existed then.

### I4-04: The host thread pool is written twice, in llama and smc
- **Severity:** moderate
- **Where:** `src/gen/llm/llama.clj:134-151` (`row-pool`, `in-parallel`); `src/gen/llm/smc.clj:95-124` (`host-pool`, `host-threads`, `default-threads`, `shared-pool`), `smc.clj:244-256` (`in-order`)
- **What:** The two pools are the same. Each is a fixed pool of daemon platform threads sized `(max 1 (min 4 (quot processors 2)))`; smc's docstring says "as the llama.cpp engine's row pool". Each has the same ordered map over it: `invokeAll`, then `.get` on every future with the `ExecutionException` unwrapped, and plain `mapv` for a single element. smc's version adds `bound-fn*` and a `false` pool. Both hint the inner fn `^Callable`, which does nothing because `invokeAll` is hinted `^Collection`. smc does not even import `Callable`.
- **Evidence:** read, side by side: `llama.clj:146-151` and `smc.clj:250-256` differ only in `bound-fn*` and the pool test.
- **Direction:** Write one small helper for the layer's host work, either a namespace of its own or part of `gen.llm.slots`, which already owns the engine threads. It would hold `pool` (a name and a size), the default size (read once, honouring `gen.llm.host-threads`) and `in-order`. §9's "as the llama.cpp engine's row pool" then holds by construction.
- **Known?** new.

### I4-05: gen.llm.constrained works on gen.llm.token's private run state
- **Severity:** moderate
- **Where:** `src/gen/llm/constrained.clj:142-195` (`end-tokens`, `dead?`, `draw-next`, `run-scores`), `constrained.clj:224-243` (`conditional-weight`); `src/gen/llm/token.clj:210-213` (the comment describing the state), `token.clj:232-241` (`append` sets `:dead?`), `token.clj:253-285` (`run-scores`, `draw-next`, `scored`)
- **What:** `constrained/draw-next` and `run-scores` destructure the completion record (`:lm :potential :end-tokens`) and the run state (`:ctx :tokens :dead?`). Both are defined and kept private by `gen.llm.token`. `:dead?` is set by token's `append` and read by constrained's `dead?`, whose docstring says "gen.llm.token's run state keeps the flag". Yet the comment that describes the run state (`token.clj:210-213`) does not list it. The interface between the two namespaces also reads badly. Token's `draw-next` returns `[t score]`, where `score` is nil or a one-element vector (constrained's docstring calls it `[t [score]]`). `scored`'s last parameter, `fresh`, is that vector of scores, while the state's `:fresh` holds the positions drawn.
- **Evidence:** read: `(defn- dead? "... (gen.llm.token's run state keeps the flag.)" [state] (boolean (:dead? state)))` against the state comment ";; A run's state: {:ctx … :tokens, :scores …, :fresh (the positions drawn), :old …}".
- **Direction:** Keep the math in constrained, with explicit arguments: `weighted`; a site function of `(lm potential end-tokens ctx tokens dead?)` that returns the site's row; `conditional-weight`; and `check-potential`. Token's transitions then own the run state, and its comment lists `:dead?`. `draw-next` returns a score as a double or nil, and `scored` takes it under a name like `drawn-score`.
- **Known?** new.

### I4-06: Two docstrings promise what the code does not do
- **Severity:** moderate
- **Where:** `src/gen/llm/smc.clj:45-46`; `src/gen/llm/lm.clj:31-32`
- **What:**
  1. smc's namespace docstring says "particle i's token at step t from (fold-in (fold-in key t) i)". The code (`step-key`, `step`) folds in a purpose first, giving `(fold-in (fold-in (fold-in key t) 0) i)`, as the comment at `smc.clj:78-80` says. Anyone reproducing a particle's draw from the docstring gets other tokens.
  2. lm's docstring says "concurrent callers are served one at a time, in the order they arrive". `CachedLM` serializes calls with `locking`, a monitor, which promises no order. The cache's choices, and on llama.cpp its results bit for bit, depend on the order of the calls. So the docstring should say that the order of concurrent callers, and with it their results, depends on timing.
- **Evidence:** `OUT/scripts/I4-06.clj`: smc's first step drew `[57 22 74 86 77 74 72 55]`; the docstring's keys give `[55 90 28 69 72 6 90 86]`; `(fold-in (fold-in (fold-in k 0) 0) i)` gives the drawn ones. The second is from reading (`lm.clj:51-59`).
- **Direction:** Correct the key in smc's docstring. In lm's, replace the clause with what holds (one call at a time, in no promised order), or use a fair `ReentrantLock` if arrival order is wanted.
- **Known?** new.

### I4-07: Failure paths that lose an error or overflow the stack
- **Severity:** minor
- **Where:** `src/gen/llm/trie.clj:143-150`; `src/gen/llm/cache.clj:175-177`; `src/gen/llm/llama.clj:511-513`
- **What:**
  - `remove*` on a path the trie does not hold recurses with the child nil and the depth unchanged until `StackOverflowError`. The docstring states the precondition, but rule 6 wants an ex-info.
  - `reading` swallows a failed `dispose!` with `(catch Throwable _)`.
  - `open-slots` runs every undo inside its catch, and an undo that throws replaces the original exception.
- **Evidence:** `OUT/scripts/I4-07.clj`: `remove-entry of an absent path: java.lang.StackOverflowError`.
- **Direction:** `remove*` throws an ex-info (`:no-entry`) when the child is missing. Secondary failures go onto the primary exception with `.addSuppressed`.
- **Known?** new.

### I4-08: Small idioms
- **Severity:** minor
- **Where and what:**
  - `(if here here entry)` should be `(or here entry)`: `trie.clj:57`.
  - `grammar/allowed` restates `allowed?`'s bit test inline; it could be `(filterv #(allowed? g s %) …)`: `grammar.clj:212` against `grammar.clj:205`.
  - `(filterv #(not (:done? …)))` reads better with `remove`: `smc.clj:270`.
  - `check-options` checks `:model` twice: `option-checks`' `string?` (`llama.clj:335`) can never fail after the check at `llama.clj:347`.
  - `split-recurrent?` is called once only for the error it may throw (`_ (split-recurrent? nil)`, `llama.clj:489`), and reads the property up to three times (`llama.clj:192, 197, 198`). Read it once into nil, true or false.
  - `recurrent/plan` decodes the whole step a second time to cost it (`(cost st seqs positions whole fixed)`, `recurrent.clj:231-232`), though `full` (`recurrent.clj:206`) already holds the cost: `fixed` plus `(extra full)`.
- **Direction:** as listed.
- **Known?** new.

### I4-09: Names that shadow core functions, or mean two things
- **Severity:** minor
- **Where and what:**
  - The local and field `first` in `bytes.clj:36, 64, 133, 140, 163, 202` forces `(clojure.core/first ts)` at `bytes.clj:71`.
  - `next` is used as a local at `bytes.clj:279, 325`, `grammar.clj:99, 107` and `potential.clj:79`.
  - `byte` is used as a local at `dfa.clj:233, 243, 255`.
  - A private `state` returns the volatile in `slots.clj:167` and its value in `cache.clj:72`.
  - smc's private `weighed` (`smc.clj:219`) sits beside constrained's public `weighted` and does something else.
  - cache's docstrings say forks are "conj!ed onto the volatile made" (`cache.clj:167, 221`), but the code is `vswap! … conj`; no transient is involved.
- **Direction:** Rename the locals (for example `offsets` for bytes' `first`, `succs` for `next`, `b` for `byte`), give the two `state` functions different names, and name smc's step function after what it returns (for example `moved`).
- **Known?** new.

### I4-10: Leftover hints and an uneven warning flag
- **Severity:** minor
- **Where and what:**
  - `^Callable` on fn forms has no effect: `llama.clj:150`, and `smc.clj:255`, which does not import the class.
  - `Callable` is imported but unused: `slots.clj:39`.
  - `^clojure.lang.Volatile` on `vswap!`'s target is redundant, since the macro tags it itself: `llama.clj:207, 213, 221, 242`, and `llama.clj:626` on a deref.
  - `*unchecked-math* :warn-on-boxed` is set in ten of the sixteen namespaces, but not in branching, slots, llama, llama.recurrent, lm or potential. With it on, `recurrent.clj:133, 183, 226, 229` and `lm.clj:117` box. That is bookkeeping, not a hot path, but setting the flag everywhere keeps it that way, as the JVM notes ask.
- **Evidence:** `OUT/scripts/I4-10.clj` (the warnings above); `OUT/scripts/I4-load.clj` (none with the files' own flags).
- **Direction:** Delete the hints and the import, and set the flag in all sixteen namespaces.
- **Known?** new.

### I4-11: Docstring reflow artifacts
- **Severity:** minor
- **Where:**
  - `grammar.clj:6-11`: the second form `(grammar vocab language {:end-tokens ts})` was inserted inside the first form's description, splitting "a byte vocabulary … (a vector of byte arrays by token)".
  - `bytes.clj:5-7`: the arrows are misaligned, and the vocabulary's description reads as the trie's.
  - Ragged lines: `bytes.clj:227-228`, `constrained.clj:17-18`, `smc.clj:57-59`.
  - Overlong lines: `recurrent.clj:18`, `smc.clj:76` (a 116-column comment).
- **Direction:** Reflow these.
- **Known?** new.

## Decisions worth revisiting

- **§9, grammar-constrained decoding, "Byte-level marginals (`bytes/byte-lm`)".**
  This concerns only the namespace, not the semantics. `gen.llm.bytes` holds two ideas:
  - the vocabulary's byte trie and its sweep, grammar machinery (`bytes.clj:1-191`);
  - an LM's text probability over its tokenizations, an inference API (`bytes.clj:193-333`).

  The docstring itself introduces the second half with "And". A separate `gen.llm.byte-lm`, requiring `gen.llm.bytes`, would leave each namespace with one idea. The callers would change from `bytes/byte-lm` to `byte-lm/byte-lm`.
- **§3, principle 4 ("Mutation is confined to three places").** This was written before M3. Four mutable pieces of the LLM runtime are sanctioned by §9's LLM entries but sit outside the three named places:
  - the LM's cache, a volatile in `gen.llm.cache` above the Branching engine;
  - the slot engine's books;
  - the llama engine's replay;
  - smc's shared pool.

  A sentence in §3.4 saying that the LLM runtime's engines and caches are membrane state would keep the principle true as written.

## Checked and sound

- **Warnings (ran):** `OUT/scripts/I4-load.clj` loads all 16 namespaces: no reflection or boxed-math warning. With boxed warnings forced on in the six namespaces that do not set them (`OUT/scripts/I4-10.clj`), only bookkeeping boxes (I4-10).
- **Hot loops (read):** these are primitive and allocate only their outputs:
  - `token/inverse-cdf`;
  - constrained's `fold-weights`, a macro that a function could not replace without boxing, the same reason as the `defunary` kernels;
  - `bytes/sweep` and `log-mass`;
  - `dfa/walk` and `step`;
  - grammar's mask-to-row bit loop;
  - `LlamaRuntime`'s `logSumExp`, `logSoftmax` and `Batch`.
- **House rules (read):**
  - The engine and LM layers and the host draws compute on host floats with clojure.core, as §9 decides, so rule 4's referred operators correctly do not appear.
  - Token draws come from gen.random with site keys (`(fold-in key i)`).
  - There is no `eval!` and no backend call; `array/backend-named :jvm` for host uniforms is gen.array's API.
  - Batched execution is detected with the same idiom as `gen.custom` and `gen.dist.host`.
  - Errors are ex-info with `:reason` throughout, except I4-03 and I4-07.
- **Branching and slots (read):**
  - Consumption and liveness, and the books as pure transitions.
  - Thread confinement through `run-on`, with an interrupt-safe `CompletableFuture/join`.
  - One close path through the `Cleanable`; resample's placement as a pure, tested function (`child-slots`); the scratch slot for `score`.
  - The earlier quality review's structure (`docs/research/llm-branching.md`) holds.
- **Cache, trie, cells (read):**
  - The cache's state is keyed by ids and tokens, never by branches, so walking it is deterministic, as documented.
  - Eviction, pinning and the consume-when-short rule.
  - The answer rounds (`answer-round!`, `still-steps`, `firsts`, `fitting`) match the docstring.
  - The trie and the ledger are small persistent values.
- **Regex and DFA (read):** a normal form for terms that keeps derivatives finite; a construction-local memo; Hopcroft minimization; canonical breadth-first numbering, so two DFAs of one language are `=`. All of it is idiomatic and well documented. The parser could move into a namespace of its own, but needn't: `gen.llm.regex` (611 lines) is one idea, as is `gen.llm.llama` (642). I recommend splitting neither.
- **Java (read):** `LlamaRuntime` covers three things:
  - the upcall swallows everything, as an upcall must;
  - errors and captured log text are kept per thread, and the version is checked at load;
  - the batch's views are made once.

  Every native resource is owned on the Clojure side (the undo list, the Cleanable).
- **Dead code (grep):** every public var is used in src, test or bench, and every private function has a caller. The only leftover is one unused import (I4-10).
- **Printing:** generative functions print as `#gen/…`, as in the core; runtime objects print as `#<…>`.

## Coverage

Every line of each file, read in the snapshot worktree:

| File | Lines read |
|---|---|
| `src/gen/llm/branching.clj` | 1-254 |
| `src/gen/llm/slots.clj` | 1-480 |
| `src/gen/llm/llama.clj` | 1-642 |
| `src/gen/llm/llama/recurrent.clj` | 1-234 |
| `src/gen/llm/cells.clj` | 1-80 |
| `src/gen/llm/trie.clj` | 1-176 |
| `src/gen/llm/cache.clj` | 1-447 |
| `src/gen/llm/lm.clj` | 1-187 |
| `src/gen/llm/token.clj` | 1-520 |
| `src/gen/llm/potential.clj` | 1-103 |
| `src/gen/llm/constrained.clj` | 1-243 |
| `src/gen/llm/smc.clj` | 1-375 |
| `src/gen/llm/bytes.clj` | 1-333 |
| `src/gen/llm/regex.clj` | 1-611 |
| `src/gen/llm/dfa.clj` | 1-290 |
| `src/gen/llm/grammar.clj` | 1-250 |
| `src-java/gen/llama/LlamaRuntime.java` | 1-242 |

**Context read:**
- CLAUDE.md, and DESIGN.md §1-§8 in full plus §9's LLM entries (lines 1138-1430);
- `docs/research/llm-branching.md`: the API, lifetimes and threading, and both reviews;
- the headers of the toy engine and tokenizers, and of `constrained_test.clj`;
- `gen.inference.particle-filter`'s `resample`, `maybe-resample` and `log-ml-estimate`, for comparison;
- beans searched for every finding, with none open.

**Scripts:** `OUT/scripts/I4-01.clj`, `I4-02.clj`, `I4-03.clj`, `I4-06.clj`, `I4-07.clj`, `I4-10.clj` and `I4-load.clj`. Each ran alone, on the JVM backend, against the toy engines.
