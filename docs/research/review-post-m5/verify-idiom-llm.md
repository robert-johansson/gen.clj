# Verification: idiom-llm.md

Snapshot 34e9a44, 2026-10-22, fresh-context verifier.

I read every cited range in the snapshot worktree, and what calls it. I reran each of the
reviewer's scripts (`I4-01`, `I4-02`, `I4-03`, `I4-06`, `I4-07`, `I4-10`, `I4-load`) alone on
the JVM backend, against the toy engines. Every one reproduced its quoted output exactly. I wrote
independent checks for the two correctness claims and for the trie and hint claims:

- `OUT/scripts/V-I4-01.clj`
- `OUT/scripts/V-I4-02.clj`
- `OUT/scripts/V-I4-07-10.clj`
- `OUT/scripts/V-I4-10-tag.clj`

I found no open bean for any finding. `beans list -S` returns only closed beans or unrelated open
ones, and none of the open LLM beans covers these findings. None of the findings is a recorded
decision. Three findings have factual slips that do not change their verdicts:

- I4-04 and I4-10 say smc does not import `Callable`. Clojure imports it by default.
- I4-03's "eight places" counts five sites that already call `br/index?`.

Several directions need a caveat, given under each finding.

## Verdicts
| ID | Claim (short) | Verdict | Severity (reviewer → yours) |
|---|---|---|---|
| I4-01 | smc's copy of the completion's argument check is weaker; a completion's own `:potential` is ignored | CONFIRMED | moderate → moderate |
| I4-02 | the end-token agreement is a type check on `Grammar`; a product, or token SMC, skips it | CONFIRMED | moderate → moderate |
| I4-03 | "a token of the vocabulary" restated; bytes' and grammar's copies regress M6 | CONFIRMED (count overstated) | moderate → moderate |
| I4-04 | the host thread pool written twice (llama, smc) | CONFIRMED (one sub-claim wrong) | moderate → moderate |
| I4-05 | constrained works on token's private run state; `fresh` means two things | CONFIRMED | moderate → moderate |
| I4-06 | smc's key docstring and lm's "in the order they arrive" are wrong | CONFIRMED | moderate → moderate |
| I4-07 | trie `remove*` overflows the stack; two cleanup paths lose an error | CONFIRMED | minor → minor |
| I4-08 | six small idioms | CONFIRMED (one direction needs care) | minor → minor |
| I4-09 | names that shadow core fns or mean two things | CONFIRMED | minor → minor |
| I4-10 | redundant hints, an unused import, the boxed-math flag set unevenly | CONFIRMED (one sub-claim wrong) | minor → minor |
| I4-11 | docstring reflow artifacts | CONFIRMED | minor → minor |

## Findings

### I4-01: CONFIRMED

**Read.** `smc/check-args` (`smc.clj:85-93`) checks only shapes: two arguments, a non-empty
sequential prompt, `nat-int?` max-tokens. `token/check-args` (`token.clj:329-354`) also checks
`br/index?` on every prompt token and that prompt plus max-tokens fit `(:limit gf)`. Its docstring
says the reason: "so that their validity never depends on the draws". In smc:

- `initialize` takes any `gen.llm.token.Completion`.
- `step` draws from `lm/batch-next-logprobs` rows (`proposed`, `smc.clj:206-217`).
- `weighed` (`smc.clj:228`) scores with `lm/score` of the completion's `:lm`.

None of these reads the completion's `:potential`. `traces` (`smc.clj:362-375`) calls
`gfi/generate`, which runs the full check.

**Reviewer's script.** `I4-01.clj` reproduced exactly:

- smc accepts `[[95 500] 3]` and `[[95] 100]` with `:limit` 64.
- traces of an early-ending run throws `:context-full`.
- The particles break `[a-c]+`, and the traces score them `##-Inf`.

**Independent check** (`V-I4-01.clj`), with a different model and other cases:

```
gfi/simulate [[95] 100]: :context-full
smc/run [[95] 100], 4 particles, keys 0..11: [:no-error :no-error :context-full :no-error ...]   ; key 2 fails, the rest succeed
smc/initialize [[95 -3] 3]: :no-error      smc/step of it: :invalid-token
smc/initialize [[95 :a] 3]: :no-error      smc/step of it: :invalid-token
;; constrained completion under [a-c]{2,3}, smc twisted by the same grammar, :masked:
;; every particle satisfies the grammar, yet
particles' :log-p (LM's p) [-16.5775 -16.5775 -12.2194 -24.5281 -12.2194 -13.7792]
traces' :score (q)         [-0.8237 -0.8237 -1.6335 -6.7251 -1.6335 -1.6263]
the unconstrained completion's scores of the same tokens: [-16.5775 -16.5775 -12.2194 ...]   ; = :log-p
```

So whether a run succeeds depends on the key, which is exactly what the completion's check exists
to prevent. With a constrained completion, smc's `:log-p` is the unconstrained p. The traces'
scores are the completion's q, even with no grammar violation. The smc docstring makes two claims
that both fail here:

- "their scores are the particles' log p";
- "p the completion's density".

There is one nuance. For an out-of-vocabulary prompt token, the failure does not depend on the
draws. It is a late but deterministic `:invalid-token` from the cache at the first step. Only the
`:limit` case depends on the draws.

**Severity.** Moderate is right for the idiom review: a duplicated check that drifted, with real
consequences, which belong to the PPL review.

**Direction.** Sound: reuse token's check, and refuse a completion with a `:potential` or define
its target. One part is unsound: "Check the twist once at initialize … rather than its size per
particle per step in `twist-row`" would drop a guard.

- `potential/potential` lets user functions return any array on any call.
- `product` checks only that its members agree with each other.
- For the `:lm` and function proposals, `twist-row` is the only size check.
- A longer array would make `end-entry` read the wrong entry silently.
- The completion itself checks every call, in `constrained/weighted`.
- The per-call check costs one `alength` against a `logw-next` call.

Keep it.

### I4-02: CONFIRMED

**Read.** `check-potential` (`constrained.clj:197-218`) compares end tokens only under
`(grammar/grammar? potential)`, an `instance?` test (`grammar.clj:74`). The `gen.llm.grammar`
require is used only at `constrained.clj:213-217` (grepped). `potential/product`
(`potential.clj:85-103`) is a bare reify. `smc/initialize` calls no potential check. A product
holding a grammar is a documented use: `docs/research/llm-grammar.md:50` gives the completion's
`:potential` as "any potential: a grammar, a product, a weighted one".

**Reviewer's script.** `I4-02.clj` reproduced exactly:

- `:mismatched-end-tokens` bare;
- `:no-error` under `product`;
- `{:zero-weight 10, :no-error 10}`;
- smc `:no-error`, then log-ml `##-Inf`.

**Independent check** (`V-I4-02.clj`). DESIGN's own failing example (`[ab]{3}` and then a token
that is an end token), with a two-member product and a wrapper:

```
1. bare g, completion :end-tokens #{14}:        :mismatched-end-tokens
   (product g soft), completion #{14}:          :no-error
   simulate [[95] 6] keys 0..9: {:zero-weight 10}
2. potential/potential over g's functions, #{14}: :no-error
3. smc/run, bare g as twist, :lm proposal: log-ml ##-Inf dead 32 of 32
   the matched grammar of the language without the end's byte, [ab]{3}, end tokens #{14}: [-17.45 {[66 65 65 14] 31, ...}]
```

By smc's documented target, Z = Σ p(x)·exp(complete(x)) is exactly 0 in case 3, because no
sequence has weight. So the -Inf is the right number, but for a target the user did not mean, and
it comes silently. The completion path at least fails loudly (`:zero-weight`), but mid-run,
depending on the draws for some languages, which is what §9's decision meant to rule out.

**Decided?** No. §9 ("A grammar knows its end tokens") decides the agreement for "a completion
under a grammar", and gives sharing grammars with token SMC as the reason grammars stay
potentials. It does not decide that wrapped grammars or SMC twists skip the check.

**Severity.** Moderate, agreed.

**Direction.** Sound in aim, with three caveats:

1. Adding `-end-tokens` to the `Potential` protocol breaks every reify of `Potential` outside the
   namespace that lacks the method: calling it throws `AbstractMethodError`. A separate optional
   protocol, which `Grammar` and `product` implement and `check-potential` tests with
   `satisfies?`, avoids that.
2. `check-potential`'s messages say "token/completion:", and its zero-prefix rule would be new
   behaviour for smc, which now returns a -Inf estimate. Give it a `what`, and decide whether smc
   should refuse a zero potential. The test `particles-all-of-weight-0-estimate-z-as-0` uses a
   potential whose `prefix []` is 0, so that test is unaffected.
3. It is still a protocol change, which §9 weighed in another form. Record it there for the owner.

### I4-03: CONFIRMED (the count is overstated)

**Reviewer's script.** `I4-03.clj` reproduced:

```
token/completion :end-tokens #{big} :invalid-end-tokens
bytes/byte-lm    :end-tokens #{big} java.lang.IllegalArgumentException
grammar/grammar  :end-tokens #{big} java.lang.IllegalArgumentException
```

**Read.** Both copies coerce before comparing, `(< -1 (long %) (long n-vocab))`, at
`bytes.clj:236` and `grammar.clj:122`. M6 of the Branching quality review
(`docs/research/llm-branching.md:665`) fixed this class of bug. The three readers of the
end-token option (token, bytes, grammar) each restate the default (`:eog`), the check and the
`(into #{} (map long))` normalization.

The "eight places" are not eight restatements. Five of the cited sites already call the rule,
`br/index?`:

- `branching.clj:131-142`;
- `cache.clj:372`;
- `token.clj:340`, `366`, `375` and `515`.

Four sites restate it:

- `lm.clj:77` (`check-tokenizer`'s `:eog`);
- `lm.clj:115-120` (`check-token`);
- `bytes.clj:236`;
- `grammar.clj:122`.

The two in lm compare generically and are big-integer safe. The messages are duplicated too:
`cache.clj:373` against `branching.clj:82-84`.

**Severity.** Moderate, agreed: rule 6 fails in two public constructors, and one option has three
readers.

**Direction.** Sound, with one caveat. The cache's `check-context` checks every context on every
read without allocating (`dotimes`). Branching's `vocabulary-tokens` builds a vector of longs with
`mapv`, so reusing it there would allocate on each read. Share the predicate and one error
constructor, not the vector builder.

### I4-04: CONFIRMED (one sub-claim wrong)

**Read.** `llama.clj:134-151` (`row-pool`, `in-parallel`) and `smc.clj:95-124, 244-256`
(`host-pool`, `host-threads`, `in-order`) share three things:

- the same fixed pool of daemon platform threads;
- the same default size, `(max 1 (min 4 (quot processors 2)))`;
- the same ordered `invokeAll` map that unwraps `ExecutionException`, with `mapv` for one element.

They differ only in `bound-fn*` and smc's `false` pool. smc's docstring ties them together ("as
the llama.cpp engine's row pool").

**Wrong:** "smc does not even import `Callable`". `java.util.concurrent.Callable` is one of
Clojure's default imports: `(ns-imports 'gen.llm.smc)` and `(ns-imports 'user)` both hold it
(`V-I4-10-tag.clj`, case 4). So the hint at `smc.clj:255` resolves.

**Severity.** Moderate is fair. The scale lists "duplication of real logic" under major, but this
is about ten lines.

**Direction.** Sound. Prefer a small namespace of its own over `gen.llm.slots`, which is the slot
engine on its thread, a different idea. Keep separate pool instances: one shared pool between the
engine's row normalization and SMC's host work could meet the queued-behind deadlock that smc's
docstring warns of.

### I4-05: CONFIRMED

**Read.** Two kinds of coupling:

- `constrained/draw-next` and `run-scores` (`constrained.clj:159-195`) destructure the completion
  (`:lm :potential :end-tokens`) and the run state (`:ctx :tokens`, `:dead?` via `dead?`).
- `token/append` sets `:dead?` (`token.clj:232-241`), but token's state comment
  (`token.clj:210-213`) does not list it.

Two misses of the same kind are not in the report: constrained's own section comment
(`constrained.clj:137-139`) gives the state as `{:ctx :tokens :scores}` without `:dead?`. And the
names clash, as the report says:

- token's `draw-next` returns `[t nil]` or `[t [score]]`;
- `scored`'s parameter `fresh` (`token.clj:275, 283`) is that vector of scores;
- the state's `:fresh` is positions (`token.clj:304`).

The coupling is documented in constrained's section comment, so it is not hidden, but both
comments disagree with the code.

**Severity.** Moderate by the scale (a misleading name, a stale comment), though borderline minor.

**Direction.** Sound.

### I4-06: CONFIRMED

**1. The key in smc's docstring.** `I4-06.clj` reproduced: smc drew `[57 22 74 86 77 74 72 55]`
and the docstring's keys give `[55 90 28 69 72 6 90 86]`. In the code (`smc.clj:82-83, 275, 277`)
`step-key` folds in the purpose, and the draws fold in `i`: particle i's token comes from
`(fold-in (fold-in (fold-in key t) 0) i)`. The state comment (`smc.clj:78-80`) and
`docs/research/llm-token-smc.md` ("Keys") are both right, so the namespace docstring
(`smc.clj:45-46`) is the only one that is wrong.

**2. lm's "in the order they arrive".** `CachedLM` (`lm.clj:51-59`) serializes calls with
`locking`, a JVM monitor, which is not fair. So `lm.clj:31-32` is wrong.

**Severity.** Moderate, agreed.

**Direction.** The first option, saying that no order is promised, is the right one. A fair
`ReentrantLock` gives arrival order, but arrival order is itself a matter of timing. So it would
not make concurrent results "a function of the program", as the same docstring claims for
batching. It would cost something without buying that property.

### I4-07: CONFIRMED

**Reviewer's script.** `I4-07.clj` reproduced the `StackOverflowError`.

**Independent check** (`V-I4-07-10.clj`) shows a second shape: a path that ends inside an edge
throws `IndexOutOfBoundsException`.

```
remove-entry [3 4] (absent):          java.lang.StackOverflowError
remove-entry [1 2] (inside an edge):  java.lang.IndexOutOfBoundsException
remove-entry [1 2 3] (held):          :no-error
```

This cannot arise from `src`:

- trie's namespace docstring states the precondition ("remove-entry takes the tokens of an entry
  the trie holds", `trie.clj:18-19`);
- the only caller (`cache.clj:98`) passes an entry's own branch tokens;
- `insert` is loud with ex-info for its own misuse (`:entry-exists`).

The cleanup paths, read:

- `cache.clj:175-177` swallows a failed `dispose!` deliberately ("the engine frees them itself if
  it is closing"), rethrowing the primary;
- `llama.clj:511-513` runs undos inside the catch, so a throwing undo replaces `t`.

**Severity.** Minor is right.

**Direction.** Sound: ex-info for a missing child or a short path, and `.addSuppressed`.

### I4-08: CONFIRMED (one direction needs care)

Each item, read:

- `trie.clj:57` `(if here here entry)` is `(or here entry)`, with the same semantics.
- `grammar.clj:212` restates `allowed?`'s bit test (`grammar.clj:205`).
- `smc.clj:270` `(filterv #(not …))`. The equivalent is `(into [] (remove …) …)`, since `live` is
  used as a vector.
- `llama.clj:335`'s `:model string?` cannot fail after `llama.clj:347`. **Caveat:** `option-checks`
  is also the set of known options (`(remove option-checks (keys opts))`, `llama.clj:342`).
  Dropping `:model` from it would make `:model` an "unknown option". Keep the key and check it in
  one place, for example by folding the file test into the map.
- `split-recurrent?` is called at `llama.clj:489` only for its error, and it reads the property up
  to three times, but only on the error path (`llama.clj:192, 197-198`).
- `recurrent.clj:231-232`. With `full = (decode st seqs positions)` (`recurrent.clj:206`),
  `(cost st seqs positions whole fixed)` decodes the same group again and returns
  `fixed + (extra full)`. `(mapv seqs (range n))` is `seqs` for a vector. Derived from `cost`
  (`recurrent.clj:165-173`), so the replay is redundant.

**Severity.** Minor.

### I4-09: CONFIRMED

Each name, read:

- `first` is a field and local at `bytes.clj:36, 64, 133, 140, 163, 202`, which forces
  `clojure.core/first` at `bytes.clj:71`.
- `next` is a local at `bytes.clj:279, 325`, `grammar.clj:99, 107` and `potential.clj:79`.
- `byte` is a local at `dfa.clj:233, 243, 255`.
- The two private `state` functions differ: the volatile at `slots.clj:167`, its value at
  `cache.clj:72`.
- `weighed` (`smc.clj:219`) sits beside constrained's `weighted`.
- "conj!ed" is used where the code calls `vswap! … conj` (`cache.clj:167` and `221`, with
  `vswap! made into` at `223`).

None of the shadowing causes a bug: I checked that `bytes.clj:284`'s `first` is core's, since the
function there shadows only `next`.

**Severity.** Minor.

### I4-10: CONFIRMED (one sub-claim wrong)

**Reviewer's scripts.** `I4-10.clj` reproduced the boxed warnings at `recurrent.clj:133, 183,
226, 229` and `lm.clj:117`. It also produced `gen/array/memory.clj:26`, which is outside this layer
(the I1 reviewer's file). `I4-load.clj` prints no warnings.

**Hints.**

- `vswap!` tags its target itself. The macroexpansion is
  `(.reset ^clojure.lang.Volatile v (assoc (.deref ^clojure.lang.Volatile v) …))`.
- The parameters at `llama.clj:207, 213, 221, 242` reach no other interop.
- The `^clojure.lang.Volatile` at `llama.clj:626` repeats the deftype field's tag, which is
  already `Volatile` (`llama.clj:250`).
- `Callable` at `slots.clj:39` is unused (grepped).

**Wrong:** the claim that smc "does not import" `Callable`. It is a default import, as in I4-04.

**Imprecise:** "no effect". A tag on a fn form is compiled as runtime metadata.
`(meta (let [x 1] ^Callable (fn [] x)))` is `{:tag java.util.concurrent.Callable}`, and a tag
naming no class fails to compile, "Unable to resolve symbol" (`V-I4-10-tag.clj`). It changes no
dispatch, though, since a Clojure fn is already a `Callable`, so deleting it stays right.

**Caveat on the flag.** `*unchecked-math* :warn-on-boxed` also makes long arithmetic unchecked
(overflow wraps). In these namespaces' counts, ticks and positions that is harmless, but the
change switches semantics as well as warnings. CLAUDE.md's note asks it of kernels, so setting it
everywhere is a consistency choice.

**Severity.** Minor.

### I4-11: CONFIRMED

Read:

- `grammar.clj:6-11`: the second form is spliced into the first form's description.
- `bytes.clj:5-7`: the arrows are misaligned, and the vocabulary's description follows "the trie
  of the tokens' bytes".
- The ragged lines at `bytes.clj:227-228`, `constrained.clj:17-18` and `smc.clj:57-59` are real.
- Overlong lines, by `awk`: `recurrent.clj:18` is a 99-column docstring line, past its neighbours' width, and
  `smc.clj:76` is a 116-column comment.

**Severity.** Minor.

## Missed (optional)

- `constrained.clj:137-139`'s comment omits `:dead?` from the run state (folded into I4-05).
- `trie/remove-entry` of a path ending inside an edge throws `IndexOutOfBoundsException` (folded
  into I4-07).
- `option-checks` doubles as the registry of known options (folded into I4-08's caveat).

The two "Decisions worth revisiting" are suggestions, not defects. Their facts hold:

- `bytes.clj`'s docstring introduces the byte-LM half with "And".
- The LLM runtime's volatiles and smc's pool lie outside the three places that DESIGN §3.4 names.
