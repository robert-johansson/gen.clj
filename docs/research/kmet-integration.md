# gen.clj and the kmet agent: a design note

*2026-10-20. Prompted by the owner: GenMLX, through mlx-node, aimed to
integrate with the pi coding agent, and kmet (a Clojure port of pi) could
play that role for gen.clj. A draft for milestones M6 and M7: nothing here
is decided beyond writing it down. Beans: M6 genclj-ytew, M7 genclj-tw4p,
and M6's first spike genclj-kaa8.*

*Revised 2026-10-21: the owner scrapped the JVM fork; `~/code/kmet-agent`
is upstream itself, which now runs on the JVM. §2, §4, §7 and §8 are
rewritten against upstream master at febc5aa7 (2026-10-21), from a
read-only survey of its code; file:line references are to that commit.*

*Revised 2026-11-04 (upstream at a4f7102d; the integration is paused, see
genclj-ytew's revision for the details):*
- *kmet2 has landed: agent handles, composition rows, and a server mode
  where a connection is an agent.*
- *The in-process provider seam of §4 now exists: the `:llm` row
  (`kmet.app.llm`, `{:send-message …}`) is replaced by a profile patch
  naming a host-classpath plugin, so compiled gen.clj can serve kmet's
  turns without an upstream change. Log-probabilities are still absent in
  kmet, so gen.clj keeps them.*
- *kmet's clojure extension evaluates in a running nREPL (`clojure_eval`).*
- *A per-agent persistent `eval` tool is planned upstream.*
- *gen.clj has since gained the chat template, sampler, agent loop and
  ext/codegen (genclj-uw7w), and is MIT-licensed.*

## 1. The idea

GenMLX treats a local LLM as a generative function. pi is a coding agent:
a model proposing tool calls in a loop. Put the two together and an agent's
behavior becomes something a probabilistic program can score, edit, condition
and search over. GenMLX started on this (§6); its glue lived across a
TypeScript host and nbb, and is the part of GenMLX graded C. kmet on the JVM
removes that boundary: the agent and gen.clj share one language, one runtime
and one process.

The thesis worth testing: **inference-time computation makes small local
models competent agents.** SMC over trajectories weighted by verifiers,
posterior-guided repair of a failed step, and the probability of success are
what a PPL contributes and an inference server cannot.

## 2. kmet today (upstream, `~/code/kmet-agent` at febc5aa7)

- **What it is.** A coding agent in Clojure, inspired by pi: a terminal UI,
  streaming conversations, and the tools read, write, edit, bash and
  `run_code`, with skills, prompt templates, packages and an EDN RPC mode.
  It's `kmetia/kmet-agent` (Marko Kocic, EPL-2.0), on Babashka (still "the
  primary runtime", `README.md:7`), Jolt, and since 2026-10-20 the JVM. It
  moves fast: 771 commits in 30 days, nearly all the maintainer's, with
  namespaces moved often; `kmet.extension` is "the published ABI".
- **The JVM host is upstream's** (13ca1adb and after). `bb start-jvm` runs
  `clojure -M:jvm:clj-pin -m kmet.agent.core` (`bb.edn:30-32`; Clojure
  1.12.5 pinned, `deps.edn:164`; `--enable-native-access=ALL-UNNAMED`,
  `deps.edn:171-181`); `bb dist-jvm` builds an AOT zip, about 2 s to first
  render. The JVM takes the `:default` reader branch; `.cljc` is shared.
  `bb test-jvm` exists, but upstream's CI runs only an attribution check, so
  the JVM tests are a local gate. The owner reports it runs very well.
- **Extensions and `run_code` are SCI only on the JVM** (`docs/building.md:121`,
  `extensions/context.cljc:1513-1567`; the JVM's native loader, "Phase 3",
  is not built, `loader/loader.md:32-34`). SCI records can now implement
  kmet's own compiled protocols (86f0dcf1), but an extension's
  dependencies are interpreted from source, Java classes never reach a
  classloader, `:local/root` dependencies are dropped off Jolt
  (`context.cljc:990`), and `definterface` and non-protocol `deftype` fail.
  So gen.clj (deftypes, FFM bindings) cannot load as an extension, and a
  `run_code` script cannot require it (`tools/run_code.cljc:473-614`).
- **Sessions** (`modules/agent/src/kmet/agent/session/`, split into store,
  listing, context, stats and tree): EDNL, one EDN map per line,
  `{:id :parent-id :role …}`, a tree with a live leaf. `append-entry` adds a
  child of the leaf; `branch!` and `reset-leaf!` move the leaf, deleting
  nothing (`store.clj:331-418`). Entries are open maps. Extensions append
  `:custom` entries and `:context-edit`s, and `:turn-end` carries the
  turn's `:message-entry-id`; only host code appends an `:assistant` entry
  at a chosen parent.
- **Providers** (`modules/ai/src/kmet/ai/llm.clj:76-196`): `send-message`
  dispatches through a closed table of nine HTTP wire APIs
  (`wires.clj:22-49`); a provider is registered by `:base-url` and `:api`,
  and a local model is a custom provider with `:api :openai-completions`
  (`docs/providers.md:71-80`; llama.cpp's `reasoning_content` is parsed).
  There is still no in-process provider ("kmet has no streamSimple",
  `provider_composer.clj:180`; listed as a pi-parity gap,
  `development/pi-alignment.md:402`), and no log-probabilities anywhere:
  the stream parser and the stored assistant message drop them
  (`api/sse.clj:31-90`, `loop/state.clj:319-337`).
- **Constrained sampling** (`constrained_sampling.clj:190-215`): the grammar
  helpers still have no callers. A model's `:sampling-params` and the
  `:before-provider-request` event can set a request's `grammar` or
  `json_schema`.
- **The loop** (`loop.clj` and `loop/{completion,state,tool_exec,message}.clj`):
  events before a model call (`:context`, `:before-provider-request`, …),
  after it (`:message-end`, `:turn-end`), and around tools (`on-tool-call`,
  `on-tool-result`; tools run in parallel); no event for session writes and
  none that sees a provider's metadata. The app holds about 18
  process-global states (`new-app.md:190`), and `-main` calls
  `System/exit`. A redesign, kmet2 (`new-app.md`, `srce.md`), would put
  sessions behind a service and give each agent a handle; it is a plan so
  far.

## 3. What integration could mean, by depth

1. **A local-model provider.** gen.clj's LM protocol (genclj-mse0) serves
   kmet's turns from a local model, with what hosted APIs cannot give:
   - every token's log-probability;
   - cheap forks of a context;
   - grammars that make a malformed tool call impossible, not merely
     unlikely (GenMLX's `toolcall.cljs`).
2. **Sessions as traces** (genclj-p79i):
   - An assistant turn is the token generative function (genclj-0cvv) at its
     rendered prompt.
   - A session's score is the sum of its turns' assess weights.
   - "Step 3 was wrong" is regenerate from a token or tool-call boundary,
     whose dependence cone is the suffix. The result rejoins kmet's tree as a
     new branch, with nothing deleted.
3. **The agent as a generative function.**
   - A trajectory is a trace: tool calls are choices of the policy (the LLM
     generative function), and tool results are observations of the
     environment.
   - The inference library then applies to agent behavior:
     - SMC over trajectories, weighted by a verifier such as "the tests
       pass";
     - MH that repairs a failed step;
     - importance estimates of the probability of success;
     - inverse planning from a session to the user's goal (genclj-xl5e,
       genclj-nt3u).
4. **The LLM proposing programs** (genclj-0izr, GenMLX's MSA): the agent
   writes gen.clj models through `run_code`, gen.clj scores them against data
   (marginal likelihood through the GFI), and the agent iterates with the
   scores as feedback. That's a modelling assistant.
5. **Learning:** session scores as rewards or likelihoods for SFT or GRPO on
   a local model (genclj-q4xv). It needs gradients, so the MLX-native engine
   (genclj-gvfa).

## 4. Architecture

- **Neither core depends on the other.** gen.clj stays Clojure only (CLAUDE.md
  rule 8), and kmet stays upstream's, untouched. The integration is a third
  artifact: a JVM library that depends on both (kmet as a git dependency at
  a pinned SHA) and launches kmet's JVM host with gen.clj on the same
  classpath, plus, if commands or UI want it, a thin SCI extension that
  calls into it. It cannot be a kmet extension itself (§2: extensions are
  SCI only on the JVM). It relies on `kmet.extension` and as few host
  namespaces as it can, since upstream moves namespaces often.
- **One hook kmet needs:** an in-process provider, a streaming function
  taking kmet's messages and tools in place of `:base-url` and `:api`, with
  per-message metadata (log-probabilities, the keys a turn was drawn with).
  It is pi parity (`streamSimple`) that kmet already lists as a gap, and it
  must work on all three hosts (kmet's AGENTS.md avoids Java interop), so it
  is a fair upstream proposal, the owner's to make. Until then gen.clj
  serves an OpenAI-compatible endpoint, which kmet takes today as a custom
  provider: kmet sees only text and tool calls, so gen.clj keeps each
  turn's log-probabilities, keys and forks on its side, and host code
  writes the scores into the session as `:custom` entries keyed by
  `:message-entry-id`. Grammars still apply, since gen.clj is the sampler.
  Patching kmet's wire table from host code would also work, but is
  unsupported and fragile.
- **Rendering is shared.** kmet converts a session to messages, and the
  model's chat template renders them to tokens: the GGUF's template through
  llama.cpp, or Jinjava (DESIGN §6). A turn's score means something only if
  that rendering is the token stream the model saw. So the integration owns
  one renderer, used both to serve and to score.
- **Randomness is gen.clj's.** Engines never sample (DESIGN §9), so with
  gen.clj as the provider every token is drawn with a site's key. A session
  records its key and can be replayed exactly. That's reproducible agent
  evaluation, which hosted APIs cannot offer.
- **Memory:** an agent, a resident model and MLX's arrays in one JVM. The
  shared budget (genclj-zztc) becomes a requirement, not an option.
  Coding-capable local models (Qwen3-Coder at 30B-A3B and up) belong on the
  64 GB MacBook or the CUDA hosts; the 32 GB Mac runs the small models.

## 5. Laws (what the tests would hold)

- **Rendering parity.** The rendered prompt of turn k (messages before k, with
  the generation prompt) is a prefix of the rendered session through turn k.
  Otherwise it is a loud error, where GenMLX's `pi_assess` reports `:parity?`
  and scores anyway.
- **A session's score is its turns' scores.** The assess of the session's
  token generative function equals the sum of the per-turn scores. It is
  exact if the token generative function computes in fixed shapes (the law-2
  decision in genclj-0cvv).
- **An edit is a regenerate:**
  - the prefix is unchanged;
  - resampling the suffix from the model weighs 0;
  - the rejoined branch is a well-formed subtree of the session (its
    `:parent-id`s).
- **Determinism.** The same key and the same session prefix give an identical
  turn.
- **The agent generative function obeys the GFI laws:** `laws/deflaws` over an
  agent with a scripted environment and the toy LM engine, so the laws run
  without a model.

## 6. GenMLX's prior art

| GenMLX | Lines | Grade | What carries over |
|---|---:|---|---|
| `llm/pi_session.cljs` | 291 | C (llm/*) | The session-to-messages mirror; with EDN, no byte-stable JSON tricks |
| `llm/pi_assess.cljs` | 202 | C | Turn scoring at the rendered prompt; its law (assess equals the one-forward session score) |
| `llm/pi_edit.cljs` | 371 | C | Resample from a boundary, rejoin as a branch |
| `llm/pi_provider.cljs` | 772 | C | Serving the agent from the engine: here a library beside kmet, not gen.clj's core |
| `llm/toolcall.cljs` | 545 | C | Tool calls as a grammar over the declared tools |
| `llm/msa.cljs`, `msa_score.cljs`, `world/llm_proposer.cljs` | 498, 231, 442 | C | The LLM proposing programs, with check feedback |
| `examples/agents_llm_policy.cljs`, `agents_llm_inverse.cljs` | 288, 210 | B | The LLM as a policy; inverse planning that fuses an agent model with an LLM's judgment |

Port the ideas, not the code (the owner's bar).

## 7. Dependencies and order

- **From gen.clj** (M3):
  - the token generative function (genclj-0cvv), with exact assess;
  - grammar middleware (genclj-q38h);
  - token SMC (genclj-loyp);
  - tool-call grammars and structured output (genclj-w120);
  - chat templates;
  - for learning, the MLX-native engine (genclj-gvfa) and training (genclj-q4xv).
- **From kmet:** the JVM host (upstream's, done); a stable session format
  (EDNL, unchanged); an in-process provider hook with per-message metadata
  (to propose upstream; the endpoint serves until then). For M7, either one
  kmet process per trajectory over its RPC mode, or kmet2's per-agent
  handles, since today's process-global state allows one agent per JVM;
  and a gen tool registered from host code, since `run_code` cannot load
  gen.clj.
- **Order:** M4, M5, then M3's remainder, then M6, then M7. M5 is independent of
  both. Nothing here should start before the token generative function
  exists, since its two open decisions (exact law 2, branch ownership) shape
  every layer above it.

## 8. Risks and open questions

- **Upstream.** kmet is Marko Kocic's project and moves fast (771 commits in
  30 days at febc5aa7). The library approach needs nothing from upstream;
  the in-process provider hook needs his interest. Ask early (the owner's
  call). kmet's CI rejects commits carrying Claude's attribution trailers
  (`scripts/attribution_check.sh`), so anything offered upstream goes
  without them.
- **The JVM host is new.** One day old at febc5aa7, and its tests run only
  locally: pin a SHA, and run kmet's `bb test-jvm` at each bump.
- **License.** kmet is EPL-2.0, and gen.clj has no license file yet. Choose
  before combining them.
- **Model capability.** Small local models are weak coding agents. The thesis
  (§1) is that inference makes them stronger; the M7 work should measure that
  against the same models without inference, not assume it.
- **Scope.** This is a research vertical (genclj-buyv's question: a separate
  library?). It must not make gen.clj's core heavier.
- **Where the code lives:** a library beside both (the evidence of §2 rules
  out an extension, and the fork is gone). Whether it is a gen.clj
  module, a separate repository, or offered to kmet is the owner's call,
  after the spike.

## 9. Proposed milestones

- **M6: gen.clj inside kmet.** A local-model provider (a library on kmet's
  JVM classpath), sessions as traces (score and edit, genclj-p79i), and
  tool-call grammars. Done means: a kmet session served by a local model
  through gen.clj; its turns scored, with rendering parity a law; an edited
  turn rejoining the tree as a branch; malformed tool calls impossible under
  the grammar.
- **M7: inference over agents.**
  - The agent as a generative function.
  - SMC over trajectories with verifiers, measured against the same model
    without inference.
  - Inverse planning over sessions (genclj-xl5e).
  - The program-synthesis loop through `run_code` (genclj-0izr).
  - Learning, when the MLX-native engine exists.
- **First task (M6): a spike.** Put gen.clj on kmet's JVM classpath (a
  wrapper deps.edn with kmet at a pinned SHA), read one recorded session,
  render its turns with the model's chat template, and score each assistant
  turn. Report:
  - rendering parity;
  - the costs;
  - what kmet would need from upstream.

## 10. The plan (owner, 2026-11-04; DESIGN §9)

*"Write the plan with your suggestions." kmet upstream at a4f7102d; gen.clj
main at 8c30bb4. Every stage below is gated: it starts on the owner's go,
and the next starts only if it answers its question.*

### 10.1 Why now, and what we do not wait for

The seam §4 asked upstream for exists: the `:llm` row
(`kmet.app.llm/plugin`, service `{:send-message (fn [opts] …)}`) is
replaced by a profile patch naming a plugin var, and on the JVM such a var
may be any compiled namespace on the classpath. Three things kmet still
lacks are optional:

- **Log-probabilities.** As the `:llm` plugin, gen.clj draws every token
  itself, so it holds each token's log-probability and the key it was
  drawn with. It records them beside kmet's session, keyed by kmet's
  message or turn id; kmet needs only the text and tool calls.
- **Persistent eval.** kmet's planned `eval` is a per-agent SCI context.
  SCI cannot load gen.clj (deftypes, the FFM bindings: §2), so it would
  not run gen.clj models even once built. gen.clj's own route is
  ext/codegen's session in a gen.codegen.process host, mounted as a tool
  row, or kmet's `clojure_eval` into an nREPL that holds gen.clj and
  memo.clj.
- **Subagents.** M7 needs many independent agents, not agents spawning
  agents. kmet2's handles and server mode (a connection is an agent, with
  its own session and working directory) give that, and within one turn
  the plugin can draw K candidates itself.

### 10.2 Coupling rules

- Only through the `:llm` contract and composition rows (profiles,
  `--patch`, rows naming host-classpath plugins). Never through kmet's
  internal namespaces or its extension API, which moves (it was
  "unfrozen" on 2026-10-30).
- kmet at a pinned commit (a git dependency in `ext/kmet/deps.edn`),
  bumped deliberately after kmet's `bb test-jvm` passes at the new commit.
- Nothing is changed in `~/code/kmet-agent`. Anything offered upstream is
  the owner's to send, without Claude's attribution trailers (kmet's CI
  rejects them).
- License: kmet is EPL-2.0 and gen.clj MIT. The module depends on kmet and
  copies none of its code.

### 10.3 Roles

- **gen.clj** owns the machinery:
  - the local model, sampling and chat format;
  - the keys that make a session replay exactly;
  - code evaluation (ext/codegen);
  - the integration (ext/kmet);
  - and M7's inference, where a trajectory is a generative function.
- **gen.llm.agent** is the research harness, where keys, log-probabilities
  and K candidates are exact. **kmet** is the shell a person uses: its
  TUI, sessions, file tools and bash.
- **memo.clj** supplies:
  - a language the agent writes;
  - its hooks (errors explained, vetting) and tasks, through its
    synthesis module, as its REPL agent (memoclj-pcw0) does;
  - models the agent may reason with (ask or act, the user's intent).
  memo.clj's core takes no LLM code.

### 10.4 The question and the measure

Does a small local model become a useful modelling assistant when it can
evaluate code?

- **Tasks:** the codegen bake-off's ten (ext/codegen, gen.codegen.tasks)
  and memo.clj's synthesis tasks.
- **Measure:** the bake-off's grader:
  - the hard stages (reads, safe, evaluates, simulates, takes the data,
    keeps the laws);
  - explains the data (the log marginal likelihood's gap to the
    reference, by the :checked estimator);
  - Wilson intervals.
- **Baselines,** at matched compute (tokens and seconds): one shot,
  best-of-K with the grader's hard stages as the verifier, and the REPL
  agent in gen.llm.agent. The 3.6's one-shot and REPL results from
  genclj-ihc4 and genclj-t1xh are the first numbers; they are rerun on
  the same seeds when the harness changes.
- **Models:** Qwen3.6-35B-A3B (the MacBook, or fenrir), and Qwen3.5-0.8B
  for development on the 32 GB Mac.

### 10.5 Stages

0. **The question, tasks and baselines** (no kmet). memo.clj's tasks join
   the bake-off's driver as a task set, and the baselines are recorded.
1. **A zero-code probe.**
   - kmet's agent at the pinned commit, with its clojure extension's
     `clojure_eval`, works a few tasks against an nREPL JVM holding gen.clj
     and memo.clj.
   - The model is served by llama.cpp's own server (v0.7.0 rebuilt with
     its tools on, the same GGUF), as kmet's custom provider
     `:api :openai-completions`.
   - The question: what does a general coding agent do with a live REPL
     on these tasks, and where does it fail? This gives qualitative
     notes, not a score.
2. **The `:llm` plugin spike** (genclj-kaa8), in `ext/kmet`:
   - a plugin that takes kmet's messages and tools, renders them with
     gen.llm.chat, samples with gen.llm.sample, parses tool calls, and
     answers through the callbacks;
   - each turn's tokens, log-probabilities and key recorded;
   - a profile that mounts it.
   - **Done means:** one kmet turn served in-process from a local Qwen;
     replaying from its key gives the same turn; the recorded
     log-probabilities equal lm/score's.
3. **A gen.clj eval tool row.** A kmet tool whose calls evaluate in an
   ext/codegen session in a gen.codegen.process host. Its prelude and
   hooks (gen.clj's, or memo.clj's from its synthesis module) come as the
   row's config. **Done means:** kmet's agent, on gen.clj's model, builds
   and checks a bake-off model through it.
4. **Best-of-K with verifiers** (M7's first stage, genclj-tw4p).
   - K candidates per turn, drawn by the plugin or by gen.llm.agent;
   - chosen by a verifier (the grader's stages, memo's laws);
   - one isolated working directory per sample (kmet's server mode gives
     one per connection).
   - **Gate:** beats the REPL agent at matched compute.
5. **Repair and SMC over trajectories,** only if stage 4's gains leave room:
   - MH or regenerate from a turn boundary;
   - SMC weighted by verifiers;
   - the probability of success by importance sampling.

At the spike's end the integration's home is decided (genclj-wqka): stay
`ext/kmet`, or become its own repository.

### 10.6 Risks

- kmet's pace (517 commits in two weeks). Mitigated by the coupling rules
  and the pin.
- The `send-message` contract could change. It is pi parity and the
  builtin's own contract, so it is the most stable seam kmet has.
- A coding agent's file tools and bash in a research loop: each sample
  gets its own working directory, and runs go on hosts the owner grants.
- Memory: a 35B model, kmet and a grader JVM at once. These runs go on the
  MacBook or fenrir, never the 32 GB Mac.
