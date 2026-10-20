# gen.clj and the kmet agent: a design note

*2026-10-20. Prompted by the owner: GenMLX, through mlx-node, aimed to
integrate with the pi coding agent, and kmet (a Clojure port of pi, which the
owner's fork is bringing to the JVM) could play that role for gen.clj. A
draft for milestones M6 and M7: nothing here is decided beyond writing it
down. Beans: M6 genclj-ytew, M7 genclj-tw4p, and M6's first spike genclj-kaa8.*

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

## 2. kmet today (the owner's fork, `~/code/kmet-agent`)

- **What it is.** A coding agent in Clojure, inspired by pi: a terminal UI,
  streaming conversations, and the tools read, write, edit, bash and
  `run_code`, with skills, prompt templates, packages and an EDN RPC mode.
  Upstream is `kmetia/kmet-agent` (Marko Kocic, EPL-2.0), on Babashka and
  Jolt. The owner's fork (`robert-johansson/kmet-agent`) adds the JVM as a
  third host, additively, so it can keep merging upstream (`jvm-plan.md`,
  D1).
- **The JVM host.** Written against e8070561 on 2026-10-19:
  - 212 of 214 namespaces load with two small patches;
  - kmet's fast suite has 3,612 tests;
  - startup is 0.6 s with AOT and JDK 25's AOT cache;
  - phases P4 and P5 have landed since.

  Its one architectural gap is extensions: on the JVM, kmet is compiled,
  and stock SCI cannot implement compiled protocols. The fix is a
  host-type bridge ("SCI+") plus an opt-in compiled engine
  (`jvm-extensions.md`).
- **Sessions** (`modules/agent/src/kmet/agent/session/core.clj`): EDNL, one EDN
  map per line, `{:id :parent-id :role …}`. It's a tree with a live leaf, and
  branching moves the leaf without deleting anything (a port of pi's session
  storage).
- **Providers** (`modules/ai/src/kmet/ai/llm.clj`): `send-message` streams over
  HTTP wire APIs (`:openai-completions`, `:anthropic-messages`, and others).
  An extension registers a provider by `:base-url` and `:api`
  (`models-register-provider!`). There is no in-process provider, a
  streaming function in place of a URL.
- **Constrained sampling** (`modules/ai/src/kmet/ai/constrained_sampling.clj`):
  strict JSON-schema tool calls on every wire pi supports. The grammar helpers
  exist but are never activated, since hosted providers do not take grammars.
- **`run_code`** (`modules/agent/src/kmet/agent/tools/run_code.cljc`): sandboxed
  Clojure scripts that return only what they emit, through SCI.
- **The loop** (`modules/agent/src/kmet/agent/loop.clj`): a state machine (idle,
  thinking, executing), with steering and follow-up queues, retries, a repeat
  guard, and lifecycle events for extensions.

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
  rule 8). The fork's rule D1 keeps kmet's shared files almost untouched, so
  merging upstream stays cheap. The integration is a third artifact: a kmet
  extension for the JVM host, or a small library that depends on both,
  loaded by one. Which one depends on how compiled code loads as a kmet
  extension (D4 in `jvm-plan.md`).
- **One hook kmet may need:** an in-process provider, a streaming function
  taking kmet's messages and tools in place of `:base-url` and `:api`. It is
  host-neutral and small, so it could be offered upstream. Without it,
  gen.clj serves an OpenAI-compatible endpoint. That works today, but loses
  the log-probabilities, the forks and the grammars, the reasons for doing
  this at all.
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
| `llm/pi_provider.cljs` | 772 | C | Serving the agent from the engine: here an extension, not library code |
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
- **From kmet:** the JVM host, with compiled code loadable as an extension; an
  in-process provider hook; a stable session format.
- **Order:** M4, M5, then M3's remainder, then M6, then M7. M5 is independent of
  both. Nothing here should start before the token generative function
  exists, since its two open decisions (exact law 2, branch ownership) shape
  every layer above it.

## 8. Risks and open questions

- **Upstream.** kmet is Marko Kocic's project and moves fast (755 commits in
  30 days, per `jvm-plan.md`). An extension needs nothing from upstream; an
  in-process provider hook needs his interest. Ask early.
- **License.** kmet is EPL-2.0, and gen.clj has no license file yet. Choose
  before combining them.
- **Model capability.** Small local models are weak coding agents. The thesis
  (§1) is that inference makes them stronger; the M7 work should measure that
  against the same models without inference, not assume it.
- **Scope.** This is a research vertical (genclj-buyv's question: a separate
  library?). It must not make gen.clj's core heavier.
- **Where the code lives:** a kmet extension, a library beside both, or a JVM
  module of the fork. The owner's call, after the spike.

## 9. Proposed milestones

- **M6: gen.clj inside kmet.** A local-model provider (an extension on kmet's
  JVM host), sessions as traces (score and edit, genclj-p79i), and
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
- **First task (M6): a spike.** Load gen.clj's LM protocol inside the kmet
  JVM fork, read one recorded session, render its turns with the model's chat
  template, and score each assistant turn. Report:
  - rendering parity;
  - the costs;
  - what kmet would need from upstream.
