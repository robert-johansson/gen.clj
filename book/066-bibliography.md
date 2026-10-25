# Bibliography

This appendix lists the works the chapters cite: the thesis and the papers
of Gen's authors, the papers behind the inference methods, the systems and
libraries the book ports or compares with, and its data. An entry begins
with its key, as the chapters write it: the authors and the year, such as
Green 1995, or a system's name. A chapter may cite a work of three
authors or more by the first: Andrieu et al. 2010 is Andrieu, Doucet and
Holenstein 2010. Then comes the reference, and last the pages that cite
the work, by their titles less any subtitle. A system's reference is the
paper its authors ask users to cite, where there is one, and the version
the book was written against, where that matters.

Not listed: programming languages, the tools that build the book
(*Installing and Running* has them), the repository's own documents,
eponyms such as Jensen's inequality or the Kalman filter, and the works
that only docstrings cite, which *The Public API* quotes.

## The thesis and Gen's papers

Part III follows the thesis definition by definition, and cites its
sections, equations and pages by number. Gen.jl's own paper is the
reference of the entry *Gen.jl*, with the systems.

- **Cusumano-Towner 2020** (*the thesis*). Marco Francis Cusumano-Towner.
  *Gen: A High-Level Programming Platform for Probabilistic Inference*.
  PhD thesis, Department of Electrical Engineering and Computer Science,
  Massachusetts Institute of Technology, September 2020. Cited in *Update
  and Regenerate*, *Speeding Up*, *The Generative Function Interface*,
  *Inference as Operations on the GFI*, *Trace Translators and Involutive
  MCMC*, *Design Rationale*, *Verification*, *Exact Structure, Made
  Explicit*, *Extending gen.clj*, *Data-Driven Proposals*, *Reversible-Jump
  MCMC and Structure Learning* and *Notation*.
- **Cusumano-Towner and Mansinghka 2018**. Marco F. Cusumano-Towner and
  Vikash K. Mansinghka. Using probabilistic programs as proposals.
  arXiv:1801.03612, 2018. Cited in *MCMC and MAP*.
- **Cusumano-Towner, Lew and Mansinghka 2020**. Marco Cusumano-Towner,
  Alexander K. Lew and Vikash K. Mansinghka. Automating involutive MCMC
  using probabilistic and differentiable programming.
  arXiv:2007.09871, 2020. Cited in *Gradient-Based MCMC*.
- **Saad et al. 2019**. Feras A. Saad, Marco Cusumano-Towner, Ulrich
  Schaechtle, Martin C. Rinard and Vikash K. Mansinghka. Bayesian
  synthesis of probabilistic programs for automatic data modeling.
  *Proceedings of the ACM on Programming Languages*
  3(POPL):37:1–37:29, 2019. Cited in *Reversible-Jump MCMC and
  Structure Learning*.

## Inference methods

### Importance sampling and diagnostics

- **Kong 1992**. Augustine Kong. A note on importance sampling using
  standardized weights. Technical Report 348, Department of Statistics,
  University of Chicago, 1992. The measure behind the effective sample
  size. Cited in *The Inference Toolkit*.
- **Vehtari et al. 2021**. Aki Vehtari, Andrew Gelman, Daniel Simpson,
  Bob Carpenter and Paul-Christian Bürkner. Rank-normalization, folding,
  and localization: an improved $\widehat{R}$ for assessing convergence
  of MCMC. *Bayesian Analysis* 16:667–718, 2021. doi:10.1214/20-BA1221.
  Cited in *The Inference Toolkit* and *Gradient-Based MCMC*.

### Markov chain Monte Carlo

- **Green 1995**. Peter J. Green. Reversible jump Markov chain Monte
  Carlo computation and Bayesian model determination. *Biometrika*
  82(4):711–732, 1995. Cited in *Trace Translators and Involutive MCMC*.
- **Hoffman and Gelman 2014**. Matthew D. Hoffman and Andrew Gelman. The
  No-U-Turn Sampler: adaptively setting path lengths in Hamiltonian Monte
  Carlo. *Journal of Machine Learning Research* 15(47):1593–1623, 2014. Cited
  in *Gradient-Based MCMC*.
- **Murray, Adams and MacKay 2010**. Iain Murray, Ryan Adams and David
  MacKay. Elliptical slice sampling. In *Proceedings of the Thirteenth
  International Conference on Artificial Intelligence and Statistics*
  (AISTATS 2010), pages 541–548, 2010. Cited in *Gradient-Based MCMC*.
- **Tierney 1998**. Luke Tierney. A note on Metropolis–Hastings kernels
  for general state spaces. *The Annals of Applied Probability*
  8(1):1–9, 1998. The thesis's argument that involutive MCMC leaves its
  target invariant (pp. 122–123) builds on it. Cited in *Trace
  Translators and Involutive MCMC*.

### Sequential Monte Carlo

- **Andrieu, Doucet and Holenstein 2010**. Christophe Andrieu, Arnaud
  Doucet and Roman Holenstein. Particle Markov chain Monte Carlo methods.
  *Journal of the Royal Statistical Society: Series B (Statistical
  Methodology)* 72(3):269–342, 2010. Cited in *SMC in Depth*.
- **Del Moral et al. 2006**. Pierre Del Moral, Arnaud Doucet and Ajay
  Jasra. Sequential Monte Carlo samplers. *Journal of the Royal
  Statistical Society: Series B (Statistical Methodology)*
  68(3):411–436, 2006. Cited in *Trace Translators and Involutive MCMC*.
- **Fearnhead and Clifford 2003**. Paul Fearnhead and Peter Clifford.
  On-line inference for hidden Markov models via particle filters.
  *Journal of the Royal Statistical Society: Series B (Statistical
  Methodology)* 65(4):887–899, 2003. doi:10.1111/1467-9868.00421. Cited
  in *SMC in Depth* and *SMC Recipes*.
- **Gilks and Berzuini 2001**. Walter R. Gilks and Carlo Berzuini.
  Following a moving target—Monte Carlo inference for dynamic Bayesian
  models. *Journal of the Royal Statistical Society: Series B
  (Statistical Methodology)* 63(1):127–146, 2001. Cited in *Sequential
  Monte Carlo*.
- **Lew et al. 2023**. Alexander K. Lew, George Matheos, Tan Zhi-Xuan,
  Matin Ghavamizadeh, Nishad Gothoskar, Stuart Russell and Vikash K.
  Mansinghka. SMCP3: sequential Monte Carlo with probabilistic program
  proposals. In *Proceedings of the 26th International Conference on
  Artificial Intelligence and Statistics* (AISTATS 2023), PMLR
  206:7061–7088, 2023. Cited in *SMC in Depth* and *SMC Recipes*.

### Variational inference and learning

- **ADEV**. Alexander K. Lew, Mathieu Huot, Sam Staton and Vikash K.
  Mansinghka. ADEV: sound automatic differentiation of expected values of
  probabilistic programs. *Proceedings of the ACM on Programming
  Languages* 7(POPL), article 5, 2023. doi:10.1145/3571198.
  Cited in *Variational Inference*.
- **Bornschein and Bengio 2015**. Jörg Bornschein and Yoshua Bengio.
  Reweighted wake-sleep. ICLR 2015. arXiv:1406.2751. Cited in *Learning*.
- **Burda, Grosse and Salakhutdinov 2016**. Yuri Burda, Roger Grosse and
  Ruslan Salakhutdinov. Importance weighted autoencoders. *International
  Conference on Learning Representations* (ICLR 2016). arXiv:1509.00519.
  Cited in *Variational Inference*.
- **Kucukelbir et al. 2017**. Alp Kucukelbir, Dustin Tran, Rajesh
  Ranganath, Andrew Gelman and David M. Blei. Automatic differentiation
  variational inference. *Journal of Machine Learning Research*
  18(14):1–45, 2017. Cited in *Variational Inference*.
- **Mnih and Rezende 2016**. Andriy Mnih and Danilo J. Rezende.
  Variational inference for Monte Carlo objectives. In *Proceedings of the
  33rd International Conference on Machine Learning* (ICML 2016), PMLR
  48:2188–2196, 2016. arXiv:1602.06725.
  Cited in *Variational Inference*.
- **Roeder, Wu and Duvenaud 2017**. Geoffrey Roeder, Yuhuai Wu and David
  Duvenaud. Sticking the landing: simple, lower-variance gradient
  estimators for variational inference. In *Advances in Neural Information
  Processing Systems 30* (NeurIPS 2017), 6925–6934. Cited in *Variational
  Inference*.

### Other algorithms

- **Best and Fisher 1979**. D. J. Best and N. I. Fisher. Efficient
  simulation of the von Mises distribution. *Journal of the Royal
  Statistical Society: Series C (Applied Statistics)*
  28(2):152–157, 1979. Cited in *The Distributions*.
- **LaValle 1998**. Steven M. LaValle. Rapidly-exploring random trees: a
  new tool for path planning. Technical Report TR 98-11, Computer Science
  Department, Iowa State University, 1998. Cited in *Data-Driven
  Proposals*.

## Systems and libraries

### Gen and its relatives

- **Gen.jl**. Marco Cusumano-Towner and contributors. Gen.jl, version
  0.4.8, with its documentation. <https://github.com/probcomp/Gen.jl>. Its
  paper: Marco F. Cusumano-Towner, Feras A. Saad, Alexander K. Lew and
  Vikash K. Mansinghka. Gen: a general-purpose probabilistic programming
  system with programmable inference. In *Proceedings of the 40th ACM
  SIGPLAN Conference on Programming Language Design and Implementation*
  (PLDI 2019), pages 221–236, 2019. doi:10.1145/3314221.3314642. Its
  tutorials are in the table below. Cited throughout.

The chapters cite Gen.jl's tutorials and how-to guides by these titles.
Three are notebooks of gen-quickstart, in its `tutorials` folder, rather
than pages of Gen.jl's documentation, which are in its `docs/src`
folder. Where a source titles itself otherwise, the source column gives
its own title.

| Tutorial | Source | Cited in |
|---------------------------|--------------------------------------------------|-----------------------|
| *Getting Started* | Gen.jl, `tutorials/getting_started.md` | *Getting Started* |
| *Introduction to Modeling* | Gen.jl, `tutorials/modeling_in_gen.md`: Introduction to Modeling in Gen | *Introduction to Modeling* |
| *A Bottom-up Introduction to Gen* | gen-quickstart, `A Bottom-Up Introduction to Gen.ipynb` | *Conditioning* |
| *Reasoning about Regenerate* | gen-quickstart, `Reasoning About Regenerate.ipynb` | *Update and Regenerate* |
| *Basics of Iterative Inference: MCMC and MAP* | Gen.jl, `tutorials/mcmc_map.md`: Basics of MCMC and MAP Inference | *MCMC and MAP* |
| *Debugging Models with Enumerative Inference* | Gen.jl, `tutorials/enumerative.md` | *Debugging with Enumeration* |
| *Object Tracking with Sequential Monte Carlo* | Gen.jl, `tutorials/smc.md` | *Sequential Monte Carlo* |
| *Variational Inference in Gen* | Gen.jl, `tutorials/vi.md` | *Variational Inference* |
| *Learning Generative Functions* | Gen.jl, `tutorials/learning_gen_fns.md` | *Learning* |
| *Speeding Up Inference with the Static Modeling Language* | Gen.jl, `tutorials/scaling_with_sml.md` | *Speeding Up* |
| *Improving Inference using Data-Driven Custom Proposals* | Gen.jl, `tutorials/data_driven_proposals.md` | *Data-Driven Proposals* |
| *Reversible-Jump MCMC* | gen-quickstart, `Reversible-Jump MCMC in Gen.ipynb` | *Trace Translators and Involutive MCMC*, *Reversible-Jump MCMC and Structure Learning* |
| *Adding New Distributions* | Gen.jl, `how_to/custom_distributions.md` | *Extending gen.clj* |
| *Adding New Types of Generative Functions* | Gen.jl, `how_to/custom_gen_fns.md` | *Extending gen.clj* |
| *Customizing Gradients* | Gen.jl, `how_to/custom_gradients.md` | *Extending gen.clj* |
| *Customizing Incremental Computation* | Gen.jl, `how_to/custom_incremental_computation.md` | *Extending gen.clj* |
| *Extending Gen* | Gen.jl, `how_to/extending_gen.md` | *Extending gen.clj* |

- **gen-quickstart**. Gen Quick Start: tutorial notebooks for Gen.jl,
  with the inverse-planning code of its data-driven proposals tutorial
  and the airline data. <https://github.com/probcomp/gen-quickstart>,
  commit 4d3f97e. Cited in *Data-Driven Proposals* and *Reversible-Jump
  MCMC and Structure Learning*.
- **GenDistributions**. GenDistributions.jl: Distributions.jl's families
  as Gen.jl distributions. <https://github.com/probcomp/GenDistributions.jl>.
  Cited in *Extending gen.clj* and *Coming from Gen.jl*.
- **GenJAX**. McCoy R. Becker and contributors. GenJAX: probabilistic
  programming with Gen, built on top of JAX. Version 0.10.3.
  <https://github.com/chisym/genjax>. Cited in *Trace Translators and
  Involutive MCMC*.
- **GenMLX**. Robert Johansson. GenMLX: a probabilistic programming
  language in ClojureScript, on MLX, with Gen's generative function
  interface. The system gen.clj ports.
  <https://github.com/robert-johansson/genmlx>. Cited in *Conditioning*,
  *The Handler Loop*, *Choice Maps, Traces and Selections*,
  *Composition*, *Gradient-Based MCMC*, *SMC in Depth*, *Learning*,
  *Speeding Up*, *Trace Translators and Involutive MCMC*, *Design
  Rationale*, *Verification*, *Arrays, Backends and Gradients*, *Exact
  Structure, Made Explicit*, *Language Models as Generative Functions*,
  *Constrained Generation and Token SMC*, *Extending gen.clj* and *A
  Gallery of Models*.
- **The GenMLX Book**. Robert Johansson. *The GenMLX Book: Probabilistic
  Programming with a Functional Core*. 2026. Most chapters of Parts I to
  IV port one of its chapters, and say which. Cited throughout.
- **GenParticleFilters**. Tan Zhi-Xuan. GenParticleFilters.jl: building
  blocks for particle filtering in Gen.jl. Version 0.2.3.
  <https://github.com/probcomp/GenParticleFilters.jl>. *SMC Recipes* ports
  its README's example. Cited in *Trace Translators and Involutive MCMC*
  and *SMC Recipes*.
- **GenSerialization**. GenSerialization.jl: Gen.jl's traces saved and
  read back. <https://github.com/probcomp/GenSerialization.jl>. Cited in
  *Extending gen.clj*.
- **GenSMCP3**. George Matheos. GenSMCP3.jl: automated SMCP3 for Gen.jl.
  Version 0.1.0. <https://github.com/probcomp/GenSMCP3.jl>. *SMC Recipes*
  ports its README's example and the challenging observations of its
  tutorial (`examples/notebooks`). Cited in *SMC Recipes*.
- **GenTraceKernelDSL**. GenTraceKernelDSL.jl: the language of
  GenSMCP3's kernels, `@kernel`.
  <https://github.com/probcomp/GenTraceKernelDSL.jl>. Cited in *SMC
  Recipes*.

### Inference, arrays and numerics

- **ArviZ**. Osvaldo A. Martin, Oriol Abril-Pla, Jordan Deklerk, Seth D.
  Axen, Colin Carroll, Ari Hartikainen and Aki Vehtari. ArviZ: a modular
  and flexible library for exploratory analysis of Bayesian models.
  *Journal of Open Source Software* 11(119):9889, 2026.
  doi:10.21105/joss.09889. Cited in *The Inference Toolkit*.
- **Commons Math**. Apache Commons Math, version 3.6.1.
  <http://commons.apache.org/proper/commons-math/>. Cited in *Extending
  gen.clj* and *The Distributions*.
- **Distributions.jl**. JuliaStats. Distributions.jl.
  <https://github.com/JuliaStats/Distributions.jl>. Cited in *Extending
  gen.clj*, *Coming from Gen.jl* and *The Distributions*.
- **JAX**. James Bradbury, Roy Frostig, Peter Hawkins, Matthew James
  Johnson, Chris Leary, Dougal Maclaurin, George Necula, Adam Paszke, Jake
  VanderPlas, Skye Wanderman-Milne and Qiao Zhang. JAX: composable
  transformations of Python+NumPy programs. 2018.
  <http://github.com/jax-ml/jax>. gen.clj's keys draw the same bits as
  JAX 0.9.1's.
  Cited in *Composition*, *Design Rationale*, *Arrays, Backends and
  Gradients* and *Coming from Gen.jl*.
- **MLX**. Awni Hannun, Jagrit Digani, Angelos Katharopoulos and Ronan
  Collobert. MLX: efficient and flexible machine learning on Apple
  silicon. 2023. <https://github.com/ml-explore>. gen.clj's MLX backend
  pins MLX 0.32.2, through mlx-c 0.7.0. Cited in *Getting Started*, *The
  Handler Loop*, *The Inference Toolkit*, *Gradient-Based MCMC*,
  *Speeding Up*, *The Generative Function Interface*, *Design Rationale*,
  *Arrays, Backends and Gradients*, *Exact Structure, Made Explicit*,
  *Language Models as Generative Functions*, *Data-Driven Proposals*,
  *Reversible-Jump MCMC and Structure Learning*, *Coming from Gen.jl* and
  *Installing and Running*.
- **NumPy**. Travis E. Oliphant et al. NumPy. <https://numpy.org>. Cited in
  *Design Rationale* and *Arrays, Backends and Gradients*.
- **PyTorch**. Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James
  Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia
  Gimelshein, Luca Antiga, et al. PyTorch: an imperative style,
  high-performance deep learning library. In *Advances in Neural
  Information Processing Systems 32*, pages 8026–8037, 2019. Cited in
  *Arrays, Backends and Gradients*.
- **Random123**. The counter-based random number generators of Salmon et
  al. (2011), Threefry among them.
  <http://www.thesalmons.org/john/random123/papers/random123sc11.pdf>.
  Cited in *Arrays, Backends and Gradients*.
- **SciPy**. SciPy. <https://scipy.org>. Cited in *The Distributions*.
- **Stan**. Bob Carpenter, Andrew Gelman, Matthew Hoffman, Daniel Lee,
  Ben Goodrich, Michael Betancourt, Marcus Brubaker, Jiqiang Guo, Peter Li
  and Allen Riddell. Stan: a probabilistic programming language. *Journal
  of Statistical Software* 76(1):1–32, 2017. Cited in *The Inference
  Toolkit*, *Gradient-Based MCMC* and *A Gallery of Models*.

### Language models

- **GenLM**. The GenLM project's Python libraries: genlm-backend, whose
  `AsyncLM` the LM protocol of `gen.llm.lm` follows,
  <https://github.com/genlm/genlm-backend>, and genlm-control,
  <https://github.com/genlm/genlm-control>. Cited in *Language Models as
  Generative Functions* and *Constrained Generation and Token SMC*.
- **llama.cpp**. llama.cpp, release v0.7.0, which `gen.llm.llama` runs.
  <https://github.com/ggml-org/llama.cpp>. Cited in *The Handler Loop*,
  *Design Rationale*, *Language Models as Generative Functions* and
  *Installing and Running*.

## Data

- **Box and Jenkins 1976**. George E. P. Box and Gwilym M. Jenkins. *Time
  Series Analysis: Forecasting and Control*, revised edition. Holden-Day,
  San Francisco, 1976. The AirPassengers series: monthly totals of
  international airline passengers, in thousands, January 1949 to
  December 1960. *Reversible-Jump MCMC and
  Structure Learning* reads it from gen-quickstart's
  `tutorials/airline.csv`, which holds the series less its mean. Cited in
  *Reversible-Jump MCMC and Structure Learning*.

## The bibliography, checked

The entries and the chapters are checked against each other, so that a
citation added to a chapter, or an entry added here, fails the book's
test until the two agree. The code reads the book's Markdown, and needs
nothing else:

<!-- code: ns -->
```clojure
(ns book.bibliography
  (:require [clojure.java.io :as io]
            [clojure.string :as str]))
```

The pages that can cite are every numbered page but a Part's heading,
*The Public API*, whose text is the docstrings' and cites as the code
does, and this appendix. The checks read a page's prose, without its
code, and its title, by which the entries name it:

<!-- code: pages -->
```clojure
(defn prose
  "A page's prose, as the checks read it: its Markdown without code blocks,
   inline code and comments, its lines joined, so that a citation broken
   across two lines reads whole."
  [md]
  (-> md
      (str/replace #"(?ms)^```.*?^```$" " ")
      (str/replace #"`[^`]*`" " ")
      (str/replace #"(?s)<!--.*?-->" " ")
      (str/replace #"\s+" " ")))

(defn title
  "A page's title, less a subtitle after a colon: Getting Started, for
   Getting Started: Linear Regression."
  [md]
  (first (str/split (second (re-find #"(?m)^# (.+)$" md)) #": ")))

(def pages
  "The pages that cite, in the book's order, as {:title :text}: every
   numbered Markdown file but a Part's heading, The Public API (the
   docstrings', which cite as the code does) and this appendix."
  (->> (.listFiles (io/file "book"))
       (map #(.getName ^java.io.File %))
       (filter #(re-matches #"\d{3}-.*\.md" %))
       (remove #(re-matches #"\d\d0-.*|063-.*|066-.*" %))
       sort
       (mapv (fn [file]
               (let [md (slurp (str "book/" file))]
                 {:title (title md) :text (prose md)})))))
```

An author-year citation is found by a pattern: the authors, then the year
in one of the forms the chapters use. One surname before a bare year
counts only inside parentheses, since a capitalized word and a number
also begin sentences:

<!-- code: pattern -->
```clojure
(def surname
  "A surname: a capital, then letters and hyphens, ending in a small letter
   (so not MCMC), perhaps after a particle (Del Moral)."
  "(?:(?:Del|De|van|von) )?\\p{Lu}[\\p{L}-]*\\p{Ll}")

(def citation-re
  "An author-year citation. An opening parenthesis or semicolon, if any;
   the authors: a surname, two joined by and, more with commas, or one with
   et al.; then the year: bare, after a comma, in parentheses, or after a
   possessive and a title, as in Cusumano-Towner's thesis (2020) and Saad
   et al.'s *Title* (POPL 2019)."
  (re-pattern
   (str "(\\(|; )?(?<![\\p{L}-])"
        "(" surname "(?:(?:, " surname ")* and " surname "| et al\\.)?)"
        "('s (?:thesis |\\*[^*]+\\* )?\\((?:\\p{Lu}+ )?|'s |,? \\(?)"
        "((?:19|20)\\d\\d)(?!\\d)")))

(defn citations
  "The author-year citations of a text, as keys: authors, a space, the
   year. One surname before a bare year is a citation only inside
   parentheses, as in (Green 1995), so that a sentence that begins In 2000
   tries is not one."
  [text]
  (vec (for [[_ opener authors link year] (re-seq citation-re text)
             :when (or opener
                       (re-find #" and | et al\." authors)
                       (re-find #"'s|\(" link))]
         (str authors " " year))))
```

On a sample, the pattern finds three citations, and passes over a
sentence that begins with a number and a possessive without a year:

<!-- code: sample -->
```clojure
(citations (str "the proposal's efficiency (Kong, 1992), Gilks and "
                "Berzuini (2001) and Saad et al.'s *Bayesian Synthesis* "
                "(POPL 2019). In 2000 tries, Cusumano-Towner's thesis"))
;=> ["Kong 1992" "Gilks and Berzuini 2001" "Saad et al. 2019"]
```

The bibliography's own Markdown gives its entries. An entry is a list
item that begins with its key in bold and ends with the pages that cite
it; the table of Gen.jl's tutorials gives each tutorial's citing pages
in its last column:

<!-- code: entries -->
```clojure
(defn items
  "The list items of Markdown md that begin with a key in bold, each on
   one line."
  [md]
  (map #(str/replace % #"\s+" " ")
       (re-seq #"(?ms)^- \*\*.*?(?=\n- |\n\n|\z)" md)))

(defn entry
  "An entry of the bibliography from its list item: {:key :alias :cited}.
   The item begins with the key in bold, and perhaps an alias in italics
   and parentheses, (*the thesis*); it ends with Cited in and the citing
   pages' titles in italics, or with Cited throughout."
  [item]
  (let [[_ k alias] (re-find #"^- \*\*([^*]+)\*\*(?: \(\*([^*]+)\*\))?\. "
                             item)]
    {:key k
     :alias alias
     :cited (if (str/ends-with? item "Cited throughout.")
              :throughout
              (set (map second (re-seq #"\*([^*]+)\*"
                                       (second (re-find #"Cited in (.+)\.$"
                                                        item))))))}))

(defn tutorials
  "The rows of the table of tutorials in md, [title cited]: the rows whose
   first cell is a title in italics, cited being the titles in their last
   cell."
  [md]
  (for [line (str/split-lines md)
        :let [[_ t cited] (re-matches #"\| \*([^*]+)\* \|.*\| (.+) \|" line)]
        :when t]
    [t (set (map second (re-seq #"\*([^*]+)\*" cited)))]))
```

A chapter can also cite a work without a year: *Trace Translators and
Involutive MCMC* follows the thesis's argument "after Tierney", and *The
Distributions* draws von Mises values "by Best and Fisher's rejection".
So a page cites an entry where it writes one of the entry's names, too:
an author-year entry's authors, as written, a system's name in either
case (the GenMLX book, genlm), and an alias (the thesis). Longer names
are found first, so that a name inside a longer one does not count
twice:

<!-- code: names -->
```clojure
(defn year-key?
  "Whether an entry's key ends in a year: Green 1995, not Stan."
  [k]
  (boolean (re-find #" \d{4}$" k)))

(defn names
  "The names that cite each entry, as [name key exact?]: an author-year
   entry's authors, as written (exact), and any other key, or an alias, in
   either case (the GenMLX book, genlm). An alias drops its article: the
   chapters write the thesis, the thesis's and thesis §4.1."
  [entries]
  (for [{k :key alias :alias} entries
        [n exact?] (cond-> [(if (year-key? k)
                              [(str/replace k #" \d{4}$" "") true]
                              [k false])]
                     alias (conj [(str/replace alias #"^the " "") false]))]
    [n k exact?]))

(defn cited-by-name
  "The keys of the entries whose names appear in a text. Longer names are
   found first, then blanked out, so that Cusumano-Towner and Mansinghka
   does not also count as the thesis's Cusumano-Towner, nor the GenMLX
   book as GenMLX."
  [text names]
  (loop [text text
         [[n k exact?] & more] (sort-by (comp count first) > names)
         found #{}]
    (if-not n
      found
      (let [re (re-pattern (str (when-not exact? "(?i)")
                                "(?<![\\p{L}\\p{N}.-])"
                                (java.util.regex.Pattern/quote n)
                                "(?![\\p{L}\\p{N}-])"))]
        (recur (str/replace text re #(apply str (repeat (count %) "~")))
               more
               (cond-> found (re-find re text) (conj k)))))))
```

A citation names an entry by its key, or, written with et al., by its
first author and year:

<!-- code: resolve -->
```clojure
(defn first-author
  "A key's first author: Andrieu, of Andrieu, Doucet and Holenstein 2010."
  [k]
  (re-find #"^.+?(?=,| and | et al\.| \d{4}$)" k))

(defn entry-of
  "The key of the entry a citation names: its own key, or, for X et al. Y,
   the one entry whose first author is X and whose year is Y, as Andrieu
   et al. 2010 names Andrieu, Doucet and Holenstein 2010."
  [ks citation]
  (or (some #{citation} ks)
      (when (str/includes? citation " et al. ")
        (let [year #(re-find #"\d{4}$" %)
              same (filter #(and (= (first-author %) (first-author citation))
                                 (= (year %) (year citation)))
                           ks)]
          (when (= 1 (count same)) (first same))))))
```

Then the checks. The three lists are the citations without an entry, the
entries that no page cites, and the entries whose list of citing pages is
wrong; all three are empty:

<!-- code: checks -->
```clojure
(def bibliography (slurp "book/066-bibliography.md"))

(def entries (map entry (items bibliography)))

(def ks (map :key entries))

(def cited-in
  "Each entry's citing pages, by title: those with a citation that names
   it, and those that write one of its names."
  (let [all-names (names entries)]
    (apply merge-with into (zipmap ks (repeat #{}))
           (for [{:keys [title text]} pages
                 k (concat (keep #(entry-of ks %) (citations text))
                           (cited-by-name text all-names))]
             {k #{title}}))))

(defn listed-right?
  "Whether an entry lists the pages that cite it: all of them, or Cited
   throughout when more than half the pages do."
  [{k :key cited :cited}]
  (if (= cited :throughout)
    (> (count (cited-in k)) (quot (count pages) 2))
    (= cited (cited-in k))))

[(for [{:keys [title text]} pages
       c (citations text)
       :when (not (entry-of ks c))]
   [title c])
 (remove (comp seq cited-in) ks)
 (map :key (remove listed-right? entries))]
;=> [[] [] []]
```

And each title in the table of tutorials is written, in italics, by every
page its last column names:

<!-- code: tutorials-check -->
```clojure
(def text-of (into {} (map (juxt :title :text) pages)))

(for [[t cited] (tutorials bibliography)
      page cited
      :when (not (str/includes? (text-of page "") (str "*" t "*")))]
  [t page])
;=> []
```
