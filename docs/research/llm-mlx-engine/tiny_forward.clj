;; A prototype for docs/research/llm-mlx-engine.md: Qwen3's and Qwen3.5's
;; layers written over gen.array's current operations, as composites, on a
;; tiny model with random weights. It checks, on the pure-JVM backend
;; (float64), what the engine will rely on:
;;   1. a cached prefill and steps give the full forward's logits;
;;   2. a batch of contexts gives each context's logits alone;
;;   3. the tape's gradients through the forward match finite differences;
;;   4. a cache is a value: continuing one copy leaves the other as it was.
;; Run: clojure -M docs/research/llm-mlx-engine/tiny_forward.clj
(ns tiny-forward
  (:refer-clojure :exclude [+ - * / < > <= >= ==])
  (:require [gen.array :as array :refer [+ - * / < > <= >= ==]]
            [gen.random :as random]))

;; ---------------------------------------------------------------------------
;; The layers' operations, as composites
;; ---------------------------------------------------------------------------

(defn rms-norm [x w eps]
  (* x w (/ 1.0 (array/sqrt (+ (array/mean (array/square x) {:axis -1 :keep-dims? true}) eps)))))

(defn l2-norm [x eps]
  (/ x (array/sqrt (+ (array/sum (array/square x) {:axis -1 :keep-dims? true}) eps))))

(defn silu [x] (* x (array/sigmoid x)))

(defn linear "y = x Wᵀ, W [out in] as Hugging Face's and MLX's GGUF loader's." [x w]
  (array/matmul x (array/transpose w)))

(defn rope
  "NEOX-style rotary embedding (halves, as llama.cpp's qwen3 and qwen35)
   of the first dims channels of x [B H T D] at positions pos [B T]."
  [x pos dims theta]
  (let [[_ _ _ d] (array/shape x)
        half (quot dims 2)
        inv (array/exp (* (array/arange half) (/ (- (Math/log theta)) half)))
        ang (array/expand-dims (* (array/expand-dims pos [-1]) inv) [1])   ; [B 1 T half]
        c (array/cos ang) s (array/sin ang)
        x1 (array/slice x -1 0 half) x2 (array/slice x -1 half dims)
        rot [(- (* x1 c) (* x2 s)) (+ (* x2 c) (* x1 s))]]
    (array/concatenate (if (< dims d) (conj rot (array/slice x -1 dims d)) rot) -1)))

(defn attention
  "Softmax attention of q [B H T D] over k, v [B KV S D] (each KV head
   serving H/KV query heads in turn), visible [T S] or [B 1 T S]."
  [q k v visible]
  (let [[b h t d] (array/shape q)
        [_ kv s _] (array/shape k)
        rep (quot h kv)
        heads (fn [x] (array/reshape (array/broadcast-to (array/expand-dims x [2]) [b kv rep s d]) [b h s d]))
        scores (* (array/matmul q (array/transpose (heads k) [0 1 3 2])) (/ 1.0 (Math/sqrt d)))]
    (array/matmul (array/softmax (array/where visible scores (array/full-like scores Double/NEGATIVE_INFINITY)))
                  (heads v))))

;; ---------------------------------------------------------------------------
;; A tiny Qwen3 (attention) and Qwen3.5 (every other layer GatedDeltaNet)
;; ---------------------------------------------------------------------------

(def cfg {:vocab 97 :hidden 32 :heads 4 :kv-heads 2 :head-dim 8 :ffn 48 :layers 4 :eps 1e-6 :theta 10000.0
          ;; GatedDeltaNet: heads, key and value dims, conv kernel
          :lin-heads 2 :dk 8 :dv 8 :conv 4})

(defn init
  "Random weights: matrices N(0, 0.2²), norms 1 + N(0, 0.1²); layer i is
   GatedDeltaNet when (kind i) is :gdn."
  [key {:keys [vocab hidden heads kv-heads head-dim ffn layers lin-heads dk dv conv]} kind]
  (let [ks (atom (random/split key 400))
        draw (fn [shape sd] (let [k (first @ks)] (swap! ks rest) (* sd (random/normal k shape))))
        norm (fn [n] (+ 1.0 (draw [n] 0.1)))
        conv-ch (+ (* 2 lin-heads dk) (* lin-heads dv))]
    {:embed (draw [vocab hidden] 0.2)
     :norm (norm hidden)
     :layers (vec (for [i (range layers)]
                    (merge {:kind (kind i) :attn-norm (norm hidden) :ffn-norm (norm hidden)
                            :gate (draw [ffn hidden] 0.2) :up (draw [ffn hidden] 0.2) :down (draw [hidden ffn] 0.2)}
                           (if (= :gdn (kind i))
                             {:qkv (draw [conv-ch hidden] 0.2) :z (draw [(* lin-heads dv) hidden] 0.2)
                              :b (draw [lin-heads hidden] 0.2) :a (draw [lin-heads hidden] 0.2)
                              :conv-w (draw [conv-ch conv] 0.3)
                              :A (- (array/exp (draw [lin-heads] 0.3)))          ; as the GGUF's ssm_a: -exp(A_log)
                              :dt-bias (draw [lin-heads] 0.3) :ssm-norm (norm dv)
                              :out (draw [hidden (* lin-heads dv)] 0.2)}
                             {:q (draw [(* heads head-dim) hidden] 0.2) :k (draw [(* kv-heads head-dim) hidden] 0.2)
                              :v (draw [(* kv-heads head-dim) hidden] 0.2) :o (draw [hidden (* heads head-dim)] 0.2)
                              :q-norm (norm head-dim) :k-norm (norm head-dim)}))))}))

(defn- attn-layer [{:keys [heads kv-heads head-dim eps theta]} w hn pos entry]
  (let [[b t _] (array/shape hn)
        proj (fn [m n norm] (-> (linear hn (m w)) (array/reshape [b t n head-dim]) (array/transpose [0 2 1 3])
                                (cond-> norm (rms-norm (norm w) eps))))
        q (rope (proj :q heads :q-norm) pos head-dim theta)
        k (rope (proj :k kv-heads :k-norm) pos head-dim theta)
        v (proj :v kv-heads nil)
        k (if entry (array/concatenate [(:k entry) k] 2) k)
        v (if entry (array/concatenate [(:v entry) v] 2) v)
        s (second (rest (array/shape k)))
        ;; key j (all cached, then the chunk's) is visible from query i when j <= s - t + i
        visible (<= (array/arange s) (+ (array/expand-dims (array/arange t) [-1]) (- s t)))
        o (-> (attention q k v visible) (array/transpose [0 2 1 3]) (array/reshape [b t (* heads head-dim)]))]
    [(linear o (:o w)) {:k k :v v}]))

(defn- gdn-layer
  "GatedDeltaNet as llama.cpp's qwen35 graph computes it from a GGUF's
   tensors: a causal depthwise conv over the q/k/v projection, l2-normed
   q and k, beta = sigmoid, decay g = exp(A softplus(a + dt-bias)), the
   gated delta rule token by token, and a gated RMSNorm."
  [{:keys [lin-heads dk dv conv eps]} w hn _ entry]
  (let [[b t _] (array/shape hn)
        hv lin-heads
        key-dim (* hv dk)
        qkv (linear hn (:qkv w))                                          ; [B T C]
        ch (last (array/shape qkv))
        padded (array/concatenate [(if entry (:conv entry) (array/zeros [b (dec conv) ch])) qkv] 1)
        conv-out (silu (reduce + (for [j (range conv)]
                                   (* (array/slice padded 1 j (+ j t))
                                      (array/reshape (array/slice (:conv-w w) 1 j (inc j)) [ch])))))
        heads (fn [from to d] (array/reshape (array/slice conv-out 2 from to) [b t hv d]))
        q (* (l2-norm (heads 0 key-dim dk) 1e-12) (/ 1.0 (Math/sqrt dk)))
        k (l2-norm (heads key-dim (* 2 key-dim) dk) 1e-12)
        v (heads (* 2 key-dim) ch dv)
        beta (array/sigmoid (linear hn (:b w)))                           ; [B T Hv]
        g (array/exp (* (:A w) (array/softplus (+ (linear hn (:a w)) (:dt-bias w)))))
        [state ys] (reduce (fn [[st ys] i]
                             (let [at (fn [x] (array/squeeze (array/slice x 1 i (inc i)) [1]))
                                   st (* st (array/expand-dims (at g) [-1 -2]))         ; [B Hv Dv Dk]
                                   kt (array/expand-dims (at k) [2])                    ; [B Hv 1 Dk]
                                   delta (* (- (at v) (array/sum (* st kt) {:axis -1})) (array/expand-dims (at beta) [-1]))
                                   st (+ st (* (array/expand-dims delta [-1]) kt))
                                   y (array/sum (* st (array/expand-dims (at q) [2])) {:axis -1})]
                               [st (conj ys y)]))
                           [(if entry (:state entry) (array/zeros [b hv dv dk])) []] (range t))
        y (array/stack ys 1)                                              ; [B T Hv Dv]
        z (array/reshape (linear hn (:z w)) [b t hv dv])
        y (array/reshape (* (rms-norm y (:ssm-norm w) eps) (silu z)) [b t (* hv dv)])]
    [(linear y (:out w)) {:conv (array/slice padded 1 t (+ t (dec conv))) :state state}]))

(defn forward
  "Logits [B T vocab] of tokens [B T] at positions pos [B T] after cache
   (one entry a layer, nil for none), and the cache after them."
  [{:keys [eps] :as c} {:keys [embed norm layers]} cache tokens pos]
  (let [[h cache'] (reduce (fn [[h nc] [w entry]]
                             (let [layer (if (= :gdn (:kind w)) gdn-layer attn-layer)
                                   [a e] (layer c w (rms-norm h (:attn-norm w) eps) pos entry)
                                   h (+ h a)
                                   m (rms-norm h (:ffn-norm w) eps)]
                               [(+ h (linear (* (silu (linear m (:gate w))) (linear m (:up w))) (:down w))) (conj nc e)]))
                           [(array/take embed tokens 0) []]
                           (map vector layers (or cache (repeat nil))))]
    [(linear (rms-norm h norm eps) embed) cache']))               ; tied output, as Qwen3-0.6B and Qwen3.5-0.8B

;; ---------------------------------------------------------------------------
;; The checks
;; ---------------------------------------------------------------------------

(defn- tokens [xs] (array/array [xs] {:dtype :int64}))
(defn- positions [from n] (array/array [(vec (range from (+ from n)))]))
(defn- max-diff [a b] (array/item (array/max (array/abs (- a b)))))

(defn check [label kind]
  (let [model (init (random/key 7) cfg kind)
        ctx [5 17 42 3 88 11 60 2 9]
        n (count ctx)
        [full _] (forward cfg model nil (tokens ctx) (positions 0 n))
        [_ c4] (forward cfg model nil (tokens (subvec ctx 0 4)) (positions 0 4))
        [chunk c7] (forward cfg model c4 (tokens (subvec ctx 4 7)) (positions 4 3))
        [s7 c8] (forward cfg model c7 (tokens [(ctx 7)]) (positions 7 1))
        [s8 _] (forward cfg model c8 (tokens [(ctx 8)]) (positions 8 1))
        other [5 17 42 3 88 11 60 2 70]
        ;; 4: the cache c8 continued with another token, then c8 again
        [o8 _] (forward cfg model c8 (tokens [(other 8)]) (positions 8 1))
        [s8' _] (forward cfg model c8 (tokens [(ctx 8)]) (positions 8 1))
        [both _] (forward cfg model nil (array/array [ctx other] {:dtype :int64})
                          (array/array [(vec (range n)) (vec (range n))]))
        [alone _] (forward cfg model nil (tokens other) (positions 0 n))
        row (fn [x i] (array/slice x 1 i (inc i)))
        ;; 3: d/dw of log p(ctx[8] | ctx[0..7]) for a weight of layer 0 and of the embedding
        target (ctx 8)
        loss (fn [m] (let [[lg _] (forward cfg m nil (tokens (subvec ctx 0 8)) (positions 0 8))]
                       (array/sum (array/pick (array/log-softmax (row lg 7)) (array/array [[target]] {:dtype :int64})))))
        wkey (if (= :gdn (kind 0)) :qkv :q)
        grads (array/grad (fn [w0 e] (loss (-> model (assoc-in [:layers 0 wkey] w0) (assoc :embed e))))
                          {:argnums [0 1]})
        [gw ge] (grads (get-in model [:layers 0 wkey]) (:embed model))
        fd (fn [path [i j]]
             (let [w (get-in model path)
                   bump (fn [d] (loss (assoc-in model path (+ w (* d (array/where (array/logical-and
                                                                                  (== (array/expand-dims (array/arange (first (array/shape w))) [-1]) i)
                                                                                  (== (array/arange (second (array/shape w))) j))
                                                                                 1.0 0.0))))))]
               (/ (- (array/item (bump 1e-5)) (array/item (bump -1e-5))) 2e-5)))
        rel (fn [g path ij] (let [a (array/item (array/slice (array/slice g 0 (first ij) (inc (first ij))) 1 (second ij) (inc (second ij))))
                                  b (fd path ij)]
                              (/ (Math/abs (- a b)) (max 1e-12 (Math/abs b)))))]
    (println (format "%s:" label))
    (println (format "  1. cached prefill + chunk + steps vs the full forward: |Δ logit| ≤ %.1e"
                     (max (max-diff (row full 6) (row chunk 2)) (max-diff (row full 7) s7) (max-diff (row full 8) s8))))
    (println (format "  2. a batch of two contexts vs the second alone: |Δ logit| ≤ %.1e"
                     (max-diff (array/slice both 0 1 2) alone)))
    (println (format "  3. gradients vs central differences, relative error ≤ %.1e"
                     (reduce max (concat (for [ij [[0 0] [3 5] [7 31]]] (rel gw [:layers 0 wkey] ij))
                                         (for [ij [[(ctx 2) 4] [target 9]]] (rel ge [:embed] ij))))))
    (println (format "  4. a cache continued twice from one value: the same logits (|Δ| = %.1e), another token's differ (%.2f)"
                     (max-diff s8 s8') (max-diff s8 o8)))))

(check "Qwen3-like, 4 attention layers" (constantly :attn))
(check "Qwen3.5-like, GatedDeltaNet with every fourth layer attention" #(if (= 3 (mod % 4)) :attn :gdn))
(shutdown-agents)
