(ns gen.llm.agent-test
  "The agent's episode loop (gen.llm.agent) with scripted replies (a
   :complete that returns them in turn, recording the contexts it was
   given) over a toy engine's LM: a tool's result read back, its other
   keys logged; submit (:done) ending the episode; a reply without a call
   answered; the turn budget's :on-budget and the countdown notes; a tool
   that throws answered with its error; K candidates and :choose; with
   thinking, :reasoning logged and a reply cut inside its think block a
   loud error, the session closed either way."
  (:require [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [gen.llm.agent :as agent]
            [gen.llm.byte-tokenizer :as bt]
            [gen.llm.lm :as lm]
            [gen.llm.toy-engine :as toy]
            [gen.random :as rng]))

(def ^:private tok (bt/tokenizer (mapv #(byte-array [(unchecked-byte %)]) (range 256))))

(defn- with-lm [f]
  (with-open [e (toy/engine {:n-vocab (:n-vocab tok) :max-branches 8 :n-ctx 4096})
              m (lm/lm e tok {:max-branches 8})]
    (f m)))

(defn- call [tool & kvs]
  (str "<tool_call>\n<function=" tool ">\n"
       (str/join (map (fn [[k v]] (str "<parameter=" k ">\n" v "\n</parameter>\n")) (partition 2 kvs)))
       "</function>\n</tool_call>"))

(defn- scripted
  "A :complete giving replies in turn (each a string, or a vector of
   candidates' strings), the contexts it was given in seen (decoded)."
  [m replies seen]
  (let [left (atom replies)]
    (fn [_ contexts _]
      (swap! seen conj (lm/decode m (first contexts)))
      (let [r (first @left)]
        (swap! left rest)
        (mapv (fn [text] {:tokens (lm/encode m text {:add-special? false}) :stop :end :end-token (:eos tok) :logprob 0.0})
              (if (vector? r) r (repeat (count contexts) r)))))))

(def ^:private tools
  {"add" {:description "Add two numbers." :params [{:name "a" :description "A number."} {:name "b" :description "A number."}]
          :fn (fn [{:strs [a b]} state] (swap! state conj [:add a b])
                {:text (str (+ (parse-long a) (parse-long b))) :data :ok})}
   "boom" {:description "Throws." :params [] :fn (fn [_ _] (throw (ex-info "it broke" {})))}
   "submit" {:description "Submit an answer." :params [{:name "answer" :description "The answer."}]
             :fn (fn [{:strs [answer]} _] {:text "submitted" :done (parse-long answer)})}})

(deftest an-episode
  (with-lm
    (fn [m]
      (let [seen (atom []) closed (atom nil)
            r (agent/run m {:system "You add." :task "What is 1 + 2?" :tools tools
                            :open #(atom []) :close #(reset! closed @%)}
                         {:key (rng/key 1) :complete (scripted m [(str "Let me add.\n" (call "add" "a" "1" "b" "2"))
                                                                  (call "submit" "answer" "3")]
                                                              seen)})]
        (is (= {:result 3 :stop :done :turns 2} (select-keys r [:result :stop :turns])))
        (is (= [[:add "1" "2"]] @closed) "the session closed, holding what the tools did")
        (testing "a tool's other keys in the log"
          (is (= {:tool "add" :params {"a" "1" "b" "2"} :text "3" :data :ok} (first (:calls (first (:log r)))))))
        (testing "the second turn reads the first's reply and the tool's response"
          (is (str/includes? (second @seen) "<tool_response>\n3\n</tool_response>"))
          (is (str/includes? (second @seen) "Let me add.")))
        (testing "the tools are offered"
          (is (str/includes? (first @seen) "\"name\": \"add\"")))))))

(deftest no-call-a-tool-error-and-the-budget
  (with-lm
    (fn [m]
      (let [seen (atom [])
            r (agent/run m {:task "Go." :tools tools :open (fn [] (atom [])) :on-budget (fn [_] :fallback)
                            :notes (fn [_ left] (when (<= left 1) (str left " turn left")))}
                         {:turns 3 :key (rng/key 2)
                          :complete (scripted m ["I am thinking." (call "boom") (call "add" "a" "2" "b" "2")] seen)})]
        (is (= {:result :fallback :stop :turns :turns 3} (select-keys r [:result :stop :turns])))
        (is (str/includes? (nth @seen 1) agent/no-call-text))
        (is (str/includes? (nth @seen 2) "error: it broke"))
        (is (str/includes? (nth @seen 2) "1 turn left") "the countdown note")))))

(deftest candidates-and-a-chooser
  (with-lm
    (fn [m]
      (let [r (agent/run m {:task "Go." :tools tools}
                         {:candidates 2 :choose (fn [cands _] (if (str/includes? (:text (first cands)) "42") 0 1))
                          :key (rng/key 3)
                          :complete (scripted m [[(call "submit" "answer" "7") (call "submit" "answer" "9")]] (atom []))})]
        (is (= 9 (:result r)) "the chooser picked the second")))))

(deftest thinking
  (with-lm
    (fn [m]
      (testing ":reasoning logged"
        (let [r (agent/run m {:task "Go." :tools tools}
                           {:thinking? true :key (rng/key 4)
                            :complete (scripted m [(str "I should submit 5.\n</think>\n\n" (call "submit" "answer" "5"))] (atom []))})]
          (is (= 5 (:result r)))
          (is (= "I should submit 5." (:reasoning (first (:log r)))))))
      (testing "a reply cut inside its think block is a loud error, the session closed"
        (let [closed (atom false)]
          (is (thrown-with-msg? clojure.lang.ExceptionInfo #"cut inside its think block"
                                (agent/run m {:task "Go." :tools tools :open (fn [] :s) :close (fn [_] (reset! closed true))}
                                           {:thinking? true :key (rng/key 5)
                                            :complete (scripted m ["still thinking and thinking"] (atom []))})))
          (is @closed))))))
