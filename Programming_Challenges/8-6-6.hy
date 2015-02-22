(import itertools)
(import [pyrsistent [v]])


;; This is supposed to get fixed in 0.11
(defmacro cut [&rest that-which-is-cut] `(slice ~@that-which-is-cut))

(defn tegmark [size]
  (apply itertools.product [[0 1]] {"repeat" size}))

(defn jurisprudence [law]
  (let [[sentencing-guidelines (->> (-> (bin law)
                                        (cut 2) (.zfill 8))
                                    (map int) (list) (apply v) (reversed))]]
    (dict-comp jurisdiction verdict
               [[jurisdiction verdict] (zip (tegmark 3)
                                            sentencing-guidelines)])))

(defn jurisdiction [state precinct]
  (tuple (map (fn [i] (get state (% i (len state))))
              [(dec precinct) precinct (inc precinct)])))

(defn advance [state law]
  (let [[common-law (jurisprudence law)]]
    (list (map (fn [precinct] (get common-law (jurisdiction state precinct)))
               (range (len state))))))

(defn garden-of-eden? [law state]
  (not (any (filter (fn [configuration] (= state configuration))
                    (genexpr (advance law configuration)
                             [configuration (tegmark size)])))))

(defmacro deftest [name &rest code]  ; boilerplate sugar
  `(defn ~name [] ~@code))

(deftest test-tegmark
  (assert (= (list (tegmark 2))
             (list (map tuple [[0 0] [0 1] [1 0] [1 1]])))))

(deftest test-jurisprudence
  (assert (= (jurisprudence 90)
             {(, 0 0 0) 0
              (, 0 0 1) 1
              (, 0 1 0) 0
              (, 0 1 1) 1
              (, 1 0 0) 1
              (, 1 0 1) 0
              (, 1 1 0) 1
              (, 1 1 1) 0})))

(deftest test-advance
  (assert (= (advance [1 1 0 1] 204)
             [1 1 0 1]))
  (assert (= (advance [1 1 0 1] 0)
             [0 0 0 0]))
  (assert (= (advance [1 1 0 1] 255)
             [1 1 1 1])))

(test-tegmark)
(test-jurisprudence)
(test-advance)
