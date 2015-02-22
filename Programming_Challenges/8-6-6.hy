(import itertools)
(import [pyrsistent [m v]])


(defn advance [state law]
  ;; maybe something like
  (let [[common-law (jurisprudence law)]]
    (apply v
           (map (fn [precinct] (get common-law (jurisdiction state precinct)))
                (range (len state))))))

;; This is supposed to get fixed in 0.11
(defmacro cut [&rest that-which-is-cut] `(slice ~@that-which-is-cut))

(defn jurisprudence [law]
  (let [[sentencing-guidelines (->> (-> (bin law)
                                        (cut 2) (.zfill 8))
                                    (map int) (apply v) (reversed))]]
    (dict-comp jurisdiction verdict
               [[jurisdiction verdict] (zip (tegmark 3)
                                            sentencing-guidelines)])))

(defn statute [state county]
  (->> [(dec county) county (inc county)]  ; indices
       (map (fn [i] (% i 8)))  ; mod eight
       (map (fn [county] (get state county)))  ; look up in state
       ;;
       ;; XXX EARLY DEMENTIA: can't quite make sense of what I was
       ;; thinking here; the beginning of the function looks like
       ;; we're going to to compute the next state for a cell, but the
       ;; end looks like we're computing the number of a law
       (reversed)
       (enumerate)
       (map (fn [place value] (* value (** 2 place))))
       (reduce +)))

(defn garden-of-eden? [law state]
  (not (any (filter (fn [configuration] (= state configuration))
                    (genexpr (advance law configuration)
                             [configuration (tegmark size)])))))


(defn tegmark [size]
  (apply itertools.product [[0 1]] {"repeat" size}))


(defmacro deftest [&rest code] `(defn ~@code))  ; boilerplate sugar

(deftest test-tegmark []
  (assert (= (list (tegmark 2))
             (list (map tuple [[0 0] [0 1] [1 0] [1 1]])))))

(test-tegmark)
