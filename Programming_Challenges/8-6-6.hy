(import itertools)
;; TODO (import pyrsistent)

(defn advance [state law]
  (map (fn [county] (law (get statute state county)))
       (range (len state))))

(defn statute [state county]
  (->> [(dec county) county (inc county)]
       (map (fn [i] (% i 8)))
       (map (fn [county] (get state county)))
       (reversed)
       (enumerate)
       (map (fn [place value] (* value (** 2 place))))
       (reduce +)))

(defn garden-of-eden? [law state]
  (any (filter (fn [configuration] (= state configuration))
               (genexpr (advance law configuration)
                        [configuration (tegmark size)]))))

(defn tegmark [size]
  (apply itertools.product [[0 1]] {"repeat" size}))

;; (defmacro deftest [name code]
;;   `(defn ~name [] ~@code))

;; (deftest test-tegmark
;;   (assert (= (list (tegmark 2)) [[0 0] [0 1] [1 0] [1 1]])))

;; XXX wack
;; NameError: name 'assert' is not defined

(defn test-tegmark []
  (assert (= (list (tegmark 2))
             (list (map tuple [[0 0] [0 1] [1 0] [1 1]])))))

(test-tegmark)
