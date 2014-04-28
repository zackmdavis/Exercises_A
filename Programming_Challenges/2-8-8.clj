(ns yahtzee
  (:use clojure.test))

(defn ks [k]
  (fn [dice] (reduce + (filter #{k} dice))))

(def names {1 'one 2 'two 3 'three 4 'four 5 'five 6 'six})

(doseq [k (keys names)]
  (intern *ns* (symbol (str (names k) "s")) (ks k)))
(defn sixes [dice]
  (sixs dice)) ;; irregular plural

(defn chance [dice]
  (reduce + dice))

;; XXX TODO FIXME "Caused by: java.lang.RuntimeException: Unable to
;; resolve symbol: five_of_a_kind in this context"
(defn k_of_a_kind [k]
  (fn [dice]
    (if (>= (count (filter #{k} dice)) k)
      (reduce + dice)
      0)))

(let [kinds (select-keys names [3 4 5])]
  (doseq [k kinds]
    (intern *ns* (symbol (str (kinds k) "_of_a_kind"))
            (k_of_a_kind k))))

(defn yahtzee [dice]
  (if (not (zero? (five_of_a_kind dice)))
    50
    0))

(deftest test_twos
  (is (= 10 (twos [2 2 2 2 2]))))

(deftest test_four_of_a_kind
  (is (= 9 (four_of_a_kind [1 2 2 2 2])))
  (is (= 0 (four_of_a_kind [1 2 3 4 5]))))

(deftest yahtzee
  (is (= 50 (yahtzee [1 1 1 1 1])))
  (is (= 0 (yahtzee [1 1 1 1 2]))))

(deftest test_sample_output
  ;; TODO transcribe
)

(run-tests)