(ns yahtzee
  (:use clojure.test))

(defn ms [m]
  (fn [dice] (reduce + (filter #{m} dice))))

(def names {1 'one 2 'two 3 'three 4 'four 5 'five 6 'six})

(println (keys (ns-publics 'yahtzee)))

(doseq [m (keys names)]
  (intern *ns* (symbol (str (names m) "s")) (ms m)))
(defn sixes [dice]
  (sixs dice)) ;; irregular plural

(defn chance [dice]
  (reduce + dice))

(defn count_of_kind_m [dice m]
  (count (filter #{m} dice)))

(defn k_of_a_kind [k]
  (fn [dice]
    (if (>= (apply max (map (partial count_of_kind_m dice)
                            (keys names)))
            k)
      (reduce + dice)
      0)))

(let [kinds (select-keys names [3 4 5])]
  (doseq [k (keys kinds)]
    (intern *ns* (symbol (str (kinds k) "_of_a_kind"))
            (k_of_a_kind k))))


(defn roll_yahtzee [dice]
  (if (not (zero? (five_of_a_kind dice)))
    50
    0))

(deftest test_twos
  (is (= 10 (twos [2 2 2 2 2]))))

(deftest test_four_of_a_kind
  (is (= 9 (four_of_a_kind [1 2 2 2 2])))
  (is (= 0 (four_of_a_kind [1 2 3 4 5]))))

(deftest yahtzee
  (is (= 50 (roll_yahtzee [1 1 1 1 1])))
  (is (= 0 (roll_yahtzee [1 1 1 1 2]))))

(deftest test_sample_output
  ;; TODO transcribe
)

(run-tests)