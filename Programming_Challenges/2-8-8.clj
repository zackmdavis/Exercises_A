(ns yahtzee
  (:use clojure.test))

(defn ms [m]
  (fn [dice] (reduce + (filter #{m} dice))))

(def names {1 'one 2 'two 3 'three 4 'four 5 'five 6 'six})

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

(defn gaps [dice]
  (let [ordered (sort dice)] 
    (map #(- %2 %1) ordered (rest ordered))))

(defn short_straight [dice]
  (if (= 3 (count_of_kind_m (gaps dice) 1))
    25
    0))

(defn long_straight [dice]
  (if (= 4 (count_of_kind_m (gaps dice) 1))
    35
    0))

(defn full_house [dice]
  (if (every? #(not= 0 %) (for [k [2 3]] ((k_of_a_kind k) dice)))
    50
    0))

(def categories [ones twos threes fours fives sixes
                 three_of_a_kind four_of_a_kind roll_yahtzee
                 short_straight long_straight
                 full_house])

(defn roll []
  (repeatedly 5 #(inc (rand-int 6))))

(deftest test_twos
  (is (= 10 (twos [2 2 2 2 2]))))

(deftest test_four_of_a_kind
  (is (= 9 (four_of_a_kind [1 2 2 2 2])))
  (is (= 0 (four_of_a_kind [1 2 3 4 5]))))

(deftest test_short
  (is (= 25 (short_straight [2 1 3 6 4])))
  (is (= 0 (short_straight [1 2 3 2 2]))))
  
(deftest test_long
  (let [exemplars [[1 2 3 4 5] [2 3 4 5 6]]]
    (doseq [example exemplars]
      (is (= 35 (long_straight example))))
    (doseq [outcome (repeatedly 10 roll)]
      (if (some #{(sort outcome)} exemplars)
        (is (= 35 (long_straight outcome)))
        (is (= 0 (long_straight outcome)))))))

(deftest test_full_house
  (is (= 50 (full_house [1 1 3 3 3])))
  (is (= 0 (full_house [1 2 3 4 5]))))

(deftest test_roll_yahtzee
  (is (= 50 (roll_yahtzee [1 1 1 1 1])))
  (is (= 0 (roll_yahtzee [1 1 1 1 2]))))

(deftest test_sample_output
  ;; TODO transcribe
)

(run-tests)