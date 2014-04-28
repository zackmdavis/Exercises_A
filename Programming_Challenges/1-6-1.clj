(ns collatz
  (:use clojure.test))

(defn collatz_procedure [n]
  (if (odd? n)
    (+ (* 3 n) 1)
    (/ n 2)))

(defn collatz_sequence [n]
  (concat (take-while #(not= 1 %) (iterate collatz_procedure n)) [1]))

(defn max_cycle [i j]
  (apply max (map #(count (collatz_sequence %)) (range i j))))

(deftest test_known_sequence
 (is (= [22 11 34 17 52 26 13 40 20 10 5 16 8 4 2 1]
        (collatz_sequence 22))))

(deftest test_sample_output
  (is (= 20 (max_cycle 1 11)))
  (is (= 125 (max_cycle 100 201)))
  (is (= 89 (max_cycle 201 211)))
  (is (= 174 (max_cycle 900 1001))))

(run-tests)