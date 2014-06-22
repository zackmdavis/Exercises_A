(ns hartals
  (:require [clojure.test :refer :all]))

(defn strike? [day strike_parameters]
  (and (not (contains? #{6 0} (mod day 7)))
       (some #(= % 0) (map #(mod day %) strike_parameters))))

(defn strike_sequence [days strike_parameters]
  (map #(strike? % strike_parameters)
       (range 1 (inc days))))

(defn count_strikes [days strike_parameters]
  (count (filter identity (strike_sequence days strike_parameters))))

(deftest test_sample_output
  (is (= 5 (count_strikes 14 [3 4 8])))
  (is (= 15 (count_strikes 100 [12 15 25 40]))))

(run-tests)
