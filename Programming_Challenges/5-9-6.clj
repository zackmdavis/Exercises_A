(ns coefficients
  (:require [clojure.test :refer :all]))

(defn factorial [m]
  (reduce * (range 1 (inc m))))

(defn multinomial_coefficient [n ks]
  (/ (factorial n) (reduce * (map factorial ks))))

(deftest test_multinomial_coefficient
  (is (= 34650 (multinomial_coefficient 11 [1 4 4 2]))))

(deftest test_sample_output
  ;; TODO
)

(run-tests)
