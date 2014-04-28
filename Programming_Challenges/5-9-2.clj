(ns reverse_and_add
  (:use clojure.test)
  (:require [clojure.string :as string]))

(defn palindrome? [n]
  (let [s (str n)]
    (if (= (string/reverse s) s)
      s)))

(defn reverse_and_add_procedure [n]
  (+ n (Integer. (string/reverse (str n)))))

(defn result [n]
  (let [applications (take 1000 (iterate reverse_and_add_procedure n))
        answer (Integer. (some palindrome? applications))]
    [(.indexOf applications answer) answer]))

(deftest test_sample_output
  (is (= [4 9339] (result 195)))
  (is (= [5 45254] (result 265)))
  (is (= [3 6666] (result 750))))

(run-tests)