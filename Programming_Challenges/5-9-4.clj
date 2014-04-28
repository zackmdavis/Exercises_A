(ns ones
  (:use clojure.test)
  (:require [clojure.string :as string]))

(defn ones? [n]
  (re-matches #"1+" (str n)))

(defn multiples [n]
  (map #(* n %) (range)))

(defn result [n]
  (count (str (some ones? (multiples n)))))

(deftest test_sample_output
  (is (= 3 (result 3)))
  (is (= 6 (result 7)))
  (is (= 12 (result 9901))))

(run-tests)
