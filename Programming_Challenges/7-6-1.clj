(ns light_more_light
  (:use clojure.test))

(defn on [n]
  (= (mod (count (filter #(= (mod n %) 0) (range 1 (inc n)))) 2) 1))

(deftest test_sample_output
  (is (not (on 3)))
  (is (on 6241))
  (is (not (on 8191))))

(run-tests)
