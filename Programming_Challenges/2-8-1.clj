(ns jolly_jumpers
  (:use clojure.test))

(defn first_difference [sequence]
  (if (seq (rest sequence))
    (Math/abs (- (first sequence) (first (rest sequence))))))

(defn jolly? [sequence]
  (= (set (range 1 (count sequence)))
     (set (filter identity
                  (map first_difference
                       (take-while #(seq %)
                                   (iterate rest sequence)))))))

(deftest test_first_difference
  (is (= 3 (first_difference [1 4 3 4 5]))))

(deftest test_sample_output
  (is (jolly? [1 4 2 3]))
  (is (not (jolly? [1 4 2 -1 6]))))

(run-tests)