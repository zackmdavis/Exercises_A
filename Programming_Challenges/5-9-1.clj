(ns primary_arithmetic
  (:use clojure.test))

(defn zip [& seqs] 
  ;; why isn't this is in the standard library already??
  (apply map vector seqs))

(defn digits [n]
  (map #(int (mod % 10))
       (take-while #(>= % 1)
                   (for [i (range)] (/ n (Math/pow 10 i))))))

(defn zero-padding [ds k]
  (concat ds (repeat (- k (count ds)) 0)))

(defn schoolbook [[subresult carry] digits]
  (let [total (+ (reduce + digits) carry)]
    [(mod total 10) (quot total 10)]))

(defn standardize_arguments [args]
  (let [places (apply max (map #(count (str %)) args))]
    (map #(zero-padding (digits %) places) args)))

(defn carries [summands]
  (let [digit_streams (standardize_arguments summands)]
    (count (filter #(not (zero? %))
                   (map second
                        (reductions schoolbook [0 0]
                                    (apply zip digit_streams)))))))

(deftest test_known_digits
  (is (= [3 1 4 1 5 9 2] (digits 2951413))))

(deftest test_zero-padding
  (is (= [2 3 5 0 0 0] (zero-padding [2 3 5] 6))))

(deftest test_standardize_arguments
  (is (= [[3 2 1 0 0] [8 7 6 5 4]]
         (standardize_arguments [123 45678]))))

(deftest test_sample_output
  (is (= 0 (carries [123 456])))
  (is (= 3 (carries [555 555])))
  (is (= 1 (carries [123 594])))
  (is (= 4 (carries [92445 85555]))))
 
(run-tests)
