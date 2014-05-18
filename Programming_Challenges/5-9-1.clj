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

;;;; I am bad at programming
;; (defn carries [summands]
;;   (let [places (max (map #(count (str %)) summands))
;;         digit_streams (map #(zero-padding (digits %) places) summands)]
;;     (count (filter #(not zero? 5)
;;                    (map second
;;                         (reductions schoolbook [0 0]
;;                                     digit_streams))))))


(deftest test_known_digits
  (is (= [3 1 4 1 5 9 2] (digits 2951413))))

(deftest test_zero-padding
  (is (= [2 3 5 0 0 0] (zero-padding [2 3 5] 6))))

(deftest test_sample_output
  (is (= 0 (carries [123 456])))
  (is (= 3 (carries [555 555])))
  (is (= 1 (carries [123 594]))))

(run-tests)
