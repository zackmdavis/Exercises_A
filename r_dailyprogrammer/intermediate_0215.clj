;; http://www.reddit.com/r/dailyprogrammer/
;; comments/36m83a/20150520_challenge_215_intermediate_validating/

(ns sorting-network-validation
  (:refer-clojure :exclude [sorted?])
  (:require [clojure.test :refer :all]))

;;; XXX seriously, fuck macros
;; (defmacro bitproduct [n]
;;   (let [indicy (fn [i] (symbol (str "i" i)))
;;         factor (range n)]
;;     `(for ~(vec (apply concat (for [i factor]
;;                                 [(indicy i) `(range 2)])))
;;        ~(vec (for [i (range n)] (indicy i))))))

(defn conj-maybe [collection & noobs]
  ;; XXX BUT NOT MY FAULT: `conj` should just have an arity-one form
  ;; (reused from Easy #121)
  (if (empty? noobs)
    collection
    (apply conj collection noobs)))

(defn cartesian-product [& factors]
  ;; with guidance from http://stackoverflow.com/a/18248031
  (if (empty? factors)
    [[]]
    (for [head (first factors)
          tail (apply cartesian-product (rest factors))]
      (apply conj-maybe [head] tail))))

(defn bitproduct [n]
  (apply cartesian-product (repeat n [0 1])))

(defn sorted? [collection]
  (apply <= collection))

(defn swap [collection i j]
  (assoc collection i (collection j) j (collection i)))

(defn swap-if->= [collection [i j]]
  (if (apply >= (map collection [i j]))
    (swap collection i j)
    collection))

(defn apply-sorting-network [collection comparators]
  (reduce (fn [collection comparator]
            (swap-if->= collection comparator))
          collection
          comparators))

(defn sorting-network-validates? [n network]
  (every? sorted? (for [possible-world (bitproduct (eval n))]
                    (apply-sorting-network possible-world network))))

(deftest test-sample-output
  (is (sorting-network-validates? 4 [[0 2] [1 3] [0 1] [2 3] [1 2]]))
  (is (not (sorting-network-validates?
            8
            [[0 2] [1 3] [0 1] [2 3] [1 2] [4 6] [5 7]
             [4 5] [6 7] [5 6] [0 4] [1 5] [2 6] [3 7]
             [2 4] [3 5] [1 2] [3 4] [6 7]]))))

(run-tests)
