(ns flapjacks
  (:use clojure.test))

;; not using the same indexing or order conventions as the book and
;; it's hardly your concern, especially conditional on me being the
;; sort of person who writes Clojure in snake_case

(defn flip [stack j]
  (let [[fixed to_flip] (split-at j stack)]
    (concat fixed (reverse to_flip))))

(defn prophecy [stack]
  (let [destiny (sort stack)]
    (into {}
          (for [cake stack]
            [cake (.indexOf stack cake)]))))

(defn pivot_to_first [stack j]
  (flip (flip stack j) 0))

(defn flapjack_sort [stack]
  ;; TODO
)

(deftest test_can_flip
  (is (= (flip [0 1 2 3 4] 2)
         [0 1 4 3 2])))

(deftest test_pivot_to_first
  (doseq [_ (range 50)]
    (let [stack (vec (shuffle (range 10)))
          pivot (rand-int 10)
          pivot_point (.indexOf stack pivot)]
      (is (= (first (pivot_to_first stack pivot_point))
             pivot)))))

(deftest test_flapjack_sort
  ;; TODO
)

(run-tests)
