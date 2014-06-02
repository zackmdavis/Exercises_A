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

(defn pivot_j_to_k [stack j k]
  (flip (flip stack j) k))

(defn pivot_to_first [stack j]
  (flip (flip stack j) 0))

;; XXX TODO FIXME this does not work
(defn flapjack_sort [stack]
  (let [destiny (sort stack)]
    (loop [current stack
           prophecies destiny]
      (if (seq prophecies)
        current
        (recur (pivot_j_to_k current (.indexOf current
                                               (first prophecies))
                             (- (count prophecies) (count current)))
               (rest prophecies))))))

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
  (doseq [_ (range 15)]
    (let [stack (vec (repeatedly 8 #(rand-int 15)))]
      (is (= (sort stack) (flapjack_sort stack))))))

(run-tests)
