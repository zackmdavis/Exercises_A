(ns cuts
  (:require [clojure.test :refer :all]))

;; The Cormen et al. book has a similar rod-cutting example to
;; introduce dynamic programming in Section 15.1 ...

(defn enumerate [sequence]
  (map-indexed vector sequence))

(defn argmax_and_max [sequence]
  (reduce #(if (apply > (map second [%2 %1])) %2 %1)
          (enumerate sequence)))

(defn revenue_at_cut [prices revenues next_size proposal]
  (+ (prices proposal) (revenues (- next_size proposal))))

(defn possible_revenues [prices revenues next_size]
  (map #(revenue_at_cut prices revenues next_size %)
       (range 1 (inc next_size))))

(defn extend_solution [[prices revenues first_cuts]]
  (let [next_size (count revenues)
        possible_revenues (possible_revenues prices revenues next_size)
        [next_choice next_revenue] (argmax_and_max possible_revenues)]
    [prices
     (conj revenues next_revenue)
     (conj first_cuts (inc next_choice))]))

(defn solution_table [prices n]
  (take n (iterate extend_solution [prices [0] [0]])))

(def example_prices     [0  1  5  8  9 10 17 17 20 24 20])
(def example_revenues   [0  1  5  8 10 13 17 18 22 25 30])
(def example_first_cuts [0  1  2  3  2  2  6  1  2  3 10])

(deftest test_revenue_at_cut
  (is (= 13
         (revenue_at_cut (subvec example_prices 0 4)
                         (subvec example_revenues 0 4)
                         5 2))))

(deftest test_possible_revenues
  (let [expectations (zipmap (range 1 (inc 5))
                               [[1] [2 5] [6 6 8] [9 10 9 9]
                                [11 13 13 10 10]])]
    (doseq [expectation expectations]
      (let [size (key expectation)]
        (is (= (possible_revenues example_prices example_revenues size)
               (val expectation)))))))

(deftest test_solution_table
  (let [[_ revenues first_cuts] (last (solution_table example_prices 10))]
    (is (= revenues (subvec example_revenues 0 10)))
    (is (= first_cuts (subvec example_first_cuts 0 10)))))

(run-tests)

;; However, while the problem posed in the Skiena and Revilla text
;; which is the focus of this directory looks similar at a glance,
;; it's quite importantly different in that the order in which we cut
;; pieces matters ...
