(ns factovisors
  (:require [clojure.test :refer :all])
  (:require [clojure.java.shell :refer [sh]])
  (:require [clojure.string :refer [split]]))

(defn factorial [k]
  (reduce * (range 1 (inc k))))

(defn factorize [m]
  (map read-string (rest (split
                          ;; /usr/bin/factor is my friend 😹
                          ((sh "factor" (str m)) :out)
                          #" |\n"))))

(defn itemized_number_bag [zahlen]
  (reduce (fn [bag consumable]
            (if (bag consumable)
              (update-in bag [consumable] inc)
              (assoc bag consumable 1)))
          {}
          zahlen))

(defn factor_bag [m]
  (itemized_number_bag (factorize m)))

(defn can_has_divisibility? [m n]
  (let [the_first_factor_bag (factor_bag m)
        the_second_factor_bag (factor_bag n)]
    (every? #(>= (get the_second_factor_bag (key %) 0) (val %))
         the_first_factor_bag)))

(defn factorial_factor_bag [k]
  (apply merge-with + (for [i (range 1 (inc k))]
                        (factor_bag i))))

(defn divides_factorial? [n k]
  (let [pedestrian_factor_bag (factor_bag n)
        relevant_factorial_factor_bag (factorial_factor_bag k)]
  (every? #(>= (get relevant_factorial_factor_bag (key %) 0) (val %))
          pedestrian_factor_bag)))
  
(deftest test_factorize
  (is (= [2 2 67957] (factorize 271828))))

(deftest test_can_has_divisibility
  (doseq [_ (range 20)]
    (let [n (+ (rand-int 10000) 1)
          m (+ (rand-int n) 1)]
      (is (= (can_has_divisibility? m n) (= (mod m n) 0))))))

(deftest test_factorial_factor_bag
  (doseq [i (range 1 12)]
    (is (= (factor_bag (factorial i)) (factorial_factor_bag i)))))

(deftest test_sample_output
  (is (divides_factorial? 9 6))
  (is (not (divides_factorial? 27 6)))
  (is (divides_factorial? 10000 20))
  (is (not (divides_factorial? 100000 20)))
  (is (not (divides_factorial? 1009 1000))))

(run-tests)
