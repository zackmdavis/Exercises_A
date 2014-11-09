(ns bee-maja
  (:require [clojure.test :refer :all]))

(defn arc [j]
  (let [repetitions [j (dec j) j j j j]]
    (apply concat (map repeat repetitions [:D :SW :NW :U :NE :SE]))))

(def direct-to-displace
  {:D  [ 0  1]
   :U  [ 0 -1]
   :SE [ 1  0]
   :NW [-1  0]
   :NE [ 1 -1]
   :SW [-1  1]})

(def ℕ+ (drop 1 (range)))

(def directs (apply concat (map arc ℕ+)))

(def displaces (map direct-to-displace directs))

(def places (reductions #(map + %1 %2) [0 0] displaces))

(defn convert [n]
  (nth places (dec n)))

(deftest test-conversion
  (is (= [[0 0] [0 1] [-1 1] [-1 0] [0 -1] [1 -1] [ 1  0]
          [1 1] [0 2] [-1 2] [-2 2] [-2 1] [-2 0] [-1 -1]])
      (map convert (range 1 15)))
  (is (= [-1 3] (convert 23))))

(run-tests)
