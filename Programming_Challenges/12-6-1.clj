(ns ant_on_a_chessboard
  (:require [clojure.test :refer :all]))

(defn arc [j clockwise?]
  (let [directions (if clockwise?
                         [[1 0] [0 -1]]
                         [[0 1] [-1 0]])]
    (apply concat (map #(repeat (dec j) %) directions))))

(defn join_after [clockwise?] 
  (if clockwise?
    [[1 0]]
    [[0 1]]))

(defn step [position offset]
  (map + position offset))

(def ℕ+ (drop 1 (range)))

(def journey
  (apply concat
         (interleave (map #(arc % (= (mod % 2) 0)) ℕ+)
                     (apply interleave
                            (map #(repeat (join_after %))
                                 [true false])))))

(def readable {[1 0] :right, [0 -1] :down, [0 1] :up, [-1 0] :left})

(def the_historical_record
  (reductions step [1 1] journey))

(deftest test_arc
  (is (= [[1 0] [1 0] [0 -1] [0 -1]]
         (arc 3 true)))
  (is (= []
         (arc 1 false))))

(deftest test_the_historical_record
  (is (= (nth the_historical_record 7) [2 3]))
  (is (= (nth the_historical_record 19) [5 4]))
  (is (= (nth the_historical_record 24) [1 5])))

(run-tests)
