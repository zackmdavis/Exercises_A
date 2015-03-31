;; http://www.reddit.com/r/dailyprogrammer/comments/19mn2d/030413_challenge_121_easy_bytelandian_exchange_1/

(ns bytelandian-exchange-1
  (:require [clojure.test :refer :all]))

(defn exchange [k]
  {:pre [(= (type k) java.lang.Long) (pos? k)]}
  (let [denomination-denominators [2 3 4]]
    (map quot
         (repeat (count denomination-denominators) k)
         denomination-denominators)))

(defn conj-maybe [collection & noobs]
  (if (empty? noobs)
    collection
    (apply conj collection noobs)))

(defn exchange-feeder [real-coins pennies]
  (if (empty? real-coins)
    pennies
    (let [change (exchange (peek real-coins))
          {penny-change true real-change false} (group-by zero? change)]
      (exchange-feeder (apply conj-maybe (pop real-coins) (or real-change []))
                       (+ pennies (count penny-change))))))

(deftest test-sample-output
  (is (= (exchange-feeder [7] 0) 15)))

(run-tests)
