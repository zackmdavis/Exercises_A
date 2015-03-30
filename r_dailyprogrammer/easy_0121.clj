;; http://www.reddit.com/r/dailyprogrammer/comments/19mn2d/030413_challenge_121_easy_bytelandian_exchange_1/

(ns bytelandian-exchange-1
  (:require [clojure.test :refer :all]))

(defn exchange [k]
  ;; {:pre [(= (type k) java.lang.Long) (pos? k)]}
  (println k (type k))
  (let [denomination-denominators [2 3 4]]
    (map quot
         (repeat (count denomination-denominators) k)
         denomination-denominators)))

(defn exchange-feeder [real-coins pennies]
  (if (empty? real-coins)
    pennies
    (let [change (exchange (peek real-coins))
          {penny-change true real-change false} (group-by zero? change)]
      ;; XXX DREADFUL if it's not `conj` or `apply conj`, WTF is the
      ;; correct way to push onto the queue?!
      ;;
      ;; except, if the order we feed coins isn't relevant, there's
      ;; really no reason to use a queue instead of a plain vector,
      ;; except for the actual reason we are doing so, which is "Look
      ;; at me, I am a Serious Programmer who is worthy of life as
      ;; evidenced by how I'm using a PersistentQueue
      (exchange-feeder (apply conj (pop real-coins) real-change)
                       (+ pennies (count penny-change))))))

(defn singleton-queue [k]
  (conj (clojure.lang.PersistentQueue/EMPTY) k))

(deftest test-sample-output
  (is (= (exchange-feeder (singleton-queue 7) 0) 15)))

(run-tests)
