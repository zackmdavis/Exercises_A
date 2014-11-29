(ns edit-step-ladders
  (:require [clojure.test :refer :all]))

(defn substitution? [one-word another]
  (and (apply = (map count [one-word another]))
       (= 1 (count (remove identity (map #(= %1 %2) one-word another))))))

(defn once-imposed [longer shorter]
  ;; TODO
  )

(defn insertion-or-deletion? [one-word another]
  (and (= 1 (abs (apply - (map count [one-word another]))))
       ;; TODO
       ))

(deftest test-substitution-predicate
  (is (substitution? "rah" "bah"))
  (is (substitution? "America" "Americo"))
  (is (substitution? "distraction" "distruction"))
  (is (not (substitution? "alleged" "fragmentary"))))

(run-tests)
