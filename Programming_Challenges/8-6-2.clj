(ns fifteen-puzzle
  (:require [clojure.test :refer :all]))

(def goal-state
  [[ :1  :2  :3  :4] 
   [ :5  :6  :7  :8] 
   [ :9 :10 :11 :12] 
   [:13 :14 :15 nil]])

(defn coordinates-of-void [state]
  (let [i (first (keep-indexed
                  (fn [i row] (if (= (count (filter identity row)) 3) i))
                  state))
        j (first (keep-indexed
                  (fn [j tile] (if-not tile j))
                  (state i)))]
    [i j]))


(deftest can-find-coordinates-of-void
  (is (= [3 3] (coordinates-of-void goal-state))))

