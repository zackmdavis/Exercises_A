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

;; I feel like I write this function way too often
(defn lookup [state coordinates]
  ((state (first coordinates)) (second coordinates)))

;; this too
(defn write [state [row col] value]
  (assoc state row (assoc (state row) col value)))

;; and this
(defn displace [coordinates direction]
  (vec (map + coordinates direction)))

;; how do we endure the hegemony of two-dimensional grids?
(defn in-bounds? [coordinates]
  (every? #(< % 4) coordinates))

(defn slide [state direction]
  (let [void-place (coordinates-of-void state)
        replace-place (displace void-place direction)
        replacement (lookup state replace-place)]
    (if (in-bounds? replace-place)
      (write (write state void-place replacement) replace-place nil)
      state)))


(deftest can-find-coordinates-of-void
  (is (= [3 3] (coordinates-of-void goal-state))))

(deftest can-slide
  (is (= [[ :1  :2  :3  :4]
          [ :5  :6  :7  :8]
          [ :9 :10 :11 :12]
          [:13 :14 nil :15]]
         (slide goal-state [0 -1]))))

(run-tests)
