(ns fifteen-puzzle
  (:require [clojure.test :refer :all]))

(def goal-state
  [[ :1  :2  :3  :4]
   [ :5  :6  :7  :8]
   [ :9 :10 :11 :12]
   [:13 :14 :15 nil]])

(defn coordinates-of-k [state k]
  (let [i (first (keep-indexed
                  (fn [i row]
                    (if (= (count (filter #(not= k %) row))
                           (dec (count state)))
                      i)) ; my whitespace around here
                  state)) ; is kind of awful somehow
        j (first (keep-indexed
                  (fn [j tile] (if (= tile k) j))
                  (state i)))]
    [i j]))

(defn coordinates-of-void [state]
  (coordinates-of-k state nil))

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
(defn in-bounds? [coordinates bound]
  (every? #(<= 0 % bound) coordinates))

(defn slide [state direction]
  (let [void-place (coordinates-of-void state)
        replace-place (displace void-place direction)]
    (if (in-bounds? replace-place (dec (count state)))
      (let [replacement (lookup state replace-place)]
        (write (write state void-place replacement) replace-place nil))
      state)))

(defn optimism [dream reality]
  (count (filter identity
                 (for [tile (flatten dream)]
                   (not= (coordinates-of-k dream tile)
                         (coordinates-of-k reality tile))))))

(def directions [[0 -1] [0 1] [-1 0] [1 0]])

(defn nearby-possible-worlds [state]
  (for [direction directions]
    [direction (slide state direction)]))

(defn productive-daydreaming [state]
  (sort-by (fn [[_move world]] (optimism goal-state world))
           (nearby-possible-worlds state)))

(defn endeavor [state history]
  (if (= state goal-state)
    history
    (some identity
          (for [[move world] (productive-daydreaming state)]
            (endeavor world (conj history move))))))

(deftest can-find-coordinates-of-void
  (is (= [3 3] (coordinates-of-void goal-state))))

(deftest can-slide
  (is (= [[ :1  :2  :3  :4]
          [ :5  :6  :7  :8]
          [ :9 :10 :11 :12]
          [:13 :14 nil :15]]
         (slide goal-state [0 -1]))))

(deftest cannot-slide-out-of-bounds
  (is (= goal-state
         (slide goal-state [0 1]))))

(deftest test-optimism
  (is (= 2
         (optimism [[:1 :2] [:3 nil]] [[:2 :1] [:3 nil]]))))

(deftest test-sample-output
  (is (endeavor [[ :2  :3  :4 nil]
                 [ :1  :5  :7  :8]
                 [ :9  :6 :10 :12]
                 [:13 :14 :11 :15]]
                [])
      [[0 -1] [0 -1] [0 -1] [1 0] [0 1] [1 0] [0 1] [1 0] [0 1]]))

(run-tests)
