(require 'clojure.set)

(defn offset [position1 position2]
  (list (- (first position2) (first position1))
        (- (last position2) (last position1))))

(defn attackable [position1 position2]
  (or (= (first position1) (first position2))
      (= (last position1) (last position2))
      (let [difference (offset position1 position2)]
        (= (Math/abs (first difference))
           (Math/abs (last difference))))))

(defn safe? [candidate placed]
  (empty? (filter
           (fn [extant] (attackable extant candidate))
           placed)))

(defn rank-positions [rank n]
  (map (fn [file] (list rank file)) (range n)))

; TODO solve the n-queens problem  

;; (defn queens [n]
;;   (if (= n 0)
;;     #{}

