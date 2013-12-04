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
  (map (fn [file] (list rank file))
       (range n)))

(defn superproblem-solns [placed n]
  (set 
   (map (fn [safe-position] (conj placed safe-position))
        (filter (fn [candidate] (safe? candidate placed))
                (rank-positions (count placed) n)))))

(defn queen-recur [k n]  
  (if (= k 0)
    #{}
    (reduce clojure.set/union
            (map (fn [subproblem] (superproblem-solns subproblem n))
                 (queen-recur (dec k) n)))))

;; TODO --- solve n-queens problem _correctly_
;; I'm doing something horribly wrong; with regrets, not sure exactly
;; what at the moment

;; also later TODO, generalize to n-hyperqueens problem

;; of interest: http://www.datagenetics.com/blog/august42012/

(println
(reduce clojure.set/union
        (map (fn [subproblem] (superproblem-solns subproblem 4))
             (superproblem-solns #{} 4)))
) ; => #{#{(0 0) (1 2)} #{(1 0) (0 2)} #{(0 0) (1 3)} #{(1 0) (0 3)} \
; #{(0 1) (1 3)} #{(1 1) (0 3)}} looks okay I guess

(println (queen-recur 1 4)) ; => #{} wtf!?