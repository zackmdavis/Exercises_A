(defn element? [e S]
  (cond (empty? S) false
        (== e (first S)) true
        :else (element? e (rest S))))

(defn adjoin [e S]
  (if (element? e S)
    S
    (cons e S)))

(defn intersection [S T]
  (cond (or (empty? S) (empty? T)) '()
        (element? (first S) T) (cons (first S)
                                     (intersection (rest S) T))
        :else (intersection (rest S) T)))

(def my-first-set (adjoin 2 (adjoin 1 '()))) ; => {1 2}
(def my-second-set (intersection (list 1 2 3) (list 1 2))) ; => {1 2}

(println my-first-set my-second-set) ; => (2 1) (1 2) [OK]

(defn union [S T]
  (cond (empty? S) T
        (empty? T) S
        (element? (first S) T) (union (rest S) T)
        :else (cons (first S) (union (rest S) T))))

(println (union (list 1 2 3) (list 3 4 5))) ; => (1 2 3 4 5) [OK]
