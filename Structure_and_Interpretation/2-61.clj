(defn element? [e S]
  (cond (empty? S) false
        (= e (first S)) true
        (< e (first S)) false
        :else (element? e (rest S))))

(defn intersection [S T]
  (if (or (empty? S) (empty? T)) '()
      (let [s1 (first S)
            t1 (first T)]
        (cond (= s1 t1) (cons s1 (intersection (rest S) (rest T)))
              (< s1 t1) (intersection (rest S) T)
              (> s1 t1) (intersection S (rest T))))))

(println (intersection (list 1 2 3 4) (list 3 4 5 6))) ; => (3 4) [OK]
(println (intersection  (list 3 4 5 6) (list 1 2 3 4))) ; => (3 4) [OK]