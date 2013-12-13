(defn union [S T]
  (cond (empty? S) T
        (empty? T) S
        (let [s1 (first S)
              t1 (first T)]
          (cond (= s1 t1) (cons s1 (union (rest S) (rest T)))
                (< s1 t1) (cons s1 (union (rest S) T))
                (> s1 t1) (cons s1 (union S (rest T)))))))

(println (intersection (list 1 2 3 4) (list 3 4 5 6))) 
(println (intersection  (list 3 4 5 6) (list 1 2 3 4)))

; TODO
;; Exception in thread "main" java.lang.IllegalArgumentException: cond
;; requires an even number of forms --- it looked even to me! 