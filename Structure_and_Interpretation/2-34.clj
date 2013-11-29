(defn horners-rule [x coefficients]
  (reduce (fn [a1 a0] (+ (* a1 x) a0)) coefficients))

; x^5 + 5x^3 + 3x + 1 at x:=2
(println (horners-rule 2 (list 1 0 5 0 3 1))) ; => 79