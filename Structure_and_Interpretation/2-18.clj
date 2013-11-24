(defn reverse [L]
  (if (<= (count L) 1)
    L
    (conj (reverse (drop-last 1 L)) (last L))))

; check
(println (reverse (list 1 2 3 4)))