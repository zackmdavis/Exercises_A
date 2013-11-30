(defn unique-pairs [n]
  (for [i (range n) j (range (+ i 1) n)] (list i j)))

(defn unique-triples [n]
  (for [i (range n) j (range (+ i 1) n) k (range (+ j 1) n)] (list i j k)))

(defn distinct-triple-sum [n s]
  (filter (fn [triple] (= (reduce + triple) s)) (unique-triples n)))

; check
(println (distinct-triple-sum 10 12))