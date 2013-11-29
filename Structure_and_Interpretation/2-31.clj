(defn treemap [f tree]
  (map (fn [subtree]
         (if (list? subtree)
           (treemap f subtree)
           (f subtree)))
       tree))

(defn square-tree [tree]
  (treemap (fn [x] (* x x)) tree))

; check
(println (square-tree
          (list 1
                (list 2 (list 3 4) 5)
                (list 6 7))))
; => (1 (4 (9 16) 25) (36 49))