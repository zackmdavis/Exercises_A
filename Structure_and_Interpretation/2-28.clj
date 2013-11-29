(defn fringe [tree]
  (loop [i 0]
    (if (< i (count tree))
      (let [branch (nth tree i)]
        (do
          (if (seq? branch)
            (fringe branch)
            (print branch " "))
          (recur (inc i)))))))

; check
(let [test-tree-a (list (list 1 2) (list 3 4))
      test-tree-b (list (list :a) (list :b (list :c :d) :e))]
  (fringe test-tree-a)
  (println)
  (fringe test-tree-b)
  (println))