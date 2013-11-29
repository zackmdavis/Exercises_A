(defn count-leaves [tree]
  (reduce + 0 (map 
               (fn [branch]
                 (if (list? branch)
                   (count-leaves branch)
                   1))
               tree)))

;
(let [my-tree (list :1
                    (list :2
                          :3
                          (list
                           (list :4
                                 :5)
                           :6)
                          :7)
                    :8
                    :9
                    (list :10
                          :11))]
  (println my-tree) ; => (:1 (:2 :3 ((:4 :5) :6) :7) :8 :9 (:10 :11))
  (println (count-leaves my-tree))) ; => 11