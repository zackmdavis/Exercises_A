(defn union [set1 set2]
  (set (concat (seq set1) (seq set2))))

(defn include-element [collection element]
  (map
   (fn [set] (union set #{element}))
   collection))

(defn powerset [set]
  (if (empty? set)
    #{#{}}
    (let [subproblem (powerset (rest set))]
      (union
       subproblem
       (include-element subproblem (first set))))))

; check
(let [ps3 (powerset #{:1 :2 :3})]
  (println ps3 (count ps3)))

(let [ps4 (powerset #{:1 :2 :3 :4})]
  (println ps4 (count ps4)))