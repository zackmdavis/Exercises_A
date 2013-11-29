;; this is WRONG
;; TODO: solve this exercise correctly

(defn union [set1 set2]
  (set (concat (seq set1) (seq set2))))

(defn powerset [set]
  (if (empty? set)
    #{#{}}
    (let [subproblem (powerset (rest set))]
      (union
       subproblem
       (map
        (fn [subset] (union subset (first set)))
        subproblem)))))

; check
(let [ps3 (powerset #{:1 :2 :3})]
  (println ps3 (count ps3)))

(let [ps4 (powerset #{:1 :2 :3 :4})]
  (println ps4 (count ps4)))