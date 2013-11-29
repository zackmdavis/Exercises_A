(defn make-mobile [left right]
  (list left right))

(defn make-branch [length structure]
  (list length structure))

(defn left [mobile]
  (first mobile))

(defn right [mobile]
  (last mobile))

(defn length [branch]
  (first branch))

(defn structure [branch]
  (last branch))

(defn weight [structure]
  (if (integer? (last structure))
    (last structure)
    (reduce + (map weight structure))))

(defn torque [branch]
  (* (length branch) (weight (structure branch))))

;; (defn balanced? [mobile]
;;   (= (torque (left mobile)) (torque (right mobile))))
;; TODO: Actually, also need to consider if all submobiles are
;; balanced.

; check 
(let [mobile-a (make-mobile
                (make-branch 2 3)
                (make-mobile
                 (make-branch 2 4)
                 (make-branch 2 6)))]

(println (weight mobile-a)) ; => 13
)