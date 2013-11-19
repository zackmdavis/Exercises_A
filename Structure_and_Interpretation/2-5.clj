(defn expt [b n]
  (if (= n 0)
    1
    (* b (expt b (dec n)))))

(defn cons [a b]
  (* (expt 2 a) (expt 3 b)))

(defn car [n]
  (loop [N n t 0]
    (if (not= (mod N 2) 0)
      t
      (recur (/ N 2) (inc t)))))
             
(defn cdr [n]
  (loop [N n t 0]
    (if (not= (mod N 3) 0)
      t
      (recur (/ N 3) (inc t)))))

; sanity check
(def x (cons 4 7))
(println (car x) (cdr x)) ; 4 7

; Actually it's better than this; we can encode arbitrarily long lists
; in an integer, too, not just pairs ...
