(defn abs [x]
  (if (< x 0)
    (* -1 x)
    x))

(defn gcd [a b]
  (if (= b 0)
    a
    (gcd b (mod a b))))

(defn make-rational [n d]
  (let [g (abs (gcd n d))]
    (let [sign (if (even?
                    (mod (count (filter (fn [x] (< x 0)) [n d])) 2))
                 1
                 -1)]
      [(* sign (/ (abs n) g))
       (/ (abs d) g)])))

; sanity check
(println (make-rational 4 16))
(println (make-rational 3 -24))
(println (make-rational -3 24))
(println (make-rational -5 -30))
