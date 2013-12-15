(defn monte-carlo [experiment trials]
  (loop [successes 0 remaining trials]
    (cond (= remaining 0) (/ successes trials)
          (experiment) (recur (inc successes) (dec trials))
          :else (recur successes (dec trials)))))

(defn random-in-range [low high]
  (let [interval (- high low)]
    (+ low (rand interval))))

(defn estimate-integral [predicate x0 x1 y0 y1 trials]
  (let [sample-point (fn []
                       (let [point (list (random-in-range x0 x1)
                                         (random-in-range y0 y1))]
                         (predicate point)))]
    (/ (monte-carlo sample-point trials)
       (* (- x1 x0) (- y1 y0)))))

(defn distance-from-origin [point]
  (let [x (first point)
        y (second point)]
    (Math/sqrt (+ (* x x) (* y y)))))

(defn estimate-pi-from-unit-circle [trials]
  (let [in-circle (fn [point]
                    (< (distance-from-origin point) 1))]
        (estimate-integral in-circle -1 1 -1 1 trials)))

(println (estimate-pi-from-unit-circle 5))

; TODO debug =>
;; $ time !!
;; time clojure 3-5.clj 
;; ^C
;; real    0m55.905s
;; user    0m56.584s
;; sys     0m0.316s
;
; it shouldn't take that long; one imagines the loop/recur is probably
; failing to terminate somewhere?