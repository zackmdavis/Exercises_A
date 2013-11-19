(defn make-segment [start end]
  [start end])

(defn start-segment [segment]
  (first segment))

(defn end-segment [segment]
  (last segment))

(defn make-point [x y]
  [x y])

(defn x-coordinate [point]
  (first point))

(defn y-coordinate [point]
  (last point))

(defn print-point [point]
  (println "(" (x-coordinate point) ", " (y-coordinate point) ")"))

(defn midpoint-segment [segment]
  (make-point
   (/ (+ (x-coordinate (start-segment segment))
         (x-coordinate (end-segment segment)))
      2)
   (/ (+ (y-coordinate (start-segment segment))
         (y-coordinate (end-segment segment)))
      2)))

; sanity check
(println (midpoint-segment
          (make-segment
           (make-point 0 0)
           (make-point 5 5))))
