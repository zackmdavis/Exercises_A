(defn for-each [f collection]
  (loop [i 0]
    (f (nth collection i))
    (if (< i (dec (count collection)))
      (recur (inc i)))))

; check
(for-each println '("I" "used" "to" "wonder"))