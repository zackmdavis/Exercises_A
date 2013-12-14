(defn make-accumulator [n]
  (let [start (atom n)]
    (fn [increment] (swap! start #(+ increment %)))))

(def A (make-accumulator 5))
(println (A 10))
(println (A 10))