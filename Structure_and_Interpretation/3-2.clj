(defn make-monitored [f]
  (let [call-counter (atom 0)]
    (fn [& args]
      (cond (= (first args) :call-counter) @call-counter
            (= (first args) :reset-counter) (swap! call-counter
                                                   (fn [x] 0))
            :else (do (swap! call-counter inc)
                      (apply f args))))))
(defn square [x]
  (* x x))

(def monitored-square (make-monitored square))

(dotimes [i 5]
  (println (monitored-square i)))

(println "number of calls" (monitored-square :call-counter))
(monitored-square :reset-counter)
(println "number of calls after reset" (monitored-square :call-counter))

; =>
;; zmd@ExpectedReturn:~/Code/[...]$ clojure 3-2.clj
;; 0
;; 1
;; 4
;; 9
;; 16
;; number of calls 5
;; number of calls after reset 0 [OK!]