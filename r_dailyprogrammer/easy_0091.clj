;; http://reddit.com
;; /r/dailyprogrammer/comments/yqydh/8242012_challenge_91_easy_sleep_sort/
(ns lets-implement-sleepsort-in-clojure)

(def tick 10)

(defn sleepsort [& sortables]
  (let [sorted (atom [])
        sorting-silos (doall (for [item sortables]
                               (future (do (Thread/sleep (* tick item))
                                           (swap! sorted #(conj % item))))))]
    (loop []
      (if (every? identity (map realized? sorting-silos))
        @sorted
        (do (Thread/sleep tick) (recur))))))

(println (apply sleepsort (shuffle (range 20))))
