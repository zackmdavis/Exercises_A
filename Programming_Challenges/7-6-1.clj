(ns light_more_light
  (:use clojure.test))

(defn on [n]
  (= (mod (count (filter #(= (mod n %) 0) (range 1 (inc n)))) 2) 1))

(defn threading-on [n]
  (->> 1 (= (mod (count (filter #(= (mod n %) 0) (->> n (inc) (range 1)))) 2))))
;; (Yes, that was trivial and pointless, but I never use the threading
;; macros, and I should know how to use the threading macros.)

(deftest test_sample_output
  (is (not (on 3)))
  (is (on 6241))
  (is (not (on 8191))))

(defn square? [n]
  (let [square-root (Math/sqrt n)]
    (== square-root (int square-root))))

(deftest test-secret-insight-from-having-seen-and-solved-this-problem-elsewhere
  (doseq [i (range 1 100)]
    (is (= (on i) (square? i)))))

(deftest test-equivalence
  (is (apply = (for [procedure-name ['on 'threading-on 'square?]]
                 (map (ns-resolve *ns* procedure-name)
                      (range 1 100))))))

(run-tests)
