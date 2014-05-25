(ns boring
  (:use clojure.test))

;; Fibonacci numbers again!? What are we, schoolchildren in the third
;; grade? But it is an okay way to see how Clojure's memoize function
;; works I suppose

(defn naïve [n]
  (if (< n 2)
    n
    (+ (naïve (- n 1)) (naïve (- n 2)))))

(def better (memoize naïve))

(def naïve_zählen (map naïve (range)))

(def better_zählen (map better (range)))

(deftest test_naïveté
  (is (= [0 1 1 2 3 5 8 13 21 34]
         (take 10 naïve_zählen))))

(deftest test_better
  (is (= [0 1 1 2 3 5 8 13 21 34]
         (take 10 better_zählen))))

(defn timing [n]
  (print "better ")
  (time (println (take n better_zählen)))
  (print "naïve ")
  (time (println (take n naïve_zählen))))

(run-tests)
(timing 33)

;; "naïve" is outperforming "better" in a timing test by more than 1.1
;; orders of magnitude for n less than or equal to 32, but then at 33,
;; the test takes longer than I have patience. I know the number of
;; calls blows up, but I didn't expect the edge between feasible and
;; not to be quite that sharp, but far more importantly, I expected
;; "memoize" to help. I don't know what black magic Clojure is doing,
;; but it may not be a coincidence that the phase transition happens
;; at a power of two?
