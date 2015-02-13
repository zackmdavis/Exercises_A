(import itertools)

(def primes [2 3 5 7 11 13 17 19 23 29 31 37 41 43 47 53 59 61
             67 71 73 79 83 89 97 101 103])

(defn primes-less-than [x]
  (list (itertools.takewhile (fn [p] (< p x)) primes)))

(defn searcher [target components]
  (if (and (= (len components) 4)
           (= target (sum components)))
    components
    (first
     (list
     (filter (fn [x] (if x x))
             (genexpr (searcher target (+ components [p]))
                      [p (primes-less-than target)]
                      (<= (sum (+ components [p])) target)))))))


(print (searcher 26 [])) ;; IndexError: list index out of range
