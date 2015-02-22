(import itertools)
(import sys)
(import time)

(def primes [2 3 5 7 11 13 17 19 23 29 31 37 41 43 47 53 59 61
             67 71 73 79 83 89 97 101 103])

(defn primes-less-than [x]
  (list (itertools.takewhile (fn [p] (< p x)) primes)))

(def total-calls 0)
(def start-time nil)

(defn searcher [target components &optional [call-depth 0]]
  (global total-calls)
  (global start-time)
  (+= total-calls 1)
  (if (not start-time)
    (setv start-time (time.time)))
  (if (and (= (len components) 4)
           (= target (sum components)))
    (do
     (sys.stderr.write

      (.format (+ "found solution {} for {} "
                  ;; XXX: it's always going to be depth 4, dummy
                  "at depth {} in the call tree "
                  "after {} calls and {} seconds\n")
               components target call-depth total-calls (- (time.time)
                                                           start-time)))
     ;; clean the campsite for next time
     (setv total-calls 0)
     (setv start-time nil)
     components)
    (next
     (filter (fn [x] (if x x))
             (genexpr (searcher target (+ components [p]) (inc call-depth))
                      ;; XXX: we need a smarter candidate-generator
                      ;; than this; we waste a lot of time iterating
                      ;; over everything like "2 + 2 + 2 + 3" when we
                      ;; know that's not going to be it
                      ;;
                      ;; PROPOSAL: sort `primes-less-than` by some
                      ;; criterion having to do with how far we are
                      ;; from the target and how many more primes we
                      ;; must requisition
                      [p (primes-less-than target)]
                      (<= (sum (+ components [p])) target)))
     nil)))


(defmacro deftest [&rest code] `(defn ~@code))  ; boilerplate sugar

(deftest test-primes-less-than []
  (assert (= (primes-less-than 10) [2 3 5 7])))

(defn validate [inputs]
  (for [i inputs]
    (let [[result (searcher i [])]]
      (when result
        (assert (all (map (lambda [k] (in k primes)) result)))
        (assert (= i (sum result)))))))

(deftest test-sample-input []
  (validate [24 36 46]))

(test-primes-less-than)
;; (test-sample-input) XXX: 46 is really slow

(validate (range 100))
