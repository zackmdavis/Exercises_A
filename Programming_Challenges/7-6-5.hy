(import itertools)
(import sys)
(import time)

(def primes [2 3 5 7 11 13 17 19 23 29 31 37 41 43 47 53 59 61
             67 71 73 79 83 89 97 101 103])

(defn primes-less-than [x]
  (list (itertools.takewhile (fn [p] (< p x)) primes)))

(defn my-prime-sorting-heuristic [x choices-remaining]
  ;; XXX: I don't think we want to be in the
  ;; (= choices-remaining 0) situation in the first place
  (if (!= choices-remaining 0)
    (fn [p] (abs (- (/ x choices-remaining) p)))
    (fn [p] p)))

(defn primes-less-than-sorted-heuristically [x choices-remaining]
  (apply sorted
         [(primes-less-than x)]
         {"key" (my-prime-sorting-heuristic x choices-remaining)}))

(def total-calls 0)
(def start-time nil)

(defn searcher [target components]
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
                  "after {} calls and {} seconds\n")
               components target total-calls (- (time.time) start-time)))
     ;; clean the campsite for next time
     (setv total-calls 0)
     (setv start-time nil)
     components)
    (next
     (filter (fn [x] (if x x))
             (genexpr (searcher target (+ components [p]))
                      [p (primes-less-than-sorted-heuristically
                          target (- 4 (len components)))]
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
(test-sample-input)

(validate (range 100))

;; XXX: wacky five-to-have-and-a-half orders-of-magnitude
;; anti-optimization on some edge cases
;;
;; I think this is why serious people bother proving theorems about
;; their heuristics.
;;
;; found solution [19, 23, 31, 3] for 76 after 5 calls and
;; 0.0001068115234375 seconds
;; found solution [19, 19, 37, 2] for 77 after 894060 calls and
;; 33.284077167510986 seconds
;; found solution [19, 23, 31, 5] for 78 after 5 calls and
;; 0.00011324882507324219 seconds
