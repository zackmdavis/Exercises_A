(import [3_streams [*]])
(require 3_streams)

(require pairs)

(defn divisible? [x y]
  (= (% x y) 0))

(def no-sevens
  (stream-filter (λ [k] (not (divisible? k 7))) ℕ+))

(assert (= (stream-nth no-sevens 100) 117))

(defn fibgen [a b]
  (cons-stream a (fibgen b (+ a b))))

(def fibs (fibgen 0 1))

(assert-stream-begins! fibs [0 1 1 2 3 5 8])

(defn sieve [stream]
  (cons-stream
   (stream-car stream)
   (sieve (stream-filter
           (λ [k] (not (divisible? k (stream-car stream))))
           (stream-cdr stream)))))

(def primes (sieve (integers-from 2)))

(assert-stream-begins! primes [2 3 5 7 11 13 17 19 23 29 31 37
                               41 43 47 53 59 61 67 71])

(def ones (cons-stream 1 ones))
(def alternatively-defined-integers
  (cons-stream 1
               (add-streams ones
                            alternatively-defined-integers)))

;; Exercise 3.54
(def factorials (cons-stream 1 (multiply-streams ℕ+ factorials)))

;;(printstream-until (partial-sums ℕ) 10)


(def just-twos-and-fives (merge (scale-stream ℕ 2) (scale-stream ℕ 5)))
(assert-stream-begins! just-twos-and-fives [0 2 4 5 6 8 10 12])

(printstream-until just-twos-and-fives 20)

;; Exercise 3.56
