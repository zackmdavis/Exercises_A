(import [functools [lru_cache :as memoize]])

(require hy.contrib.anaphoric)

(import [pairs [*]])
(require pairs)

;; TODO: experiment with memoization techniques (it would be
;; interesting if there is there a way to use the lru_cache decorator
;; with macros?)

(defmacro delay [code]
  `(λ [] ~code))

(defn force [what-was-delayed]
  (what-was-delayed))

(defmacro cons-stream [first rest]
  `(cons ~first (delay ~rest)))

(defn stream-car [stream]
  (car stream))

(defn stream-cdr [stream]
  (force (cdr stream)))

(defn stream-simple-map [procedure stream]
  (when stream
    (cons-stream (procedure (stream-car stream))
                 (stream-simple-map procedure (stream-cdr stream)))))

;; Exercise 3.50
(defn stream-map [procedure &rest streams]
  (when (car streams)
    (cons-stream
     (apply procedure (map stream-car streams))
     (apply stream-map (cons procedure (map stream-cdr streams))))))

(defn stream-filter [predicate? stream]
  (let [[next (stream-car stream)]]
    (if (predicate? next)
      (cons-stream next (stream-filter predicate? (stream-cdr stream)))
      (stream-filter predicate? (stream-cdr stream)))))

(defn stream-nth [stream n]
  (if (= n 0)
    (stream-car stream)
    (stream-nth (stream-cdr stream) (dec n))))

(defn call-for-each [procedure stream]
  (when stream
    (do (procedure (stream-car stream))
        (call-for-each procedure (stream-cdr stream)))))

(defn printstream  ; I can't articulate why I want to spell this as one word
  [stream]
  (call-for-each (λ [item] (apply print [item] {"end" " "})) stream)
  (print))

(defn printstream-until [stream n]
  (if (and stream (> n 0))
    (do
     (apply print [(stream-car stream)] {"end" " "})
     (printstream-until (stream-cdr stream) (dec n)))
    (print)))

(defn integers-from [n]
  (cons-stream n (integers-from (inc n))))

(def ℕ (integers-from 0))
(def ℕ+ (integers-from 1))

(defn add-streams [first-stream second-stream]
  (stream-map (λ [a b] (+ a b)) first-stream second-stream))

(defn scale-stream [stream factor]
  (stream-map (λ [x] (* factor x)) stream))

(defn multiply-streams [first-stream second-stream]
  (stream-map (λ [a b] (* a b)) first-stream second-stream))

(defn exponentiate-stream [stream power]
  (stream-map (λ [x] (** x power)) stream))

(defn merge [first-stream second-stream]
  (cond [(not first-stream) second-stream]
        [(not second-stream) first-stream]
        [:else
         (let [[first-car (stream-car first-stream)]
               [second-car (stream-car second-stream)]]
           (cond [(< first-car second-car)
                  (cons-stream first-car (merge (stream-cdr first-stream)
                                                second-stream))]
                 [(< second-car first-car)
                  (cons-stream second-car (merge (stream-cdr second-stream)
                                                 first-stream))]
                 [:else
                  (cons-stream first-car (apply merge
                                                (list
                                                 (map stream-cdr
                                                      [first-stream
                                                       second-stream]))))]))]))

(defmacro/g! assert-stream-begins! [stream reference-list]
  `(for [[g!i reference-item] (enumerate ~reference-list)]
     (assert (= reference-item (stream-nth ~stream g!i)))))

(defn monotone-stream [x]
  (cons-stream x (monotone-stream x)))

(defn multitone-stream [iterable]
  (cons-stream (first iterable)
               (multitone-stream (+ (rest iterable) [(first iterable)]))))

;; Exercise 3.55
(defn partial-sums [stream]
  (cons-stream (stream-car stream)
               (add-streams (monotone-stream (stream-car stream))
                            (partial-sums (stream-cdr stream)))))

(defn eval-series-at [series x terms]
  (setv y 0)
  (setv remaining series)
  (for [i (range terms)]
    (+= y (* (stream-car remaining) (** x i)))
    (setv remaining (stream-cdr remaining)))
  y)

;; Exercise 3.59(a)
(defn integrate-series [df-stream]
  (let [[f-scale-stream (exponentiate-stream ℕ+ -1)]]
    (multiply-streams df-stream f-scale-stream)))

;; Exercise 3.59(b)
(def exp-series
  (cons-stream 1 (integrate-series exp-series)))

(def cos-series
  (cons-stream 1 (integrate-series (scale-stream sin-series -1))))

(def sin-series
  (cons-stream 0 (integrate-series cos-series)))

;; Exercise 3.60
;;
;; (a_0 + a_1*x + a_2*x^2 + ...)(b_0 + b_1*x + b_2*x^2 + ...)
;;  = a_0*b_0 + a_1*b_0*x + a_2*b_0*x^2 + ...
;;              a_1*b_1*x + a_1*b_1*x^2 + a_2*b_1*x^3 + ...
(defn multiply-series [a b]
  (cons-stream (* (stream-car a) (stream-car b))
               (add-streams (scale-stream (stream-cdr a) (stream-car b))
                            (multiply-series a (stream-cdr b)))))

(defn square-series [series]
  (multiply-series series series))

;; Exercise 3.61
(defn invert-unit-series [series]
  (cons-stream 1
               (multiply-series
                (scale-stream series -1)
                (invert-unit-series series))))

;; Exercise 3.62
;; `invert-unit-series` works on a series with a constant term of
;; unity, so I think the only slightly subtle part here is getting
;; this to work with divisors with arbitrary nonzero constant term
;;
;; XXX: this is most probably wrong, as we can infer from tan-series
;; being wrong
(defn divide-series [dividend divisor]
  (let [[denormalized-by (stream-car divisor)]
        [normalizer (/ 1 denormalized-by)]
        [normalized (scale-stream divisor normalizer)]]
    (multiply-series dividend
                     (scale-stream (invert-unit-series normalized)
                                   normalizer))))

;; XXX: this is wrong
;; => (eval-series-at tan-series 0.5 10)
;; 0.33318569176621754
;; => (tan 0.5)
;; 0.5463024898437905
(def tan-series (divide-series sin-series cos-series))

(defn sqrt-improve [guess x]
  (/ (+ guess (/ x guess)) 2))

(defn sqrt-stream [x]
  (def guesses
    (cons-stream 1
                 (stream-map (λ [guess]
                                (sqrt-improve guess x))
                             guesses)))
  guesses)

(defn π-summands [n]
  (cons-stream (/ 1 n)
               (stream-map (λ [t] (* -1 t)) (π-summands (+ n 2)))))

(def π-stream
  (scale-stream (partial-sums (π-summands 1)) 4))

(def pi-stream π-stream) ; easier to type

(defn euler-transform [stream]
  (let [[s0 (stream-nth stream 0)]
        [s1 (stream-nth stream 1)]
        [s2 (stream-nth stream 2)]]
    (cons-stream (- s2 (/ (** (- s2 s1) 2)
                          (+ s0 (* -2 s1) s2)))
                 (euler-transform (stream-cdr stream)))))

(defn make-tableau [transform stream]
  (cons-stream stream
               (make-tableau transform
                             (transform stream))))

(defn accelerando [transform stream]
  (stream-map stream-car (make-tableau transform stream)))

(defn zipstream [&rest streams]  ; I also think this is just one word
  (cons-stream (list (map stream-car streams))
               ;; XXX TypeError: Iteration on malformed cons
               (apply zipstream (list (stream-map stream-cdr streams)))))

(defn enumerate-stream [stream]
  (zipstream ℕ stream))

;; Exercise 3.64
(defn stream-limit [stream tolerance &optional [step 0]]
  (let [[next (stream-car stream)]
        [postnext (stream-car (stream-cdr stream))]]
    (if (< (abs (- postnext next)) tolerance)
      {:result postnext :steps step}
      (stream-limit (stream-cdr stream) tolerance (inc step)))))
