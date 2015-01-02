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

;; Exercise 3.55
(defn partial-sums [stream]
  ;; XXX wrong these are squares; you need to understand the problem
  ;; and not just guess
  (cons-stream (stream-car stream)
               (add-streams stream (partial-sums (stream-cdr stream)))))

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
;;; ummmm, this is the "Cauchy product", right? but the template the
;;; book suggests we complete doesn't seem to fit that
(defn multiply-series [])
