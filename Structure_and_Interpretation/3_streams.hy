(import [functools [lru_cache :as memoize]])

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

(defn stream-filter [predicate stream]
  ;; TODO
)

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

(defn integers-from [n]
  (cons-stream n (integers-from (inc n))))

(def ℕ (integers-from 0))
(def ℕ+ (integers-from 1))
