#lang racket

(define (f x)
  (define f
    (lambda (y) 0))
  x)

(require rackunit)

;; okay, we were asked for a function that gave different results
;; depending on order of evaluation, and I thought the trick would be
;; "redefine the symbol name while evaluating the first argument" but
;; empirically this doesn't actually work
(check-equal? (+ (f 0) (f 1)) 0) ;; XXX Check failure
(check-equal? (+ (f 1) (f 0)) 1)
