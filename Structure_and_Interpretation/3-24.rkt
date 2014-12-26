#lang racket

(define (recorded key-equivalence? key records)
  ;; the textbook calls this `assoc`, but I don't like that name because it
  ;; means something different in Clojure
  (cond [(null? records) #f]
        [(key-equivalence? key (mcar (mcar records))) (mcar records)]
        [else (recorded key (mcdr records))]))

;; following the text ...
(define (make-table key-equivalence?)
  (let ([local-table (list '*table*)])
    (define (lookup k1 k2)
      (let ([subtable (recorded key-equivalence? k1 (mcdr local-table))])
        (if subtable
            (let ([record (recorded key-equivalence? k2 (mcar subtable))])
              (if record
                  (mcdr record)
                  #f))
        #f)))
    (define (insert! k1 k2 value)
      (let ([subtable (recorded key-equivalence? k1 (mcdr local-table))])
        (if subtable
            (let ([record (recorded key-equivalence? k2 (cdr subtable))])
              (if record
                  (set-mcdr! record value)
                  (set-mcdr! subtable
                             (mcons (mcons k2 value)
                                    (mcdr subtable)))))
            (set-mcdr! local-table
                       (cons (list k1
                                   (mcons k2 value))
                             (mcdr local-table)))))
      'ok)
    (define (dispatch m)
      (cond ([(equal? m 'lookup-procedure) lookup]
             [(equal? m 'insert!-procedure) insert!])))
    dispatch))

(provide (all-defined-out))

;; XXX: ??
;; > (require "3-24.rkt")
;; > (define ops-table (make-table equal?))
;; > (define get (ops-table 'lookup-procedure))
;; application: not a procedure;
;;  expected a procedure that can be applied to arguments
;;   given: #t
;;   arguments...:
;;    #<procedure:lookup>
