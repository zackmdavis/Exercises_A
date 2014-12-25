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
      (let ([subtable (recorded key-equivalence? k1 (mcdr table))])
        (if subtable
            (let ([record (recorded key-equivalence? k2 (mcar subtable))])
              (if record
                  (mcdr record))
                  #f)))
      ;; TODO continue
      )))
