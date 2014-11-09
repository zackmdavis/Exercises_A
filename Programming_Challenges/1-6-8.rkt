#lang racket

(define (tally items)
  (let ([empty-counter (make-hash (for/list ([item items]) (cons item 0)))])
    (for/fold ([counter empty-counter]) ([item items])
      (hash-update! counter item (lambda (x) (+ x 1)))
      counter)))


(define (election candidates votes)
  ;; TODO
  '())


(require rackunit)

(check-equal? (tally '("a" "b" "b"))
              (hash-copy #hash(("a" . 1) ("b" . 2))))

;; TODO
(check-equal? (election '("John Doe" "Jane Smith" "Jane Austen")
                        '((1 2 3) (2 1 3) (2 3 1) (1 2 3) (3 1 2)))
              "John Doe")
