
(define (front-ptr queue) (car queue))
(define (rear-ptr queue) (cdr queue))
(define (set-front-ptr! queue item) (set-mcar! queue item))
(define (set-rear-ptr! queue item) (set-mcdr! queue item))

(define (empty-queue? queue) (null? (front-ptr queue)))
(define (make-queue) (mcons '() '()))

;; TODO

(require rackunit)
