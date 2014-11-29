#lang racket

(define (front-ptr queue) (mcar queue))
(define (rear-ptr queue) (mcdr queue))
(define (set-front-ptr! queue item) (set-mcar! queue item))
(define (set-rear-ptr! queue item) (set-mcdr! queue item))

(define (empty-queue? queue) (null? (front-ptr queue)))
(define (make-queue) (mcons '() '()))

(define (front-queue queue)
  (when (not (empty-queue? queue))
    (mcar (front-ptr queue))))

(define (push-queue! queue item)
  (let ([new-pair (mcons item '())])
    (cond [(empty-queue? queue) (set-front-ptr! queue new-pair)
                                (set-rear-ptr! queue new-pair)
                                queue]
          [else (set-mcdr! (rear-ptr queue) new-pair)
                (set-rear-ptr! queue new-pair)
                queue])))

(define (pop-queue! queue)
  (when (not (empty-queue? queue))
      (set-front-ptr! queue (mcdr (front-ptr queue))))
  queue)

(provide (all-defined-out))

(require rackunit)

(check-true (empty-queue? (make-queue)))
(check-equal? (front-queue (push-queue! (make-queue) 3)) 3)

(define my-queue (push-queue! (push-queue! (make-queue) 3) 2))
(check-equal? (front-queue my-queue) 3)
(pop-queue! my-queue)
(check-equal? (front-queue my-queue) 2)
(pop-queue! my-queue)
(check-true (empty-queue? my-queue))
