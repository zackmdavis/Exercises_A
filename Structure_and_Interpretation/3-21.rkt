#lang racket

(require "3-queues.rkt")

(define (print-queue queue)
  (displayln (mcar queue)))

(print-queue (push-queue! (push-queue! (push-queue! (make-queue) 1) 2) 3))
