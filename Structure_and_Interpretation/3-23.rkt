;; Let's build a deck (I mean, deque)!

;; When building lists out of pairs, one member of a pair would store
;; the element, and another would be the link to the rest of the list,
;; but if we want a deque with constant-time operations, I can't see any
;; way around needing to base it on a _doubly_-linked list, but there's
;; not enough room in a pair to store two links and an element, so we
;; need to use two pairs per node.

#lang racket

(define (make-node element)
  (mcons element
         (mcons null null)))

(define (make-deque)
  (mcons null null))

(define (empty-deque? deque)
  (null? (mcar deque)))

(define (other end)
  (cond [(equal? end "front") "back"]
        [(equal? end "back") "front"]))

(define (end-to-get-operation end)
  (cond [(equal? end "front") mcar]
        [(equal? end "back") mcdr]))

(define (end-to-set-operation end)
  (cond [(equal? end "front") set-mcar!]
        [(equal? end "back") set-mcdr!]))

(define (peek end deque)
  (when (not (empty-deque? deque))
    (mcar ((end-to-get-operation end) deque))))

(define (peek-front deque)
  (peek "front" deque))

(define (peek-back deque)
  (peek "back" deque))

(define (push! end deque item)
  (let ([new-node (make-node item)])
    (cond [(empty-deque? deque) (for ([set-bang '(set-mcar! set-mcdr!)])
                                  ((eval set-bang) deque new-node))]
          [else (let ([old-tail ((end-to-get-operation end) deque)])
                  ((end-to-set-operation (other end)) (mcdr new-node) old-tail)
                  ((end-to-set-operation end) (mcdr old-tail) new-node)
                  ((end-to-set-operation end) deque new-node))])))

(define (push-front! deque item)
  (push! "front" deque item))

(define (push-back! deque item)
  (push! "back" deque item))

(define (pop! end deque)
  (when (not (empty-deque? deque))
    (let* ([node-to-pop ((end-to-get-operation end) deque)]
           [new-tail ((end-to-get-operation (other end)) (mcdr node-to-pop))])
      ((end-to-set-operation end) deque new-tail)
      (cond [(null? new-tail) (for ([set-bang '(set-mcar! set-mcdr!)])
                                ((eval set-bang) deque null))]
            [else ((end-to-set-operation end) (mcdr new-tail) null)])
      (mcar node-to-pop))))

(define (pop-front! deque)
  (pop! "front" deque))

(define (pop-back! deque)
  (pop! "back" deque))

(define (forward-traverse from-node callback)
  (when (not (null? from-node))
    (callback from-node)
    (forward-traverse (mcdr (mcdr from-node)) callback)))

(define (variadic-display . displayables)
  (for ([displayable displayables])
    (display displayable)))

(define (display-deque deque)
  (display "#deque( ")
  (forward-traverse (mcar deque) (λ (node) (variadic-display (mcar node) " ")))
  (display ")\n"))

(provide (all-defined-out))


(require rackunit)

(test-case
 "deque operations behave like we expect"
 (define my-new-deque (make-deque))

 (for ([i (range 5)])
   (push-front! my-new-deque i)
   (display-deque my-new-deque)) ; 4 3 2 1 0

 (check-equal? (peek-front my-new-deque) 4)
 (check-equal? (peek-back my-new-deque) 0)

 (for ([i (range 3)])
   (check-equal? (pop-back! my-new-deque) i)
   (display-deque my-new-deque)) ; 4 3

 (for ([i '(4 3)])
   (check-equal? (pop-front! my-new-deque) i)
   (display-deque my-new-deque)) ; - the void -

 (check-pred empty-deque? my-new-deque))

;;;; success!!
;; > (require "3-23.rkt")
;; #deque( 0 )
;; #deque( 1 0 )
;; #deque( 2 1 0 )
;; #deque( 3 2 1 0 )
;; #deque( 4 3 2 1 0 )
;; #deque( 4 3 2 1 )
;; #deque( 4 3 2 )
;; #deque( 4 3 )
;; #deque( 3 )
;; #deque( )
