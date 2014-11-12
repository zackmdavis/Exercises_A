#lang racket

(define (tally items)
  (let ([empty-counter (make-hash (for/list ([item items]) (cons item 0)))])
    (for/fold ([counter empty-counter]) ([item items])
      (hash-update! counter item (lambda (x) (+ x 1)))
      counter)))

(define (total-tallied tallied)
  (for/fold ([the-total 0]) ([(_ tallies) tallied])
    (+ the-total tallies)))

(define (sorted-tallied tallied)
  (sort (for/fold ([container '()]) ([(key tallies) tallied])
          (cons (list key tallies) container))
        (lambda (x y) (apply > (map second (list x y))))))

(define (argmajority turnout standings)
  (if (> (/ (cdr (first standings)) turnout) 0.5)
      (car (first standings))
      #f))

(define (election candidates ballots)
  (let* ([picks (map first ballots)]
         [totals (tally picks)]
         [turnout (total-tallied totals)]
         [standings (sorted-tallied totals)]
         [winner (argmajority turnout standings)])

    ;; TODO

    ;; DEBUG
    (for ([value (list picks totals turnout standings winner)])
      (displayln value))
  '()))


(require rackunit)

;;; PREREQUISITES FOR DEMOCRACY

;; we can count
(check-equal? (tally '("a" "b" "b"))
              (hash-copy #hash(("a" . 1) ("b" . 2))))
(check-equal? (total-tallied (tally '("|" "|" "|" "|")))
                             4)

;; we can tell which numbers are bigger
(check-equal? (sorted-tallied #hash(("Jones" . 1) ("Kevin" . 11)
                                    ("America" .  9) ("Jennifer" . 14)))
              '(("Jennifer" 14) ("Kevin" 11)
                ("America"  9) ("Jones" 1)))

;; we can distinguish a majority
(check-equal? (argmajority 3 (list (cons "red" 2) (cons "blue" 1)))
              "red")

;; TODO
(check-equal? (election '("John Doe" "Jane Smith" "Jane Austen")
                        '((1 2 3) (2 1 3) (2 3 1) (1 2 3) (3 1 2)))
              "John Doe")
