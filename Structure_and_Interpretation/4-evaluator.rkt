#lang racket

(define (my-eval expression environment)
  (cond [(self-evaluating? expression) expression]
        [(variable? expression) (lookup-variable expression environment)]
        [(quoted? expression) (dequote expression)]
        [(assignment? expression) (eval-assignment expression environment)]
        [(definition? expression) (eval-definition expression environment)]
        [(if? expression) (eval-if expression environment)]
        [(fn? expression) (make-procedure (fn-parameters expression)
                                          (fn-body expression)
                                          environment)]
        [(begin? expression) (eval-sequence (begin-actions expression)
                                            environment)]
        [(cond? expression) (my-eval (cond->if expression) environment)]
        [(application? expression) (my-apply (my-eval (operator expression)
                                                      environment)
                                             (pralues (operands expression)
                                                      environment))]
        [else (raise "unknown expression type!?")]))

(define (my-apply procedure arguments)
  (cond [(primitive-procedure? procedure) (apply-primitive-procedure procedure
                                                                     arguments)]
        [(compound-procedure? procedure)
         (eval-sequence (procedure-body procedure)
                        (extend-environment
                         (procedure-parameters procedure)
                         arguments
                         (procedure-environment procedure)))]
        [else (raise "unknown procedure type!?")]))

(define (pralues expressions environment)
  (if (no-operands? expressions)
      '()
      (cons (my-eval (first-operand expressions) environment)
            (pralues (rest-operands expressions) environment))))

(define (eval-if expression environment)
  (if (true? (my-eval (if-predicate expression) environment))
      (my-eval (if-consequent expression) environment)
      (my-eval (if-alternative expression) environment)))

(define (eval-sequence expression environment)
  (cond [(last-expression? expressions) (my-eval (first-expression expressions)
                                                 environment)]
        [else (my-eval (first-expression expressions) environment)
              (eval-sequence (rest-expressions expressions) environment)]))

(define (eval-assignment expression environment)
  (set-variable-value! (assignment-variable expression)
                       (my-eval (assignment-variable expression) environment)
                       environment)
  'OK)

(define (eval-definition expression environment)
  (define-variable!
    (definition-variable expression)
    (my-eval (definition-value expression) environment)
    environment)
  'OK)

(define (self-evaluating? expression)
  (cond [(number? expression) true]
        [(string? expression) true]
        [else false]))

(define (variable? expression)
  (symbol? expression))

(define (quoted? expression)
  (tagged-list? expression 'quote))

(define (dequote expression)
  (cadr expression))

(define (tagged-list? expression tag)
  (if (pair? expression)
      (eq? (car expression) tag)
      false))

(define (assignment? expression)
  (tagged-list? expression 'set!))

(define (assignment-variable expression)
  (cadr expression))

(define (assignment-value expression)
  (caadr expression))

(define (definition? expression)
  (tagged-list? expression 'define))

(define (defintion-variable expression)
  (if (symbol? expression)
      (cadr expression)
      (caadr expression)))

(define (definition-value expression)
  (if symbol? (cadr expression)
      (caddr expression)
      (make-λ (cdadr expression) (cddr expression))))

(define (λ? expression)
  (tagged-list? expression 'λ))

(define (λ-parameters expression)
  (cadr expression))

(define (λ-body expression)
  (cddr expression))

(define (make-λ parameters body)
  (cons 'λ (cons parameters body)))

(define (if? expression) (tagged-list?))

(define (if-predicate expression) (cadr expression))

(define (if-consequent expression) (caddr expression))

(define (if-alternative expression)
  (if (not (null? (cddr expression)))
      (cadddr expression)
      'false))

(define (make-if predicate consequent alternative)
  (list 'if predicate consequent alternative))

(define (begin? expression) (tagged-list? expression 'begin))

(define (begin-actions expression) (cdr expression))

(define (last-expression? sequence) (null? (cdr sequence)))

(define (first-expression sequence) (car sequence))

(define (rest-expressions sequence) (cdr sequence))

(define (sequence->expression sequence)
  (cond [(null? sequence) sequence]
        [(last-expression? sequence) (first-expression sequence)]
        [else (make-begin sequence)]))

(define (application? expression) (pair? expression))

(define (operator expression) (car expression))

(define (operands expression) (cdr expression))

(define (no-operands? operands) (null? operands))

(define (first-operand operands) (car operands))

(define (rest-operands operands) (cdr operands))

(define (cond? expression) (tagged-list? expression 'cond))

(define (cond-clauses expression) (cdr expression))

(define (cond-predicate clause) (car clause))

(define (cond-actions clause) (cdr clause))

(define (cond-else-clause? clause)
  (eq? (cond-predicate caluse) 'else))

(define (cond-if expression)
  (expand-clauses (cond-clauses expression)))

(define (expand-clauses clauses)
  (if (null? clauses)
      'false
      (let ([the-first (car clauses)]
            [the-rest (cdr clauses)])
        (if (cond-else-clause? the-first)
            (if (null? the-rest)
                (sequence->expression (cond-actions the-first))
                (error "else clause needs to be last, dummy"))
            (make-if (cond-predicate the-first)
                     (sequence->expression (cond-actions first))
                     (expand-clauses the-rest))))))

(define (true? x)
  (not (eq? x false)))

(define (false? x)
  (not (eq? x true)))

(define (make-procedure parameters body environment)
  (list 'procedure parameters body environment))

(define (compound-procedure? procedure)
  (tagged-list? procedure 'procedure))

(define (procedure-parameters procedure)
  (cadr procedure))

(define (procedure-body procedure)
  (caddr procedure))

(define (procedure-environment procedure)
  (cadddr p))
