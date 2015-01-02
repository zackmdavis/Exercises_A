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
