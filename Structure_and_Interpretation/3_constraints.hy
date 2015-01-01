(import [itertools [dropwhile]])

(require pairs)

;; This doesn't really need to be a macro, but (pick an excuse)
;; practice/YOLO
(defmacro run-for-each-except [excluded iterable procedure]
   (with-gensyms [i]
     `(for [i ~iterable]
        (if (!= i ~excluded)
          (~procedure i)))))

(defclass Connector []
  [[value nil]
   [informant nil]
   [constraints []]

   [has-value? (λ [self] (bool self.informant))]

   [set! (λ [self setter new-value]
            (cond [(not (self.has-value?))
                   (do (setv value new-value)
                       (setv informant setter)
                       (print "DEBUG" self.constraints)
                       (run-for-each-except setter
                                            self.constraints
                                            .process-new-value))]
                  [(!= new-value value)
                   (raise (ValueError "contradiction!" value new-value))]))]
   [forget! (λ [self retractor]
               (when (= retractor informant)
                 (setv self.informant false)
                 (run-for-each-except retractor
                                      constraints
                                      .process-forget-value)))]
   [connect! (λ [self new-constraint]
                (if (not (list (dropwhile (λ [c] (!= new-constraint c))
                                          self.constraints)))
                  (.append self.constraints new-constraint))
                (if (self.has-value?)
                  (.process-new-value new-constraint))
                'done)]])

(defclass Adder []
  [[__init__ (λ [self first-summand second-summand the-sum]
                (def self.first-summand first-summand)
                (def self.second-summand second-summand)
                (def self.the-sum the-sum)
                (for [connector [self.first-summand self.second-summand
                                 self.the-sum]]
                  (.connect! connector self)))]
   [process-new-value (λ [self]
                         (cond [(and (.has-value? self.first-summand)
                                     (.has-value? self.second-summand))
                                (.set! self.the-sum
                                       self
                                       (+ (.value self.first-summand)
                                          (.value self.second-summand)))]
                               [(and (.has-value? self.first-summand)
                                     (.has-value? self.the-sum))
                                (.set! self.second-summand
                                       self
                                       (- (.value self.the-sum)
                                          (.value self.first-summand)))]
                               [(and (.has-value? self.second-summand)
                                     (.has-value? self.the-sum))
                                (.set! self.first-summand
                                       self
                                       (- (.value self.the-sum)
                                          (.value self.second-summand)))]))]
   [process-forget-value (λ [self]
                            (.forget! self.the-sum self)
                            (.forget! self.first-summand self)
                            (.forget! self.second-summand self)
                            (process-new-value))]])

(defclass Constant []
  [[__init__ (λ [self value connector]
                (def self.value value)
                (def self.connector connector)
                (.connect! connector self)
                (.set! connector self value))]

   ;;; fffffff "ValueError: constants are not mutable: 2 != None"
   ;; [process-new-value (λ [self]
   ;;                       (when (!= self.value self.connector.value)
   ;;                         (raise (ValueError
   ;;                                 (.format
   ;;                                  "constants are not mutable: {} != {}"
   ;;                                  self.value self.connector.value)))))]])
   ;;
   ;;; what happens if we don't try to catch errors??
   ;;;; it is a horrible tragedy of the human condition that writing
   ;;;; new code with new ideas is more fun than debugging the old
   ;;;; stuff that never quite acutally worked, and one of these days
   ;;;; it is going to kill me dead
   [process-new-value (λ [self])]])


(defclass Probe []
  [[__init__ (λ [self name connector]
                (def self.name name)
                (def self.connector connector)
                (.connect! connector self)
                nil)] ; the Python interpreter requires that __init__
                      ; returns None (something that had never come up
                      ; for me while writing _Python_)

   [process-new-value (λ [self]
               (print (.format "{}={}" self.name self.connector.value)))]

   [process-forget-value (λ [self] (print (.format "{}=??" self.name)))]])
