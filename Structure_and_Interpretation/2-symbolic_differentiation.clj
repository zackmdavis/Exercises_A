(defn variable? [expression]
  (symbol? expression))

(defn same-variable? [v1 v2]
  (and (symbol? v1)
       (symbol? v2)
       (= v1 v2)))

(defn sum? [expression]
  (and (list? expression)
       (= (first expression) :+)))

(defn summand1 [expression]
  (first (rest expression)))

(defn summand2 [expression]
  (rest (rest expression)))

(defn make-sum [summand1 summand2]
  (list :+ summand1 summand2))

(defn product?  [expression]
  (and (list? expression)
       (= (first expression) :*)))

(defn factor1 [expression]
  (first (rest expression)))

(defn factor2 [expression]
  (rest (rest expression)))

(defn make-product [factor1 factor2]
  (list :* factor1 factor2))

(defn derivative [expression variable]
  (cond
   (number? expression) 0
   (variable? expression) (if (same-variable? expression variable)
                            1
                            0)
   (sum? expression) (make-sum (derivative (summand1 expression))
                               (derivative (summand2 expression)))
   (product? expression) (make-sum
                          (make-product (factor1 expression)
                                        (derivative (factor2 expression)))
                          (make-product (derivative (factor1 expression))
                                        (factor2 expression)))
   :else "ERROR"))

(def my-expression '(:+ x 3))
(def my-variable 'x)
(println (derivative my-expression my-variable))

;; "Exception in thread "main" clojure.lang.ArityException: Wrong
;; number of args (1) passed to: user$derivative"? wtf?!
;;
;; perhaps I'm not having a good day