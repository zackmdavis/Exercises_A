(defn cons [a b] [a b])
(defn pair? [thing] (and (isinstance thing list) (= (len thing) 2)))

(defn car [pair] (first thing))
(defn cdr [pair] (second thing))

(defn lispt [elements]
  "Given a Python-list, return the corresponding Lisp-list (i.e., made from
cons cells, which actuallly happen to actually be Python-lists in this
implementation)."
  (if elements
    (cons (first elements) (lispt (rest elements)))
    nil))

(defn set-car! [pair value]
  (assoc pair 0 value))

(defn set-cdr! [pair value]
  (assoc pair 1 value))

(defn nil? [thing] (is thing nil))
