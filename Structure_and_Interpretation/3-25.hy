(import [pairs [*]])

(defn find-record [records key]
  (cond [(nil? records) false]
        [(= key (car (car records))) (car records)]
        [true (find-record (cdr records) key)]))

(defn lookup [table keys]
  (if (nil? (cdr keys))
    (find-record (cdr table) (car keys))
    (lookup (find-record (cdr table) (car keys)) (cdr keys))))

(defn insert! [table keys value]
  (let [[record (find-record (cdr table) (car keys))]]
    (if (nil? (cdr keys))
      (if record
        (set-cdr! record value)
        (set-cdr! table (cons (cons key value)
                              (cdr table))))
      (insert! record (cdr keys) value))))


(defn make-multitable []
  (lispt ['multitable]))


(def my-multitable (make-multitable))

(insert! my-multitable "rah" 2)
(print my-multitable)
