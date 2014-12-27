(import [pairs [*]])

(defn find-record [records key]
  (cond [(empty? records) nil]
        [(= key (car (car records))) (cdr (car records))]
        [true (find-record (cdr records) key)]))

(defn lookup [table keys]
  (if (empty? (cdr keys))
    (find-record (cdr table) (car keys))
    (lookup (find-record (cdr table) (car keys)) (cdr keys))))

(defn insert! [table keys value]
  (let [[record (find-record (cdr table) (car keys))]]
    (if (empty? (cdr keys))
      (if record
        (set-cdr! record value)
        (set-cdr! table (cons (cons (car keys) value)
                              (cdr table))))
      (insert! record (cdr keys) value))))

(defn make-multitable []
  (lispt ['multitable]))


(when (= --name-- "__main__")
  ;; I remember reading that inline unittest doesn't actually work
  ;; (https://github.com/hylang/hy/issues/594), so let's just use
  ;; `assert`
  (def my-multitable (make-multitable))

  (print "we can insert a value at depth one")
  (insert! my-multitable ["rah"] 2)
  (def retrieved (lookup my-multitable ["rah"]))
  (assert (= retrieved 2))
  (print "check\n")

  ;; we _should_ be able to insert a value at depth two ...
  ; (insert! my-multitable ["hello" "American"] "robots")
  ; (def retrieved (lookup my-multitable ["hello" "American"]))
  ; (assert (= retrieved "robots"))
  ;; XXX TODO FIXME: and yet we can't because we're stupid
  ;;
  ;; File "3-25.hy", line 14, in _hy_anon_fn_3
  ;;   (let [[record (find-record (cdr table) (car keys))]]
  ;; TypeError: 'NoneType' object is not subscriptable
)
