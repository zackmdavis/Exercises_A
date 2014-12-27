;; XXX: adorable
(defmacro λ [&rest code]
  `(lambda ~@code))

(defn lispt [elements]
  (if elements
    (cons (first elements) (lispt (rest elements)))
    nil))

(defn construct-slice [start stop step]
  ;; workaround thanks to @olasd's Jan. 4 comment on
  ;; https://github.com/hylang/hy/issues/381
  ;;
  ;; (this might get fixed in Hy 0.11
  ;; (https://github.com/hylang/hy/pull/652#issuecomment-63171260))
  ((get --builtins-- "slice") start stop step))

(defn set-car! [pair value]
  (assoc pair 0 value))

(defn set-cdr! [pair value]
  (assoc pair (construct-slice 1 None None) value))
