(defn construct-slice [start stop step]
  ;; workaround thanks to @olasd's Jan. 4 comment on
  ;; https://github.com/hylang/hy/issues/381
  ;;
  ;; (it looks like this was actually fixed upstream in
  ;; hylang/hy@89cdcc4a, but YOLO)
  ((get --builtins-- "slice") start stop step))

(defn lispt [elements]
  (if elements
    (cons (first elements) (lispt (rest elements)))
    nil))

(defn set-car! [pair value]
  (assoc pair 0 value))

(defn set-cdr! [pair value]
  (assoc pair (construct-slice 1 None None) value))
