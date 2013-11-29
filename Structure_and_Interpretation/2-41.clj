; TODO; understanding in progress

;(defn distinct-triple-sum [n s]

(map (fn [i]
       ; map to (i j) for j in range(i+1,n)
       (range n)

   (fn [j]
     ; map to (i j k) for k in range(j+1, n)
     (

  ;; (map (fn [i]
  ;;        (map (fn [j]
  ;;               (map (fn [k]
  ;;                      (list i j k))
  ;;                    (range j n))
  ;;               (range i n))
  ;;               (list i j))
  ;;             (range i n)))
  ;;      (range n))