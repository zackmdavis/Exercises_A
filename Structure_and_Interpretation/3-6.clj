;; TODO

; 2^32 = 4294967296

(defn my-linear-congruential-generator [a X c m]
  (mod (+ (* a X) c) m))

;; (def my-rand
;;   (let state (atom 