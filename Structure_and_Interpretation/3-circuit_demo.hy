(import [3-circuit_simulator ;; the real filename is "3_circuit_simulator.hy"
         ;; until I figure out how to mix hyphens and underscores in
         ;; Hy module filenames (which I don't think is actually possible)
         [*]])

(def the-agenda (make-agenda))

(def input-1 (make-wire))
(def input-2 (make-wire))
(def sum (make-wire))
(def carry (make-wire))

(probe 'sum the-agenda sum)
(probe 'carry the-agenda carry)

(half-adder the-agenda input-1 input-2 sum carry)
(set-signal! input-1 true)
(propogate the-agenda)
(set-signal! input-2 true)
(propogate the-agenda)

;; XXX BROKEN: why isn't the simulation taking time into account!?!
;; sum time=0 new_value=False
;; carry time=0 new_value=False
;; carry time=0 new_value=True
