(import [3-circuit_simulator ;; the real filename is "3_circuit_simulator.hy"
         ;; until I figure out how to mix hyphens and underscores in
         ;; Hy module filenames (!)
         [*]])

(def the-agenda (make-agenda))
(def inverter-delay 2)
(def and-gate-delay 3)
(def or-gate-delay 5)

(def input-1 (make-wire))
(def input-2 (make-wire))
(def sum (make-wire))
(def carry (make-wire))

(probe 'sum sum)
(probe 'carry carry)

(half-adder input-1 input-2 sum carry)
(set-signal! input-1 true)
(propogate)
(set-signal! input-2 true)
(propogate)
