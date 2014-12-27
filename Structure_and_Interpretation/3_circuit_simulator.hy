(import [queue [PriorityQueue]])

(defn make-wire []
  (let [[signal-value 0]
        [action-procedures []]]

    (defn set-signal! [new-value]
      (if (!= signal-value new-value)
        (setv signal-value new-value)
        'done))
    ;; XXX WACK: is it unreasonable to expect `set-signal!` to see
    ;; locals defined in the enclosing function?
    ;;   File
    ;;   "/home/zmd/Code/Textbook_Exercises_A/Structure_and_Interpretation/
    ;; 3_circuit_simulator.hy",
    ;;   line 8, in set_signal!
    ;;     (if (!= signal-value new-value)
    ;; UnboundLocalError: local variable 'signal_value' referenced
    ;; before assignment

    (defn accept-action-procedure! [procedure]
      (.append action-procedures procedure))

    (defn dispatch [m]
      (cond [(= m 'get-signal) signal-value]
            [(= m 'set-signal!) set-signal!]
            [(= m 'add-action!) accept-action-procedure!]
            [true (raise (ValueError "unknown operation"))]))
    dispatch))

(defn call-each [procedures]
  (for [procedure procedures]
    (procedure))
  'done)

(defn get-signal [wire]
  (wire 'get-signal))

(defn set-signal! [wire new-value]
  ((wire 'set-signal!) new-value))

(defn add-action! [wire action-procedure]
  ((wire 'add-action!) action-procedure))

(defn inverter [input output]
 (defn invert-input []
    (let [[new-value (not (get-signal input))]]
      (after-delay inverter-delay
                   (fn []
                     (set-signal! output new-value)))))
  (add-action! input invert-input)
  'OK)

;; TODO: unify and-gate and or-gate (I think such a generalized
;; superfunction might actually need to be a macro because you might
;; not be able to pass "and" or "or" as arguments because they're not
;; actually functions??)

(defn and-gate [a1 a2 output]
  (defn conjunct-inputs []
    (let [[new-value (and (get-signal a1) (get-signal a2))]]
      (after-delay and-gate-delay
                   (fn []
                     (set-signal! output new-value)))))
  (add-action! a1 conjunct-inputs)
  (add-action! a2 conjunct-inputs)
  'OK)

;; (This is Exercise 3.28 (too easy!).)
(defn or-gate [o1 o2 output]
  (defn disjunct-inputs []
    (let [[new-value (or (get-signal o1) (get-signal o2))]]
      (after-delay and-gate-delay
                   (fn []
                     (set-signal! output new-value)))))
  (add-action! o1 disjunct-inputs)
  (add-action! o2 disjunct-inputs)
  'OK)

(defn half-adder [a b s c]
  (let [[d (make-wire)]
        [e (make-wire)]]
    (or-gate a d b)
    (and-gate a b c)
    (inverter c e)
    (and-gate d e s)
    'OK))

(defn full-adder [a b c-in sum c-out]
  (let [[s (make-wire)]
        [c1 (make-wire)]
        [c2 (make-wire)]]
    (half-adder b c-in s c1)
    (half-adder a s sum c2)
    (or-gate c1 c2 c-out)
    'OK))

(defn make-agenda []
  (PriorityQueue))

(defn empty-agenda? [agenda]
  (.empty agenda))

(defn peek-at-agenda [agenda]
  ;; you have to wonder why Python's queue.PriorityQueue doesn't just
  ;; expose this in the first place; thanks to
  ;; http://stackoverflow.com/a/9288155
  (. agenda queue [0] [1]))

(defn pop-from-agenda! [agenda]
  (second (apply agenda.get [] {"block" false})))

(defn add-to-agenda! [agenda time action]
  (.put agenda (, time action)))

(defn current-time [agenda]
  ;; without loss of generality, we can suppose that the "current"
  ;; time is that of the next event (I think), because time without
  ;; events is irrelevant to the simulation
  (first (peek-at-agenda ag)))

(defn after-delay [delay action]
  ;; we seem to be assuming that `the-agenda` will already be in the
  ;; envrionment
  (add-to-agenda! (+ delay (current-time the-agenda))
                  action
                  the-agenda))

(defn propogate []
  (if (.empty the-agenda)
    'done
    (do
     ((pop-from-agenda! agenda))
     (propogate))))

(defn probe [name wire]
  (add-action! wire
               (lambda []
                 (print (.format "{name} time={time} new_value={new-value}"
                                 {"name" name "time" time
                                         "new-value" new-value})))))
