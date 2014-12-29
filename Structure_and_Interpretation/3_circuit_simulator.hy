(import [pairs [*]])
(import [queue [PriorityQueue]])
(import [3_circuit_make_wire [make_wire]])
(import time)

(require pairs)

(def inverter-delay 2)
(def and-gate-delay 3)
(def or-gate-delay 5)

(defn get-signal [wire]
  (wire 'get-signal))

(defn set-signal! [wire new-value]
  ((wire 'set-signal!) new-value))

(defn add-action! [wire action-procedure]
  ((wire 'add-action!) action-procedure))

(defn inverter [agenda input output]
 (defn invert-input []
    (let [[new-value (not (get-signal input))]]
      (after-delay agenda
                   inverter-delay
                   (fn []
                     (set-signal! output new-value)))))
  (add-action! input invert-input)
  'OK)

;; TODO: unify and-gate and or-gate (I think such a generalized
;; superfunction might actually need to be a macro because you might
;; not be able to pass "and" or "or" as arguments because they're not
;; actually functions??)

(defn and-gate [agenda a1 a2 output]
  (defn conjunct-inputs []
    (let [[new-value (and (get-signal a1) (get-signal a2))]]
      (after-delay agenda
                   and-gate-delay
                   (fn []
                     (set-signal! output new-value)))))
  (add-action! a1 conjunct-inputs)
  (add-action! a2 conjunct-inputs)
  'OK)

;; (This is Exercise 3.28 (too easy!).)
(defn or-gate [agenda o1 o2 output]
  (defn disjunct-inputs []
    (let [[new-value (or (get-signal o1) (get-signal o2))]]
      (after-delay agenda
                   and-gate-delay
                   (fn []
                     (set-signal! output new-value)))))
  (add-action! o1 disjunct-inputs)
  (add-action! o2 disjunct-inputs)
  'OK)

(defn half-adder [agenda a b s c]
  (let [[d (make-wire)]
        [e (make-wire)]]
    (or-gate agenda a d b)
    (and-gate agenda a b c)
    (inverter agenda c e)
    (and-gate agenda d e s)
    'OK))

(defn full-adder [agenda a b c-in sum c-out]
  (let [[s (make-wire)]
        [c1 (make-wire)]
        [c2 (make-wire)]]
    (half-adder agenda b c-in s c1)
    (half-adder agenda a s sum c2)
    (or-gate agenda c1 c2 c-out)
    'OK))

(defn make-agenda []
  (PriorityQueue))

(defn empty-agenda? [agenda]
  (.empty agenda))

(defn current-time [agenda]
  (let [[underlying-queue (. agenda queue)]]
    (if underlying-queue
      (. underlying-queue [0] [0])
      0)))

(defn peek-at-agenda [agenda]
  ;; you have to wonder why Python's queue.PriorityQueue doesn't just
  ;; expose this in the first place; thanks to
  ;; http://stackoverflow.com/a/9288155
  (let [[underlying-queue (. agenda queue)]]
    (if underlying-queue
      (. underlying-queue [0] [2])
      nil)))

(defn pop-from-agenda! [agenda]
  (get (apply agenda.get [] {"block" false}) 2))

(defn add-to-agenda! [agenda simulation-time action]
  (.put agenda (, simulation-time
                  ;; Python's PriorityQueue uses a heap, which expects
                  ;; things to be orderable, which doesn't work with
                  ;; functions ("TypeError: unorderable types:
                  ;; function() < function()"). Let's just use system
                  ;; time as a tiebreaker.
                  (time.time)
                  action)))

(defn after-delay [agenda delay action]
  (add-to-agenda! agenda
                  (+ delay (current-time agenda))
                  action))

(defn propogate [agenda]
  (if (.empty agenda)
    'done
    (do
     ((pop-from-agenda! agenda))
     (propogate agenda))))

(defn probe [name agenda wire]
  (add-action! wire
               (λ []
                  (print
                   (apply (. "{name} time={time} new_value={new-value}" format)
                          []
                          {"name" name "time" (current-time agenda)
                                  "new-value" (get-signal wire)})))))
