(import [collections [Counter]])

(import [3_serializers [*]])

(import [pairs [*]])
(require pairs)

;; (defn short-random-delay-or-not []
;;   (let [[alea-iacta-est (random)]]
;;     (when (< alea-iacta-est .5)
;;       (sleep alea-iacta-est))))

;; (def x 10)
;; (parallel-execute (λ [] (global x)
;;                      (short-random-delay-or-not) (setv x (* x x)))
;;                   (λ [] (global x)
;;                      (short-random-delay-or-not) (setv x (+ x 1))))

;; (sleep 0.5)
;; (print x)  ; empirically, prints 101 or 121, depending on how the dice fell

;; but the text is saying that one of _five_ values are possible and
;; that you need serialization just to get down to two. So maybe the
;; interpreter is smart and fast enough such that just one short
;; random delay in each procedure isn't enough, and I need to
;; introduce even more random read/write lag to easily observe the
;; madness of shared state---

(defn short-random-delay-or-not []
  (let [[alea-iacta-est (random)]]
    (when (< alea-iacta-est .5)
      (sleep (/ alea-iacta-est 100)))))

(defn concurrency-experiment [trials]
  (def results [])

  (for [_ (range trials)]
    (set-global! "x" 10)
    (parallel-execute (λ []
                         (short-random-delay-or-not)
                         (setv read-x-once (get-global "x"))
                         (short-random-delay-or-not)
                         (setv read-x-again (get-global "x"))
                         (set-global! "x" (* read-x-once read-x-again)))
                      (λ []
                         (short-random-delay-or-not)
                         (setv read-x-once (get-global "x"))
                         (set-global! "x" (+ read-x-once 1))))

    (sleep 0.03)
    (.append results (get-global "x")))
  (Counter results))

;; (print (concurrency-experiment 50))

;; well, you get the idea, I guess, even if this doesn't reproduce all
;; possible interleavings of events---

;; zmd@ExpectedReturn:~/Code/Textbook_Exercises_A/Structure_and_Interpretation$
;; hy 3_concurrency_demo.hy
;; Counter({101: 22, 121: 18, 110: 10})


(defn serialized-concurrency-experiment [trials]
  ;; same as above, but we're wrapping each of the procedures with a
  ;; serializer `s`
  (def results [])

  (for [_ (range trials)]
    (set-global! "x" 10)
    (def s (make-serializer))
    (parallel-execute (s (λ []
                            (short-random-delay-or-not)
                            (setv read-x-once (get-global "x"))
                            (short-random-delay-or-not)
                            (setv read-x-again (get-global "x"))
                            (set-global! "x" (* read-x-once read-x-again))))
                      (s (λ []
                            (short-random-delay-or-not)
                            (setv read-x-once (get-global "x"))
                            (set-global! "x" (+ read-x-once 1)))))

    (sleep 0.03)
    (.append results (get-global "x")))
  (Counter results))

(print (concurrency-experiment 50))
(print (serialized-concurrency-experiment 50))

;; zmd@ExpectedReturn:~/Code/Textbook_Exercises_A/Structure_and_Interpretation$
;; hy 3_concurrency_demo.hy
;; Counter({101: 20, 121: 15, 110: 15})
;; Counter({101: 50})
