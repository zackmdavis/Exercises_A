
;; what happens if we use this section of the book as an excuse to
;; simultaneously learn more about how to use threads in Python??

(import [threading [Thread]])
(import [time [sleep]])
(import [random [random]])

(import [threading [Lock :as mutex]])

(require pairs)

(defn parallel-execute [&rest procedures]
  (for [[i procedure] (enumerate procedures)]
    (let [[new-thread (apply Thread []
                             {"name" (.format "Thread for Procedure #{}" i)
                              "target" procedure})]]
      (.start new-thread))))

(defn make-serializer []
  (let [[our-mutex (mutex)]]
    (λ [procedure]
       (λ [&rest args]
          (.acquire our-mutex)
          (let [[value (apply procedure args)]]
            (.release our-mutex)
            value)))))

;; Exercise 3.47(a) asks us to implement a semaphore in terms of
;; mutexes. You might naively think you can just keep a list of
;; mutexes: when you get an acquisition request, go through the list
;; and offer the first avalable lock, or (b)lock if none are
;; available. But then what happens when one of the mutices gets
;; released just after you've checked it? Maybe we could protect our
;; list of locks with ... another lock?
(defclass MySemaphore []
  [[__init__
    (λ [self n]
       (def self.master-lock (mutex))
       (def self.locks (list-comp (mutex) [_ (range n)])))]

   ;; XXX TODO FIXME: test if this actually works, fix it since it
   ;; won't work because nothing ever works

   [acquire
    (λ [self]
       (.acquire self.master-lock)
       (setv acquired-inner-lock false)
       (setv i 0)
       (while (not acquired-inner-lock)
         (let [lock (get (self.locks) i)]
           (setv acquired-inner-lock (apply lock.acquire
                                            [] {"blocking" false})))
         (setv i (% (inc i))))
       (.release self.master-lock))]

   [release
    (λ [self]
       ;; XXX boilerplate
       (.acquire self.master-lock)
       (setv released-inner-lock false)
       (setv i 0)
       (while true
         (let [lock (get (self.locks) i)]
           ;; XXX this is awful
           (try
              (lock.release)
              (setv i (% (inc i)))
              (break)
           (catch [e ThreadError]))))
       (.release self.master-lock))]])
