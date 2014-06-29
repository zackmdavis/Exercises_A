(ns interpreter
  (:require [clojure.test :refer :all]))

(defn operation_mod_n [operation n]
  (fn [& args]
    (mod (apply operation args) n)))

(def m+ (operation_mod_n + 1000))
(def m* (operation_mod_n * 1000))

(defn clean_machine []
  (atom {:memory (vec (for [_ (range 1000)] 0))
         :registers (vec (for [_ (range 10)] 0))
         :current 0
         :clock 0
         :halted false}))

(defn initialize_memory! [machine content]
  (swap! machine
         (fn [state]
           (assoc state :memory
                  (vec (concat content
                               (drop (count content) (state :memory))))))))

(defn lookup_register [state r]
  ((state :registers) r))

(defn lookup_memory [state j]
  ((state :memory) j))

;; XXX: this probably still isn't general enough (I say "swap!
;; machine" elsewhere, too)
(defn mutate_state! [machine component k operation]
  (swap! machine
         (fn [state]
           (assoc state
             component (assoc (state component)
                          k (operation ((state component) k)))))))

(defn mutate_register! [machine r operation]
  (mutate_state! machine :registers r operation))

(defn mutate_memory! [machine j operation]
  (mutate_state! machine :memory j operation))

(defn set_register! [machine r n]
  (mutate_register! machine r (fn [_] n)))

(defn add_to_register! [machine r n]
  (mutate_register! machine r (fn [x] (m+ x n))))

(defn multiply_to_register! [machine r n]
  (mutate_register! machine r (fn [x] (m* x n))))

(defn set_register_to_register! [machine destination source]
  (mutate_register! machine destination (lookup_register machine source)))

(defn add_register_to_register! [machine destination source]
  (add_to_register! machine destination (lookup_register machine source)))

(defn multiply_register_to_register! [machine destination source]
  (multiply_to_register! machine destination (lookup_register machine source)))

(defn set_register_from_address! [machine r pointer]
  (set_register! machine r (lookup_memory machine
                                      (lookup_register machine pointer))))

(defn set_memory_from_register! [machine pointer source]
  (mutate_memory! machine (lookup_register pointer)
                  (fn [_] (lookup_register source))))

(defn halt! [machine]
  (swap! machine
         (fn [state]
           (assoc :halted true))))

(defn conditional_jump! [machine then condition]
  (swap! machine
         (fn [state]
           (if (= (lookup_register condition) 0)
             state
             (assoc state :current (lookup_register then))))))

(defn expt [a b]
  (Math/pow a b))

(defn digit [word i]
  (mod (quot word (expt 10 i)) 10))

(defn tick! [machine]
  (swap! machine
         (fn [state]
           (reduce #(update-in %1 [%2] inc)
                   state [:current :clock]))))

(defn operation [instruction]
  (condp #(digit % 2) instruction
    1 halt!
    ;; TODO: macro maybe??
    2 #(set_register! % (digit instruction 1) (digit instruction 0))
    3 #(add_to_register! % (digit instruction 1) (digit instruction 0))
    4 #(multiply_to_register! % (digit instruction 1) (digit instruction 0))
    5 #(set_register! % (digit instruction 1) (digit instruction 0))
    6 #(set_register_to_register! % (digit instruction 1) (digit instruction 0))
    7 #(add_register_to_register! % (digit instruction 1) (digit instruction 0))
    8 #(multiply_register_to_register! % (digit instruction 1)
                                       (digit instruction 0))
    9 #(set_register_from_address! % (digit instruction 1) (digit instruction 0))
    0 #(conditional_jump! % (digit instruction 1) (digit instruction 0))))

(defn execute! [machine instruction]
  ((operation instruction) machine))

(defn step! [machine]
  (let [instruction (lookup_memory @machine (@machine :current))]
    (tick! machine)
    (execute! machine instruction)))

(defn run! [machine]
  (while (not (@machine :halted))
    (step! machine)))

(deftest test_set_register
  (let [test_machine (clean_machine)]
    (set_register! test_machine 1 12)
    (is (= ((@test_machine :registers) 1) 12))))

(deftest test_mutate_register
  (let [test_machine (clean_machine)]
    (set_register! test_machine 1 500)
    (mutate_register! test_machine 1 #(m+ % 512))
    (is (= ((@test_machine :registers) 1) 12))))

(deftest test_tick
  (let [test_machine (clean_machine)]
    (doseq [_ (range 5)]
      (tick! test_machine))
    (is (= (@test_machine :current) 5))
    (is (= (@test_machine :clock) 5))))

(deftest test_initialize_memory
  (let [test_machine (clean_machine)]
    (initialize_memory! test_machine [1 2 3 4 5])
    (is (= (take 5 (@test_machine :memory)) [1 2 3 4 5]))
    (is (every? zero? (drop 5 (@test_machine :memory))))))

;;; XXX TODO FIXME not done debugging the entire program
(deftest test_sample_output
  (let [program [299 492 495 399 492 495 399 283 279 689 78 100]
        test_machine (clean_machine)]
    (initialize_memory! test_machine program)
    (run! test_machine)
    (is (= (@test_machine :clock) 16))))

(run-tests)
