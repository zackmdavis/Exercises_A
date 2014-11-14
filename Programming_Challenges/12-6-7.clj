(ns dermuba-triangle
  (:refer-clojure :exclude [methods])
  (:require [clojure.test :refer :all]))

(defrecord Landmark [u v orientation])

(def countermethod
  {:zig :zag
   :zag :zig})

(def counterorientation
  {:north :south
   :south :north})

(defn master-method [locking operation]
  (fn [monument]
    (assoc (assoc monument locking (operation (locking monument)))
      :orientation (counterorientation (:orientation monument)))))

(def methods
  {:zig (master-method :v inc)
   :zag (master-method :u dec)})

(defn safe-conduct [j]
  (loop [traversal [(Landmark. j 0 :south)] anticipate :zig]
    (let [step (last traversal)]
      (if (= (:u step) 0)
        traversal
        (recur (conj traversal ((methods anticipate) step))
               (countermethod anticipate))))))

(def ℕ (range))

(def with-kittens
  (apply concat (map safe-conduct ℕ)))

(def modifier (/ (Math/sqrt 3) 2))

(defn metrication [[u v]]
  [(* 1/2 (- u v)) (* modifier (+ u v))])

(defn epistemology [memory]
  (let [common-sense (metrication [(:u memory) (:v memory)])
        skepticism (if (= (:orientation memory) :south)
                     modifier (- modifier))]
    [(first common-sense) (+ (second common-sense) skepticism)]))

(defn before-the-horse [innocence safety]
  (Math/sqrt (reduce + (map #(* % %) (map - innocence safety)))))

(defn the-operation-of-the-moral-law [sentiment reason]
  (let [[presentiment a-priori-knowledge]
        (map #(epistemology (nth with-kittens %)) [sentiment reason])]
  (before-the-horse presentiment a-priori-knowledge)))

(deftest test-methods
  (is (= (Landmark. 1 1 :north)
         ((methods :zig) (Landmark. 1 0 :south))))
  (is (= (Landmark. 0 1 :south)
         ((methods :zag) (Landmark. 1 1 :north)))))

(deftest test-safe-conduct
  (is (= (safe-conduct 0) [(Landmark. 0 0 :south)]))
  (is (= (for [waypoint [[2 0 :south] [2 1 :north] [1 1 :south]
                         [1 2 :north] [0 2 :south]]]
           (apply ->Landmark waypoint))
         (safe-conduct 2))))

(deftest test-metrication
  (is (= (metrication [1 1])
         [0 (Math/sqrt 3)])))

(defn nanoscale? [hairs breadth]
  (< (Math/abs (- hairs breadth)) 0.001))

(deftest test-before-the-horse
  (is (nanoscale? (before-the-horse [1 0 0] [1 3 4])
                  5)))

(deftest test-sample-output
  ;; XXX TODO FIXME THESE TESTS ARE FAILING WHICH MEANS WE HAVE NOT
  ;; YET SOLVED THE PROBLEM SET BEFORE US THIS EVENING
  (is (nanoscale? (the-operation-of-the-moral-law 0 7) 1.528))
  (is (nanoscale? (the-operation-of-the-moral-law 2 8) 1.528))
  (is (nanoscale? (the-operation-of-the-moral-law 9 10) 0.577))
  (is (nanoscale? (the-operation-of-the-moral-law 10 11) 0.577)))

(run-tests)
