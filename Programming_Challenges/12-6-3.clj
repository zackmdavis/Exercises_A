(ns star
  (:require [clojure.test :refer :all])
  (:require [clojure.set :refer :all]))

(defn three-cartesian-product [a b c]
  ;; turns out this is not actually what I want
  (for [ae a be b ce c]
    #{ae be ce}))

(defn row [master trail-out trail-in rank]
  ;; this is unfortunately rather _ad hoc_
  (let [length (condp = rank :major 11 :minor 9)
        tail (condp = rank :major 3 :minor 1)]
    (set (map (fn [& args] (set args))
              (repeat length master)
              (flatten (concat
                        (for [symbol trail-out] [symbol symbol])
                        (repeat tail nil)))
              (flatten (concat
                        (repeat tail nil)
                        (for [symbol trail-in] [symbol symbol])))))))

(row :A [:E :F :G :H] [:I :J :K :L] :major)

(def row-groups [[:A :B :C :D] [:E :F :G :H] [:I :J :K :L]])
(def rank-pattern [:major :minor :minor :major])

(defn enumerate [iterable]  ; I still <3 Python
  (map-indexed vector iterable))

(def trying-cells
  (apply union
         (for [[i row-group] (enumerate row-groups)
               [j row-label] (enumerate row-group)]
           (row row-label
                (row-groups (mod (+ i 2) 3))
                (row-groups (mod (+ i 1) 3))
                (rank-pattern j)))))

(row :A (row-groups 1) (row-groups 2) :major)
(row :B (row-groups 1) (row-groups 2) :minor)
(row :C (row-groups 1) (row-groups 2) :minor)
(row :D (row-groups 1) (row-groups 2) :major)

;; XXX FATALLY FLAWED: this entire strategy of trying to redundantly
;; generate the cells in each row and then relying on the coercion of
;; seq to set to handle deduplication is fatally flawed, because there
;; are places near the points where two cells will share
;; row-label-sets.

;; maybe I should identify the cells with just two dimensions, and
;; then overlay the third not-actually-dimensional row??

(- 65 48)

(defn bound [])

(deftest test-sample-output
  (is (bound {:A 5 :B 7 :C 8 :D 9 :E 6 :F 1 :G 9 :H 0 :I 9 :J 8 :K 4 :L 6})
      {:lower 40 :upper 172}))
