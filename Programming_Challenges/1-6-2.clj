(ns minesweeper
  (:use clojure.test)
  (:require [clojure.string :as string]))

(def offsets ;; reused from Reversey
  (remove #(= [0 0] %)
          (for [i (range -1 2)
                j (range -1 2)]
            [i j])))

(defn neighborhood [field position]
   (filter identity
           (for [offset offsets]
             (let [neighbor (map + position offset)]
               (get (get field (first neighbor))
                    (second neighbor))))))

(defn parse_field [field_map]
  (vec (string/split field_map #"\n")))

(defn number_field [field_map]
  (let [field (parse_field field_map)]
    ;; XXX TODO unfinished
))

(deftest test_neighborhood
  (let [field [[:a :b :c] [:d :e :f] [:g :h :i]]]
    (is (= (set (neighborhood field [1 1]))
           #{:a :b :c :d :f :g :h :i}))
    (is (= (set (neighborhood field [0 0]))
           #{:b :d :e}))))

(deftest test_sample_output
  (is (=
(number_field
"*...
....
.*..
....")

"*100
2210
1*10
1110"))

  (is (=
(number_field
"**...
.....
.*...")

"**100
33200
1*100")))

(run-tests)