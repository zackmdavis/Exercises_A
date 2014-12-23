(ns star
  (:require [clojure.test :refer :all])
  (:require [clojure.set :refer :all]))

(def cells
  ;; this is so dumb
  ;;
  ;; (the hint in the book said "Can we compute the upper and lower
  ;; bounds for each digit in isolation?" and if I don't know how to
  ;; programmatically generate the cells, computing the bounds after
  ;; typing out the cells' line-memberships is still computing,
  ;; however mortifying)
  [#{:D :K} #{:D :L} #{:D :L}  #{:D :E} #{:G :I} #{:H :J} #{:H :I}
   #{:H :I} #{:E :L} #{:E :K} #{:E :L} #{:F :L} #{:A :I} #{:A :I}
   #{:C :L} #{:D :E} #{:D :F} #{:A :J} #{:B :H} #{:C :E} #{:A :G}
   #{:A :H} #{:A :H} #{:B :I} #{:B :I :E} #{:B :J :E} #{:B :J :F}
   #{:B :K :F} #{:B :K :G} #{:B :L :G} #{:B :L :H} #{:C :I :E}
   #{:C :I :F} #{:C :J :F} #{:C :J :G} #{:C :K :G} #{:C :K :H}
   #{:C :L :H} #{:D :I :F} #{:D :I :G} #{:D :J :G} #{:D :J :H}
   #{:D :K :H} #{:A :J :E} #{:A :K :E} #{:A :K :F} #{:A :L :F}
   #{:A :L :G}])

(def kword-to-symbol (comp symbol name))
(def kword-to-fn (comp eval kword-to-symbol))

(defn bind [cell global-upper-bounds]
  (into {}
        (for [[bound extreme] [[:lower min] [:upper max]]]
          [bound (min (map (fn [label]
                             ;; XXX TODO FINISH
                             )))])))

(defn star-bounds [cells-to-bind])

(deftest test-sample-output
  (is (star-bounds {:A 5 :B 7 :C 8 :D 9 :E 6 :F 1 :G 9 :H 0 :I 9 :J 8 :K 4 :L 6})
      {:lower 40 :upper 172}))
