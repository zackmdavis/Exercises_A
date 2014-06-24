(ns graphical_editor
  (:require [clojure.test :refer :all]))

(def the_workspace (atom nil))

(defn initialize [columns rows] ; `reset!` this into the workspace
  (vec (for [r (range rows)]
         (vec (for [c (range columns)] :O)))))

(defn clear [canvas] ; `swap!`
   (vec (for [row canvas]
    (vec (map (fn [x] :O) row)))))

(defn color_at_point [canvas x y color]
  (map-indexed  ; I feel like there must be a better way
   (fn [index row]
     (if (= index y)
       (assoc (canvas y) x color)
       row))
   canvas))

(defn hit_pixel [x y color] ; returns function to be `swap!`ed
  (fn [canvas]
    (color_at_point canvas x y color)))
