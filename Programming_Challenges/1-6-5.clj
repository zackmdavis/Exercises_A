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
  (assoc canvas y (assoc (canvas y) x color)))

(defn hit_pixel [x y color] ; returns function to be `swap!`ed
  (fn [canvas]
    (color_at_point canvas x y color)))

(defn issue [workspace command]
  (let [opcode (first command)
        arguments (rest command)]
    (swap! workspace (action opcode arguments))))

(def operations {})

(defn action [opcode arguments]
  (fn [canvas]
    (apply (operations opcode) (concat [canvas] arguments))))
