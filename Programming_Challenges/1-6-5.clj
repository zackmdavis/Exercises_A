(ns graphical_editor
  (:require [clojure.test :refer :all]))

(def the_workspace (atom nil))

(defn initialize [_canvas columns rows]
  (vec (for [r (range rows)]
         (vec (for [c (range columns)] :O)))))

(defn clear [canvas]
   (vec (for [row canvas]
    (vec (map (fn [x] :O) row)))))

(defn color_at_point [canvas x y color]
  (assoc canvas y (assoc (canvas y) x color)))

(defn hit_pixel [x y color]
  (fn [canvas]
    (color_at_point canvas x y color)))

(def operations {:I initialize
                 :C clear
                 :L hit_pixel})

(defn action [opcode arguments]
  (fn [canvas]
    (apply (operations opcode) (concat [canvas] arguments))))

(defn issue [workspace command]
  (let [opcode (first command)
        arguments (second command)]
    (swap! workspace (action opcode arguments))))

(deftest test_color_at_point
  (let [our_canvas (initialize nil 3 3)]
     (is (= (color_at_point (initialize nil 3 3) 1 1 :Y)
            [[:O :O :O] [:O :Y :O] [:O :O :O]]))))

(deftest test_action
  (is (= ((action :I [2 2]) nil)
         [[:O :O] [:O :O]]))
  (is (= ((action :C []) [[:Y :B] [:B :K]])
         [[:O :O] [:O :O]])))

(deftest test_issue_operations
  (let [test_workspace (atom nil)]
    (issue test_workspace [:I [2 2]])
    (is (= (deref test_workspace)
           [[:O :O] [:O :O]])))
  (let [test_workspace (atom [[:Y :B] [:B :K]])]
    (issue test_workspace [:C []])
    (is (= (deref test_workspace)
           [[:O :O] [:O :O]])))
  ;;; I don't know why I'm having such a hard time keeping the call
  ;;; stack straight in my head today; it doesn't seem like a hard
  ;;; problem at all
  ;; (let [test_workspace (atom [[:O :O] [:O :O]])]
  ;;   (issue test_workspace [:L [0 0 :T]])
  ;;   (is (= (deref test_workspace)
  ;;          [[:T :O] [:O :O]])))
)

(run-tests)
