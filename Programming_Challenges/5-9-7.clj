(ns stern-brocot
  (:use clojure.test))

(defn sane_numerator [x]
  (cond (= (type x) (type 1)) x
        (= (type x) (type [1 0])) (first x)
        :else (numerator x)))

(defn sane_denominator [x]
  (cond (= (type x) (type 1)) 1
        (= (type x) (type [1 0])) (last x)
        :else (denominator x)))

(defn introduction [seen]
  (/ (reduce + (map sane_numerator seen))
     (reduce + (map sane_denominator seen))))

(defn spawn [newlyweds]
  [(first newlyweds) (introduction newlyweds) (last newlyweds)])

(defn glance [frame target]
  (- target (second frame)))

(defn advance [frame direction]
  (cond (= direction :L) (spawn (take 2 frame))
        (= direction :R) (spawn (take-last 2 frame))))   

(defn search [frame target path]
  (let [verdict (glance frame target)]
    (cond (< verdict 0) (search (advance frame :L)
                                target
                                (conj path :L))
          (> verdict 0) (search (advance frame :R)
                                target
                                (conj path :R))
          (= verdict 0) path)))

(defn represent [x]
  (search [0/1 1 [1 0]] x []))

(deftest test_introduction
  (is (= 5/3 (introduction [3/2 2/1])))
  (is (= 7/5 (introduction [4/3 3/2]))))

(deftest test_advance
  (is (= [1 4/3 3/2] 
         (advance [1 3/2 2] :L))))

(deftest test_sample_output
  (is (= [:L :R :R :L] (represent 5/7)))
  (is (= [:R :R :L :R :R :L :R :L :L :L :L :R :L :R :R :R]
         (represent 878/323))))

(run-tests)
