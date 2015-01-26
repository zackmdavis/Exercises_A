;;  Exercise 18.13 asks use to "create the largest possible rectangle
;;  of letters such that every row forms a word (reading left to
;;  right) and every column forms a word (reading top to bottom)."
;;  Having already failed at the 4x4 special case of this around this
;;  time last year (thinking that it would be interesting to have a
;;  program that generates crossword puzzles), I don't feel bad about
;;  looking at the solution early, which involves recursive
;;  backtracking and prefix tries, which itself is a promising enough
;;  hint to give me renewed hope for my crossword dream.

(ns xrossword-proof-of-concept
  (:require [clojure.test :refer :all]))

(defn string-to-sequence [word]
  (map #(keyword (clojure.string/upper-case (str %))) word))

(def our-dictionary
  (map string-to-sequence
       (filter (fn [word] (not (.contains word "'")))
               (clojure.string/split-lines
                (slurp "/usr/share/dict/words")))))

;; XXX WACK, REDUNDANCY: just memoize it or something
(defn compile-n-dictionary [n] (filter #(= (count %) n) our-dictionary))
(def two-dictionary (compile-n-dictionary 2))
(def three-dictionary (compile-n-dictionary 3))
(def four-dictionary (compile-n-dictionary 4))
(def five-dictionary (compile-n-dictionary 5))

(def n-dictionary
  {2 two-dictionary
   3 three-dictionary
   4 four-dictionary
   5 five-dictionary})

(defn is-prefix? [word letters]
  (= letters (subvec (vec word) 0 (count letters))))

(defn valid-prefix? [dictionary letters]
  (let [prefix (take-while #(not (nil? %)) letters)]
    (some #(is-prefix? % prefix) dictionary)))

(defn empty-grid [m n]
  (vec (for [row (range m)]
    (vec (for [col (range n)] nil)))))

(defn lookup [grid coordinates]
  ((grid (first coordinates)) (second coordinates)))

(defn write [grid coordinates letter]
  (assoc grid
         (first coordinates)
         (assoc (grid (first coordinates)) (second coordinates) letter)))

(defn read-row [grid row]
  (grid row))

(defn write-row [grid row word]
  (assoc grid row (vec word)))

(defn read-col [grid col]
  (vec (map #(nth % col) grid)))

(defn write-col [grid col word]
  (reduce (fn [state row] (write state [row col] (nth word row)))
          grid
          (range (count word))))

(defn solved? [grid]
  (let [width (count (read-row grid 0))
        height (count (read-col grid 0))]
    (every? identity
            (concat (for [i (range height)]
                      (some #{(read-row grid i)} (n-dictionary width)))
                    (for [j (range width)]
                      (some #{(read-col grid j)} (n-dictionary height)))))))

(defn solvable? [grid]
  (let [width (count (read-row grid 0))
        height (count (read-col grid 0))]
    (every? identity
            (concat (for [i (range height)]
                      (valid-prefix? (n-dictionary width)
                                     (read-row grid i)))
                    (for [j (range width)]
                      (valid-prefix? (n-dictionary height)
                                     (read-col grid j)))))))

(defn first-blank-row-index [grid]
  (some identity
        (map-indexed (fn [index row]
                       (if (every? #(nil? %) row)
                         index
                         nil))
                     grid)))

(defn solve [grid]
  (if (solved? grid)
    grid
    (when (solvable? grid)
      (let [next-row-index (first-blank-row-index grid)]
        (some identity
              (for [word (n-dictionary (count (read-row grid next-row-index)))]
                (solve (write-row grid next-row-index word))))))))

(deftest test-write-lookup
  (is (= :A
         (lookup (write (empty-grid 2) [0 0] :A)
                 [0 0]))))
