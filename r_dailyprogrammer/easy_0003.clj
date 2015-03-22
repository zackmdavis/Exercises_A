(ns ceasar-cipher
  (:require [clojure.test :refer :all]))

(def ^:dynamic
  ;; I don't actually expect wanting to rebind this, but hey, earmuffs
  ;; are in this season!
  *alphabet* (map char (range 65 91)))

(defn shift-letters [our-alphabet shift]
  (let [[start-slice end-slice] (split-at (- (count our-alphabet) shift)
                                          our-alphabet)]
    (concat end-slice start-slice)))

(defn cipher [our-alphabet shift]
  (zipmap our-alphabet (shift-letters our-alphabet shift)))

(defn interactive-cipher []
  (println "Ceasar cipher! How many letters should we shift?")
  (let [shift-response (.readLine *in*)
        shift (Integer/parseInt shift-response)
        selected-cipher (cipher *alphabet* shift)]
    (println "selected cipher is " selected-cipher)
    ;; TODO
    )
  )

(deftest test-shift-letters
  (is (= [\W \X \Y \Z \A \B \C \D \E \F \G \H \I
          \J \K \L \M \N \O \P \Q \R \S \T \U \V]
       (shift-letters *alphabet* 4))))

(deftest test-cipher
  (is (= {\A \W, \B \X, \C \Y, \D \Z, \E \A, \F \B, \G \C, \H \D, \I \E,
          \J \F, \K \G, \L \H, \M \I, \N \J, \O \K, \P \L, \Q \M, \R \N,
          \S \O, \T \P, \U \Q, \V \R, \W \S, \X \T, \Y \U, \Z \V}
         (cipher *alphabet* 4))))

(run-tests)

(defn -main []
  (interactive-cipher))
(-main)
