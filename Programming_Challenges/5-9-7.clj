(ns stern-brocot
  (:use clojure.test))

;; to be continued ...
;; note that Clojure's Ratio type doesn't support integers (!?!)---
;; user=> (numerator 2/3)
;; 2
;; user=> (numerator 2/1)
;; ClassCastException java.lang.Long cannot be cast to
;; clojure.lang.Ratio clojure.core/numerator (core.clj:3183)

(defn introduction [seen]
  (/ (reduce + (map numerator seen))
     (reduce + (map denominator seen))))

(deftest test_introduction
  (is (= 5/3 (introduction [3/2 2/1]))))

(run-tests)
