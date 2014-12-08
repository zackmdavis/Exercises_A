(ns edit-step-ladders
  (:require [clojure.test :refer :all]))

(defn substitution? [one-word another]
  (and (apply = (map count [one-word another]))
       (= 1 (count (remove identity (map #(= %1 %2) one-word another))))))

(defn removed [string index]
  (let [splitten (split-at index string)]
    (clojure.string/join (concat (first splitten) (rest (second splitten))))))

(defn once-imposed? [longer shorter]
  (let [positions (map concat (map-indexed vector longer) (map vector shorter))
        seam (some #(if (not= (% 1) (% 2)) (% 0)) positions)]
    ;; XXX FIXME
    (= (removed longer seam) shorter)))

(once-imposed? "rah" "rahh")

(defn insertion-or-deletion? [one-word another]
  (and (= 1 (Math/abs (apply - (map count [one-word another]))))
       ;; TODO
       ))

(deftest test-substitution?
  (is (substitution? "rah" "bah"))
  (is (substitution? "America" "Americo"))
  (is (substitution? "distraction" "distruction"))
  (is (not (substitution? "alleged" "fragmentary"))))

(deftest test-removed
  (is (= "linter" (removed (removed "splinter" 0) 0))))

(deftest test-once-imposed?
  ;; XXX: clojure.lang.LazySeq cannot be cast to clojure.lang.IFn
  (is (once-imposed? "bate" "bat"))
  (is (once-imposed? "heir" "eir"))
  (is (not (once-imposed? "affine" "fine"))))

(run-tests)
