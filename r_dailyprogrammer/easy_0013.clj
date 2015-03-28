;; http://www.reddit.com/r/dailyprogrammer/comments/pzo4w/2212012_challenge_13_easy/

;; day number from calendar date

(ns day-count
  (:require [clojure.test :refer :all]))

(def month-to-index (zipmap [:January :February :March :April
                             :May :June :July :August
                             :September :October :November :December]
                            (range 1 13)))

(def month-lengths [nil 31 28 31 30 31 30 31 31 30 31 31 31])

(defn day-of-year [month day]
  ;; (+ (reduce ) TODO
)

(deftest test-known-dates
  ;; TODO
)
