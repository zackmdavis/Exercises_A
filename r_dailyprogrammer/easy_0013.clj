;; http://www.reddit.com/r/dailyprogrammer/comments/pzo4w/2212012_challenge_13_easy/

;; day number from calendar date

(ns day-count
  (:require [clojure.test :refer :all]))

(def month-to-index (zipmap [:January :February :March :April
                             :May :June :July :August
                             :September :October :November :December]
                            (range 12)))

(def month-lengths [31 28 31 30 31 30 31 31 30 31 30 31])

(defn day-of-year [month day]
  (let [this-months-index (month-to-index month)
        days-from-months-past (reduce + (take this-months-index
                                              month-lengths))]
    (+ days-from-months-past day)))

(deftest test-known-dates
  (are [computed day-number] (= computed day-number)
       (day-of-year :January 1) 1
       (day-of-year :December 31) 365
       (day-of-year :March 29) 88
       (day-of-year :July 2) 183))

(run-tests)
