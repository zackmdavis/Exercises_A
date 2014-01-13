;; In this exercise, we reconsider the bank-account simulation we made
;; in exercise 3.3 and add the ability to make joint accounts. Recall
;; the earlier bank account (really I should be requiring it instead
;; of pasting it here, but apparently I'm lazy and bad at
;; Clojure??)---

(defn make-account [initial-balance password]
  (let [balance (atom initial-balance)
        withdraw (fn [amount]
                   (if (>= @balance amount)
                     (do (swap! balance #(- % amount))
                         @balance)
                     "insufficient funds"))
        deposit (fn [amount]
                  (swap! balance #(+ % amount))
                  @balance)
        authenticate (fn [credential]
                       (= credential password))
        dispatch (fn [credential action amount]
                   (if (authenticate credential)
                     (cond (= action :withdraw) (withdraw amount)
                           (= action :deposit) (deposit amount)
                           :else "the bank doesn't understand that")
                     "invalid password"))]
    dispatch))


(defn make-joint [account password0 password1]
  (let [authenticate (fn [credential]
                       (= credential password1))
        dispatch (fn [credential action amount]
                   (if (authenticate credential)
                     (account password0 action amount)
                     "invalid password"))]
    dispatch))

(def our-checking-account (make-account 100 "hon3sty"))
(def your-alias-to-same
  (make-joint our-checking-account "hon3sty" "loy4lty"))
(println (your-alias-to-same "loy4lty" :deposit 4)) ; => 104