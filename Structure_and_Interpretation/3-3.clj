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

(def my-checking-account (make-account 100 "fr13ndship"))
(println (my-checking-account "fr13ndship" :deposit 25))
(println (my-checking-account "fr13ndship" :withdraw 20))
(println (my-checking-account "fr13ndship" :withdraw 200))
(println (my-checking-account "fr13ndship" :credit-default-swap 20))
(println (my-checking-account "password1" :withdraw 1000))

; =>
;; zmd@ExpectedReturn:~/Code/[...]$ clojure 3-3.clj 
;; 125
;; 105
;; insufficient funds
;; the bank doesn't understand that
;; invalid password
; [OK on all counts!]