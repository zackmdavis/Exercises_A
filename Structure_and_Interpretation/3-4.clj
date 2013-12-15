(defn call-the-cops []
  (println
   "too many failed access attempts; authorities have been notified"))

(defn make-account [initial-balance password]
  (let [balance (atom initial-balance)
        failed-authentications (atom 0)
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
                     (do (swap! failed-authentications (fn [x] 0))
                         (cond (= action :withdraw) (withdraw amount)
                               (= action :deposit) (deposit amount)
                               :else "the bank doesn't understand that"))
                     (if (>= @failed-authentications 7)
                       (call-the-cops)
                       (do (swap! failed-authentications inc) 
                           "invalid password"))))]
    dispatch))

(def my-checking-account (make-account 100 "fr13ndship"))
(println (my-checking-account "fr13ndship" :deposit 25))
(println (my-checking-account "fr13ndship" :withdraw 20))
(println (my-checking-account "fr13ndship" :withdraw 200))
(println (my-checking-account "fr13ndship" :credit-default-swap 20))
(dotimes [i 8]
  (println (my-checking-account "password1" :withdraw 1000)))

