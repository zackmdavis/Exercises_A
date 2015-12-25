use std::cmp;
use std::thread::sleep;
use std::time::Duration;

use book::{Ask, Bid, Cents, Fill, Position, Shares, format_money};
use trade::{Stockfighter, Action, OrderType};


// This is my first Rust macro ever, so perhaps I can be forgiven if it's not
// as general or elegant as you might want
macro_rules! min_i64 {
    ( $( $x:expr ),* ) => {
        {
            let mut minimum: Option<i64> = None;
            $(
                match minimum {
                    None => {
                        minimum = Some($x);
                    }
                    Some(current_minimum) => {
                        if $x < current_minimum {
                            minimum = Some($x);
                        }
                    }
                }
            )*
           minimum.unwrap()
        }
    };
}


pub fn poll_order_book(fighter: Stockfighter) {
    loop {
        if let Some(book) = fighter.get_order_book() {
            info!("order book is: {}", book)
        }
    }
}



pub fn maker(fighter: Stockfighter, mut our_position: Position) {
    const BOUND_LONG: Shares = 500;
    const BOUND_SHORT: Shares = -500;

    let mut market = 0;

    let mut top_bid = Bid::new_bid(0, 0);
    let mut top_ask = Ask::new_ask(0, 0);

    // XXX this is significantly worse than chance

    loop {
        info!("Start of the trading loop!");
        let book_maybe = fighter.get_order_book();
        let book = match book_maybe {
            Some(b) => {
                info!("order book is: {}", b);
                b
            },
            None => {
                warn!("couldn't get book, continuing from top of loop ...");
                continue;
            }
        };

        let buy_limit = BOUND_LONG - our_position.inventory;
        let sell_limit = our_position.inventory - BOUND_SHORT;

        let mut buy_order_id = None;
        let mut sell_order_id = None;

        // up to risk limits, I can sell as many shares as I remember having
        // bought for not more
        if book.bids.len() > 0 {
            top_bid = book.bids[0];
            let net_shares_acquired_cheaper = our_position
                .trades_in_range(None, Some(top_bid.price));
            if net_shares_acquired_cheaper >= 0 {
                let sell_this_many = if net_shares_acquired_cheaper == 0 {
                    cmp::min(sell_limit, top_bid.quantity)
                } else {
                    min_i64!(sell_limit, top_bid.quantity,
                             net_shares_acquired_cheaper)
                };
                if sell_this_many > 0 {
                    sell_order_id = Some(fighter.order(
                        sell_this_many, top_bid.price, Action::Sell,
                        OrderType::Limit
                    ));
                }
            }
        }

        // up to risk limits, I can buy as many shares as I remember having
        // sold for not less
        if book.asks.len() > 0 {
            top_ask = book.asks[0];
            let net_shares_sold_more_expensively = our_position
                .trades_in_range(Some(top_ask.price), None);
            if net_shares_sold_more_expensively <= 0 {
                let buy_this_many = if net_shares_sold_more_expensively == 0 {
                    cmp::min(buy_limit, top_ask.quantity)
                } else {
                    min_i64!(buy_limit, top_ask.quantity,
                             -net_shares_sold_more_expensively)
                };
                if buy_this_many > 0 {
                    buy_order_id = Some(fighter.order(
                        buy_this_many, top_ask.price, Action::Buy,
                        OrderType::Limit
                    ));
                }
            }
        }

        market = (top_bid.price + top_ask.price) / 2;

        // How did we do??
        if let Some(order_id) = buy_order_id {
            let buy_fills = fighter.cancel_order(order_id);
            for buy in &buy_fills {
                our_position.record_fill(buy);
            }
        }

        if let Some(order_id) = sell_order_id {
            let sell_fills = fighter.cancel_order(order_id);
            for sale in &sell_fills {
                our_position.record_fill(sale);
            }
        }

        info!("curent cash is {}", format_money(our_position.cash));
        info!("curent inventory is {}", our_position.inventory);
        warn!("current profit is {}", format_money(our_position.profit(market)));
    }
}
