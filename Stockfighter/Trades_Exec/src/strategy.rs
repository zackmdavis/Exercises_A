use std::cmp;
use std::u64::{MIN, MAX};
use std::collections::VecDeque;
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



fn assess_market(fighter: &Stockfighter) -> (Option<Cents>, Option<Cents>) {
    let mut book;
    loop {
        let book_maybe = fighter.get_order_book();
        book = match book_maybe {
            Some(b) => {
                info!("order book is: {}", b);
                b
            },
            None => {
                warn!("couldn't get book, continuing from top of loop ...");
                continue;
            }
        };
        break;
    }
    let tightest_bid = if book.bids.len() > 0 {
        Some(book.bids[0].price)
    } else {
        None
    };
    let tightest_ask = if book.asks.len() > 0 {
        Some(book.asks[0].price)
    } else {
        None
    };
    (tightest_bid, tightest_ask)
}

const BOUND_LONG: Shares = 500;
const BOUND_SHORT: Shares = -500;

fn assess_exposure(position: &Position) -> (Shares, Shares) {
    let mut buy_limit = BOUND_LONG - position.inventory;
    if buy_limit < 0 {
        buy_limit = 0;
    }
    let mut sell_limit = position.inventory - BOUND_SHORT;
    if sell_limit < 0 {
        sell_limit = 0;
    }
    (buy_limit, sell_limit)
}


fn assess_a_virtue_which_is_nameless(position: &Position, market: Cents) {
    info!("curent cash is {}", format_money(position.cash));
    info!("curent inventory is {}", position.inventory);
    warn!("current profit is {}", format_money(position.profit(market)));
}


pub fn maker(fighter: Stockfighter, mut our_position: Position) {
    let mut outstanding_buy_order_ids = VecDeque::new();
    let mut outstanding_sell_order_ids = VecDeque::new();

    let mut tightest_bid = None;
    let mut tightest_ask = None;

    let mut buy_limit = BOUND_LONG;
    let mut sell_limit = BOUND_SHORT;

    let market_assessment = assess_market(&fighter);
    // XXX so bad
    let mut our_last_bid: Cents = market_assessment.0.unwrap();
    let mut our_last_ask: Cents = market_assessment.1.unwrap();

    loop {
        // XXX: this is absolutely hideous code; why aren't I taking this
        // seriously? Real fake money is at stake!!
        info!("Start of the trading loop!");
        warn!("Outstanding buy order IDs: {:?}", outstanding_buy_order_ids);
        warn!("Outstanding sell order IDs: {:?}", outstanding_sell_order_ids);

        let market_assessment = assess_market(&fighter);
        tightest_bid = market_assessment.0;
        tightest_ask = market_assessment.1;
        let exposure_assessment = assess_exposure(&our_position);
        buy_limit = exposure_assessment.0;
        sell_limit = exposure_assessment.1;
        assess_a_virtue_which_is_nameless(&our_position,
                                          our_last_bid + our_last_ask / 2);
        info!("tight: ({:?}, {:?}); exposure limits: ({}, {})",
              tightest_bid, tightest_ask, buy_limit, sell_limit);

        match tightest_bid {
            Some(tight_bid) => {
                if tight_bid > our_last_bid {
                    // we've been outbid; cancel our outstanding, stale orders
                    while outstanding_buy_order_ids.len() > 0 {
                        if let Some(previous_order_id) =
                            outstanding_buy_order_ids.pop_front() {
                                let buy_fills = fighter
                                    .cancel_order(previous_order_id);
                                for buy in &buy_fills {
                                    our_position.record_fill(buy);
                                }
                            }
                    }

                    let exposure_assessment = assess_exposure(&our_position);
                    buy_limit = exposure_assessment.0;
                    sell_limit = exposure_assessment.1;

                    // and refile
                    let our_new_bid = tight_bid + 2;
                    let buy_order_id = fighter.order(
                        buy_limit/2, our_new_bid, Action::Buy,
                        OrderType::Limit
                    );
                    outstanding_buy_order_ids.push_back(buy_order_id);
                    our_last_bid = our_new_bid;
                }
            },
            None => {
                // we get to set the bid price
                let our_new_bid = our_last_ask - 200;
                let buy_order_id = fighter.order(
                    buy_limit/2, our_new_bid, Action::Buy,
                    OrderType::Limit
                );
                outstanding_buy_order_ids.push_back(buy_order_id);
                our_last_bid = our_new_bid;
            }
        }

        let market_assessment = assess_market(&fighter);
        tightest_bid = market_assessment.0;
        tightest_ask = market_assessment.1;
        let exposure_assessment = assess_exposure(&our_position);
        buy_limit = exposure_assessment.0;
        sell_limit = exposure_assessment.1;
        assess_a_virtue_which_is_nameless(&our_position,
                                          our_last_bid + our_last_ask / 2);
        info!("tight: ({:?}, {:?}); exposure limits: ({}, {})",
              tightest_bid, tightest_ask, buy_limit, sell_limit);

        match tightest_ask {
            Some(tight_ask) => {
                if tight_ask < our_last_ask {
                    // we've been outasked; cancel our last order
                    while outstanding_sell_order_ids.len() > 0 {
                        if let Some(previous_order_id) =
                            outstanding_sell_order_ids.pop_front() {
                                let sell_fills = fighter
                                    .cancel_order(previous_order_id);
                                for sale in &sell_fills {
                                    our_position.record_fill(sale);
                                }
                            }
                    }
                    let exposure_assessment = assess_exposure(&our_position);
                    buy_limit = exposure_assessment.0;
                    sell_limit = exposure_assessment.1;

                    // and refile
                    let our_new_ask = tight_ask - 2;
                    let sell_order_id = fighter.order(
                        sell_limit/2, our_new_ask, Action::Sell,
                        OrderType::Limit
                    );
                    outstanding_sell_order_ids.push_back(sell_order_id);
                    our_last_ask = our_new_ask;
                }
            },
            None => {
                // we get to set the ask price
                let our_new_ask = our_last_ask + 200;
                let sell_order_id = fighter.order(
                    sell_limit/2, our_new_ask, Action::Sell,
                    OrderType::Limit
                );
                outstanding_sell_order_ids.push_back(sell_order_id);
                our_last_ask = our_new_ask;
            }
        }
    }
}
