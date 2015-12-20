use std::cmp;
use std::thread::sleep;
use std::time::Duration;

use book::{Cents, Fill, Position, Shares, format_money};
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


pub fn maker(fighter: Stockfighter, mut our_position: Position) {
    const BOUND_LONG: Shares = 800;
    const BOUND_SHORT: Shares = -800;

    // XXX this is a lot saner than my previous attempts, but this bot still
    // doesn't make money

    loop {
        info!("Start of the trading loop!");
        let quote_maybe = fighter.get_quote();
        let quote = match quote_maybe {
            Some(q) => q,
            None => {
                warn!("couldn't get quote, continuing from top of loop ...");
                continue;
            }
        };
        if quote.bid.is_none() || quote.ask.is_none() {
            // not sure what to do if no one's already provided the
            // other side of the spread
            warn!("quote was {}; continuing ...", quote);
            continue;
        }
        info!("quote is {}", quote);

        let bid = quote.bid.unwrap();
        let ask = quote.ask.unwrap();
        let market = (bid + ask) / 2;
        let undercut = (ask - bid) / 5;
        let underask = ask - undercut;
        let overbid = bid + undercut;

        let buy_limit = BOUND_LONG - our_position.inventory;
        let sell_limit = our_position.inventory - BOUND_SHORT;

        let shares = min_i64!(quote.bid_size, quote.ask_size,
                              buy_limit, sell_limit);
        info!("We're going to try buying and selling {} shares", shares);

        let buy_order_id = fighter.order(
            shares, overbid, Action::Buy, OrderType::Limit);
        let sell_order_id = fighter.order(
            shares, underask, Action::Sell, OrderType::Limit);
        assert!(underask - overbid > 0);

        let buy_fills = fighter.cancel_order(buy_order_id);
        for buy in &buy_fills {
            our_position.record_fill(buy);
        }

        let sell_fills = fighter.cancel_order(sell_order_id);
        for sell in &sell_fills {
            our_position.record_fill(sell);
        }

        info!("curent cash is {}", format_money(our_position.cash));
        info!("curent inventory is {}", our_position.inventory);
        info!("current profit is {}", format_money(our_position.profit(market)));
    }
}
