use std::thread::sleep;
use std::time::Duration;

use book::{Cents, Fill, Position, Shares};
use trade::{Stockfighter, Action, OrderType};


pub fn maker() {
    // this doesn't do much and I don't understand what's going on

    const BOUND_LONG: Shares = 800;
    const BOUND_SHORT: Shares = -800;

    let stockfighter = Stockfighter::new();
    let mut our_position = Position::new();

    loop {
        println!("Start of the loop!");
        let quote_maybe = stockfighter.get_quote();
        let quote = match quote_maybe {
            Some(q) => q,
            None => {
                print!("couldn't get quote, continuing loop ...");
                continue;
            }
        };
        if quote.bid.is_none() || quote.ask.is_none() {
            continue;  // not sure what to do with this kind of quote yet
        }
        let bid = quote.bid.unwrap();
        let ask = quote.ask.unwrap();
        let market = (bid + ask)/2;
        let holdings = our_position.holdings();

        let we_can_buy = BOUND_LONG - holdings;
        let we_can_sell = holdings - BOUND_SHORT;

        if we_can_buy > 0 {
            let buy_order_id = stockfighter.order(
                we_can_buy, market,
                Action::Buy, OrderType::ImmediateOrCancel
            );
            loop {
                let status_maybe = stockfighter.get_closed_order_status(
                    buy_order_id);
                match status_maybe {
                    None => {
                        println!("Waiting for status of order {}", buy_order_id);
                    },
                    Some(fills) => {
                        for fill in &fills {
                            println!("buy fill: {:?}", fill);
                            our_position.record_fill(fill);
                        }
                    }
                }
            }
        }

        println!("Finished buy phase of the loop, on to sell phase!");

        // XXX TODO FIXME: so copy-pastey; this is disgusting
        if we_can_sell > 0 {
            let sell_order_id = stockfighter.order(
                we_can_sell, market,
                Action::Sell, OrderType::ImmediateOrCancel
            );
            loop {
                let status_maybe = stockfighter.get_closed_order_status(
                    sell_order_id);
                match status_maybe {
                    None => {
                        println!("Waiting for status of order {}",
                                 sell_order_id);
                    },
                    Some(fills) => {
                        for fill in &fills {
                            println!("sell fill: {:?}", fill);
                            our_position.record_fill(fill);
                        }
                    }
                }
            }
        }
        println!("done with loop; current profit is {}", our_position.profit());
        println!("sleeping for 1 second ...");
        sleep(Duration::from_secs(1));
    }
}
