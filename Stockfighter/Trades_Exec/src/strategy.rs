use std::cmp;
use std::thread::sleep;
use std::time::Duration;

use book::{Cents, Fill, Position, Shares, format_money};
use trade::{Stockfighter, Action, OrderType};


pub fn maker(fighter: Stockfighter, mut our_position: Position) {
    const BOUND_LONG: Shares = 800;
    const BOUND_SHORT: Shares = -800;


    loop {
        info!("Start of the loop!");
        let quote_maybe = fighter.get_quote();
        let quote = match quote_maybe {
            Some(q) => q,
            None => {
                warn!("couldn't get quote, continuing from top of loop ...");
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

        let buy_gap = BOUND_LONG - holdings;
        let we_could_buy = cmp::max(buy_gap, 0);

        let sell_gap = holdings - BOUND_SHORT;
        let we_could_sell = cmp::max(sell_gap, 0);

        info!("We feel like we could buy {} and sell {} \
               at the market price of {} ",
              we_could_buy, we_could_sell, market);

        let buy_order = if we_could_buy > 0 {
            Some(fighter.order(
                we_could_buy, market,
                Action::Buy, OrderType::ImmediateOrCancel
            ))
        } else { None };
        let sell_order = if we_could_sell > 0 {
            Some(fighter.order(
                we_could_sell, market,
                Action::Sell, OrderType::ImmediateOrCancel
            ))
        } else { None };

        if let Some(buy_order_id) = buy_order {
            let buy_fills = fighter.wait_for_closed_order_status(buy_order_id);
            for buy in &buy_fills {
                our_position.record_fill(buy);
            }
        }
        if let Some(sell_order_id) = sell_order {
            let sell_fills = fighter.wait_for_closed_order_status(sell_order_id);
            for sale in &sell_fills {
                our_position.record_fill(sale);
            }
        }
        info!("current position is {:?}", our_position);
        info!("curent net holdings are {}", our_position.holdings());
        info!("current profit is {}", format_money(our_position.profit(market)));
    }
}
