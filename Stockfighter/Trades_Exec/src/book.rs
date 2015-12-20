use std::collections::btree_map::BTreeMap;
use std::collections::Bound;
use std::fmt;

use time;
use serde::Deserialize;

use trade::Action;

pub type Cents = i64;
pub type Shares = i64;


pub fn format_money(amount: Cents) -> String {
    format!("${:.2}", amount as f32 / 100.)
}


#[derive(Debug, Copy, Clone)]
pub struct Fill {
    pub action: Action,
    pub quantity: Shares,
    pub price: Cents
}

impl Fill {
    pub fn new(action: Action, quantity: Shares, price: Cents) -> Self {
        Fill { action: action, quantity: quantity, price: price }
    }
}


impl fmt::Display for Fill {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(f, "{:?} {}@{}¢", self.action, self.quantity, self.price)
    }
}


#[derive(Debug, Copy, Clone)]
pub struct Bid {
    quantity: Shares,
    price: Cents
}

#[derive(Debug, Copy, Clone)]
pub struct Ask {
    quantity: Shares,
    price: Cents
}

#[derive(Debug, Clone)]
pub struct OrderBook {
    venue: String,
    symbol: String,
    // time: time::Tm, TODO
    bids: Vec<Bid>,
    asks: Vec<Ask>
}


#[derive(Debug, Clone, Deserialize)]
pub struct Quote {
    pub symbol: String,
    pub venue: String,
    pub bid: Option<i64>,
    pub ask: Option<i64>,
    // "size" is about the best (market-tracking) orders, "depth" is about all
    // orders
    #[serde(rename="bidSize")]
    pub bid_size: i64,
    #[serde(rename="askSize")]
    pub ask_size: i64,
    #[serde(rename="bidDepth")]
    pub bid_depth: i64,
    #[serde(rename="askDepth")]
    pub ask_depth: i64,
    pub last: i64,
    #[serde(rename="lastSize")]
    pub last_size: i64,
    // TODO
    // last_trade: time::Tm,
    // quote_time: time::Tm,

    // XXX: it turns out that the magical JSON-to-struct deserializer I'm
    // trying out is not very flexible; as a workaround, include the garbage or
    // yet-unused fields present in the actual JSON
    // https://github.com/serde-rs/serde/issues/44
    #[serde(rename="lastTrade")]
    last_trade: String,
    #[serde(rename="quoteTime")]
    quote_time: String,
    ok: bool
}

impl fmt::Display for Quote {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(f, "{}: bid: {} (depth {}); ask: {} (depth {})",
               self.symbol,
               match self.bid {
                   Some(bid) => format_money(bid),
                   None => "(none)".to_owned()
               },
               self.bid_depth,
               match self.ask {
                   Some(ask) => format_money(ask),
                   None => "(none)".to_owned()
               },
               self.ask_depth)
    }
}


#[derive(Debug)]
pub struct Position {
    pub cash: Cents,
    pub inventory: Shares,
    trades: BTreeMap<Cents, Shares>
}

impl Position {
    pub fn new() -> Self {
        Position {
            cash: 0,
            inventory: 0,
            trades: BTreeMap::new(),
        }
    }

    pub fn record_fill(&mut self, fill: &Fill) {
        info!("got fill: {}", fill);
        self.record_trade(fill.action, fill.quantity, fill.price)
    }

    pub fn record_trade(&mut self,
                        action: Action, quantity: Shares, price: Cents) {
        let direction = match action {
            Action::Buy => 1,
            Action::Sell => -1
        };
        let stance = quantity * direction;
        self.cash -= price * stance;
        self.inventory += stance;
        *self.trades.entry(price).or_insert(0) += stance;
    }

    pub fn trades_in_range(&self, min_price: Option<Cents>,
                             max_price: Option<Cents>)
                             -> Shares {
        let mut l = 0;
        let mut u = 0;
        let lower_bound = min_price.map_or(
            Bound::Unbounded,
            |min| {
                l = min;
                Bound::Excluded(&l)
            });
        let upper_bound = max_price.map_or(
            Bound::Unbounded,
            |max| {
                u = max;
                Bound::Excluded(&u)
            });

        self.trades.range(lower_bound, upper_bound)
            .map(|(_amount, quantity)| quantity).sum()
    }

    pub fn profit(&self, market: Cents) -> Cents {
        self.cash + self.inventory * market
    }

}


#[cfg(test)]
mod tests {
    use super::Position;
    use trade::Action;

    #[test]
    fn concerning_our_position() {
        let mut our_position = Position::new();
        our_position.record_trade(Action::Buy, 10, 100);
        // We don't profit from making a trade—if the "value" of the security
        // is what we paid for it, the value of the inventory is exactly
        // cancelled out by the decreased cash.
        assert_eq!(-1000, our_position.cash);
        assert_eq!(0, our_position.profit(100));
        // But if the price were to go up, we would count ourselves a profit.
        assert_eq!(10, our_position.profit(101));

        // imagine the price goes down, and we "cut our losses" by selling half
        // of our shares
        our_position.record_trade(Action::Sell, 5, 50);
        assert_eq!(-750, our_position.cash);
        assert_eq!(5, our_position.inventory);
        assert_eq!(-500, our_position.profit(50));
    }
}
