use std::collections::btree_map::BTreeMap;
use std::collections::Bound;
use std::fmt;

use time;
use serde::Deserialize;

use trade::Action;

pub type Cents = i64;
pub type Shares = i64;


#[derive(Copy, Clone)]
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


impl fmt::Debug for Fill {
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


#[derive(Debug)]
pub struct Position {
    stakes: BTreeMap<Cents, Shares>
}

impl Position {
    pub fn new() -> Self {
        Position { stakes: BTreeMap::new() }
    }

    pub fn record_fill(&mut self, fill: &Fill) {
        self.record_trade(fill.action, fill.quantity, fill.price)
    }

    pub fn record_trade(&mut self,
                        action: Action, quantity: Shares, price: Cents) {
        let direction = match action {
            Action::Buy => 1,
            Action::Sell => -1
        };
        let stance = quantity * direction;
        *self.stakes.entry(price).or_insert(0) += stance;
    }

    pub fn holdings(&self) -> Shares {
        self.stakes.values().sum()
    }

    pub fn holdings_in_range(&self, min_price: Option<Cents>,
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

        self.stakes.range(lower_bound, upper_bound)
            .map(|(_amount, quantity)| quantity).sum()
    }

    pub fn profit(&self) -> Cents {
        self.stakes.iter().map(|(amount, quantity)| -1 * amount * quantity).sum()
    }
}


mod tests {
    use super::Position;
    use trade::Action;

    #[test]
    fn concerning_our_position() {
        let mut our_position = Position::new();
        our_position.record_trade(Action::Buy, 10, 100);
        our_position.record_trade(Action::Sell, 5, 50);
        assert_eq!(5, our_position.holdings());
        assert_eq!(-750, our_position.profit());
    }
}
