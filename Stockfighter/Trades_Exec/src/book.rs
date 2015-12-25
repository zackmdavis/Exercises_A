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
        write!(f, "{:?} {}@${}",
               self.action, self.quantity, format_money(self.price))
    }
}


#[derive(Debug, Copy, Clone)]
pub struct Proposal {
    pub action: Action,
    pub quantity: Shares,
    pub price: Cents,
}

pub type Bid = Proposal;
pub type Ask = Proposal;

impl Proposal {
    pub fn new_bid(quantity: Shares, price: Cents) -> Self {
        Bid { quantity: quantity, price: price, action: Action::Buy }
    }

    pub fn new_ask(quantity: Shares, price: Cents) -> Self {
        Ask { quantity: quantity, price: price, action: Action::Sell }
    }
}

impl fmt::Display for Proposal {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(f, "{:?} {}@{}",
               self.action, self.quantity, format_money(self.price))
    }
}

#[derive(Debug, Clone)]
pub struct OrderBook {
    pub venue: String,
    pub symbol: String,

    pub bids: Vec<Bid>,
    pub asks: Vec<Ask>,

    pub timestamp: time::Tm
}

impl fmt::Display for OrderBook {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        try!(write!(f, "Time: {}\n\n",
                    time::strftime("%+", &self.timestamp).unwrap()));
        try!(write!(f, "Bids:\n"));
        for bid in &self.bids {
            try!(write!(f, "{}\n", bid));
        }
        if self.bids.len() == 0 {
            try!(write!(f, "—\n"));
        }
        try!(write!(f, "Asks:\n"));
        for ask in &self.asks {
            try!(write!(f, "{}\n", ask));
        }
        if self.asks.len() == 0 {
            try!(write!(f, "—\n"));
        }
        write!(f, "\n")
    }
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
    trades: BTreeMap<Cents, Shares>,
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
