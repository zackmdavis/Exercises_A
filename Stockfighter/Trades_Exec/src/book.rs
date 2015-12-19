use std::collections::btree_map::BTreeMap;

use time;
use serde::Deserialize;


pub type Cents = i64;
pub type Shares = i64;


#[derive(Debug, Copy, Clone)]
pub struct Fill {
    pub price: Cents,
    pub quantity: Shares
}

impl Fill {
    pub fn new(quantity: Shares, price: Cents) -> Self {
        Fill { price: price, quantity: quantity }
    }
}


#[derive(Debug, Copy, Clone)]
pub struct Bid {
    price: Cents,
    quantity: Shares
}

#[derive(Debug, Copy, Clone)]
pub struct Ask {
    price: Cents,
    quantity: Shares
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
