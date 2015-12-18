use time;

use serde::Deserialize;


#[derive(Debug, Copy, Clone)]
pub struct Bid {
    price: u32,
    quantity: u32
}

#[derive(Debug, Copy, Clone)]
pub struct Ask {
    price: u32,
    quantity: u32
}

#[derive(Debug, Clone)]
pub struct OrderBook {
    venue: String,
    symbol: String,
    // time: time::Tm, TODO
    bids: Vec<Bid>,
    asks: Vec<Ask>
}


//   "ok": true,
//   "symbol": "KFM",
//   "venue": "YMPEX",
//   "bid": 7545,
//   "ask": 7582,
//   "bidSize": 264,
//   "askSize": 14,
//   "bidDepth": 1788,
//   "askDepth": 42,
//   "last": 7452,
//   "lastSize": 40,
//   "lastTrade": "2015-12-18T05:08:55.471582667Z",
//   "quoteTime": "2015-12-18T05:08:55.547223841Z"
// }


#[derive(Debug, Clone, Deserialize)]
pub struct Quote {
    pub symbol: String,
    pub venue: String,
    pub bid: Option<u32>,
    pub ask: Option<u32>,
    // "size" is about the best (market-tracking) orders, "depth" is about all
    // orders
    #[serde(rename="bidSize")]
    pub bid_size: u32,
    #[serde(rename="askSize")]
    pub ask_size: u32,
    #[serde(rename="bidDepth")]
    pub bid_depth: u32,
    #[serde(rename="askDepth")]
    pub ask_depth: u32,
    pub last: u32,
    #[serde(rename="lastSize")]
    pub last_size: u32,
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
