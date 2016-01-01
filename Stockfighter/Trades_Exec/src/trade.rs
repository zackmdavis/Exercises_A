use std::collections::BTreeMap;
use std::io;
use std::io::{Read, Write};

use hyper::client::Client;
use hyper::header::Headers;
use hyper::method::Method;
use serde_json;
use time;

use book::{Fill, Quote, Cents, Shares, Bid, Ask, OrderBook};
use office::parse_config;

type OrderId = u64;

type JsonObject = BTreeMap<String, serde_json::Value>;


const BASE_URL: &'static str = "https://api.stockfighter.io/ob/api";

#[derive(Debug, Clone, Copy)]
pub enum Action {
    Buy,
    Sell
}

#[derive(Debug, Clone, Copy)]
pub enum OrderType {
    Market,
    Limit,
    FillOrKill,
    ImmediateOrCancel
}

#[derive(Debug, Clone)]
pub struct Stockfighter {
    venue: String,
    symbol: String,
    account: String,
    api_key: String,
}

impl Stockfighter {

    pub fn new(api_key: String,
               venue: String, symbol: String, account: String) -> Self {
        Stockfighter {
            venue: venue,
            symbol: symbol,
            account: account,
            api_key: api_key,
        }
    }

    pub fn request(&self, method: Method, path: String) -> String {
        // TODO? Option<Body> to use this for POST, too
        let client = Client::new();
        let mut headers = Headers::new();
        headers.set_raw("X-Starfighter-Authorization",
                        vec![self.api_key.clone().into_bytes()]);
        let mut response = client.request(method.clone(), &path)
            .headers(headers).send().unwrap();
        let mut response_buffer = String::new();
        response.read_to_string(&mut response_buffer).unwrap();
        debug!("response to {:?} {} was: {:?}", method, path, response_buffer);
        response_buffer
    }

    pub fn get(&self, path: String) -> String {
        self.request(Method::Get, path)
    }

    pub fn healthcheck(&self) -> String {
        self.get(format!("{}/heartbeat", BASE_URL))
    }

    pub fn venue_healthcheck(&self) -> String {
        self.get(format!("{}/venues/{}/heartbeat", BASE_URL, self.venue))
    }

    pub fn get_stocks(&self) -> String {
        self.get(format!("{}/venues/{}/stocks", BASE_URL, self.venue))
    }

    pub fn get_order_book(&self) -> Option<OrderBook> {
        let response_buffer = self.get(
            format!("{}/venues/{}/stocks/{}", BASE_URL,
                    self.venue, self.symbol));
        let response_value: serde_json::Value = serde_json::from_str(
            &response_buffer).unwrap();
        let response_object: &JsonObject = response_value.as_object().unwrap();
        let ok = response_object.get("ok").unwrap().as_boolean().unwrap();
        if ok {
            let timestamp = time::strptime(
                response_object.get("ts").unwrap().as_string().unwrap(),
                // like "2015-12-04T09:02:16.680986205Z"
                "%Y-%m-%dT%T"
            ).expect("couldn't parse timestamp");

            // XXX: in Python, I would have immediately written
            // ```
            // bids, asks = [], []
            // for proposal_type, proposals in (("bids", bids), ("asks", asks)):
            //     ...
            // ```
            // to avoid the code-duplication, because I've done it thousands of
            // times and know that it works. But in Rust, it feels subjectively
            // easier to just copy-paste to save myself the pain of maybe
            // having to fight the borrow checker
            let mut bids = Vec::new();
            if let Some(bid_data) = response_object
                .get("bids").unwrap().as_array() {
                    for bid_datum in bid_data {
                        let bid_object = bid_datum.as_object().unwrap();
                        bids.push(
                            Bid::new_bid(
                                bid_object.get("qty")
                                    .unwrap().as_i64().unwrap(),
                                bid_object.get("price")
                                    .unwrap().as_i64().unwrap()
                            )
                        )
                    }
                }
            let mut asks = Vec::new();
            if let Some(ask_data) = response_object
                .get("asks").unwrap().as_array() {
                    for ask_datum in ask_data {
                        let ask_object = ask_datum.as_object().unwrap();
                        asks.push(
                            Ask::new_ask(
                                ask_object.get("qty")
                                    .unwrap().as_i64().unwrap(),
                                ask_object.get("price")
                                    .unwrap().as_i64().unwrap()
                            )
                        )
                    }
                }

            Some(OrderBook { venue: self.venue.clone(),
                             symbol: self.symbol.clone(),
                              bids: bids, asks: asks,
                             timestamp: timestamp })
        } else {
            let error_message = response_object.get("error")
                .unwrap().as_string().unwrap();
            warn!("couldn't get order book: {}", error_message);
            None
        }
    }

    pub fn get_quote(&self) -> Option<Quote> {
        let response_buffer = self.get(
            format!("{}/venues/{}/stocks/{}/quote", BASE_URL,
                    self.venue, self.symbol));
        let quote_result = serde_json::from_str(&response_buffer);
        match quote_result {
            Ok(quote) => {
                Some(quote)
            }
            Err(err) => {
                error!("couldn't get quote: {:?}; response buffer was {:?}",
                       err, response_buffer);
                None
            }
        }
    }

    pub fn order(&self, quantity: Shares, price: Cents,
                 action: Action, order_type: OrderType) -> OrderId {
        let client = Client::new();
        let direction_parameter = match action {
            Action::Buy => "buy",
            Action::Sell => "sell"
        };
        let order_type_parameter = match order_type {
            OrderType::Market => "market",
            OrderType::Limit => "limit",
            OrderType::FillOrKill => "fill-or-kill",
            OrderType::ImmediateOrCancel => "immediate-or-cancel"
        };
        let mut headers = Headers::new();
        headers.set_raw("X-Starfighter-Authorization",
                        vec![self.api_key.clone().into_bytes()]);
        let body = format!("\
                {{\
                \"account\": \"{}\",\
                \"venue\": \"{}\",\
                \"stock\": \"{}\",\
                \"qty\": {},\
                \"price\": {},\
                \"direction\": \"{}\",\
                \"orderType\": \"{}\"\
                }}",
                           self.account, self.venue, self.symbol,
                           quantity, price,
                           direction_parameter, order_type_parameter);
        let mut response = client.post(
            &format!("{}/venues/{}/stocks/{}/orders", BASE_URL,
                     self.venue, self.symbol))
            // XXX: format argument must be a string literal
            .body(&body)
            .headers(headers)
            .send().unwrap();
        let mut response_buffer = String::new();
        response.read_to_string(&mut response_buffer).unwrap();
        debug!("Response to order was: {:?}", response_buffer);
        let response_value: serde_json::Value = serde_json::from_str(
            &response_buffer).unwrap();
        response_value.as_object().unwrap().get("id").unwrap().as_u64().unwrap()
    }

    pub fn cancel_order(&self, order_id: OrderId) -> Vec<Fill> {
        let response_buffer = self.request(
            Method::Delete,
            format!("{}/venues/{}/stocks/{}/orders/{}",
                    BASE_URL, self.venue, self.symbol, order_id)
        );
        // XXX EVIL: copy-pasted code because that's apparently how I feel
        // today
        let response_value: serde_json::Value = serde_json::from_str(
            &response_buffer).unwrap();
        let response_object = response_value.as_object().unwrap();
        let action = match response_object.get("direction").unwrap()
            .as_string().unwrap() {
                "buy" => Action::Buy,
                "sell" => Action::Sell,
                a @ _ => panic!("unexpected action {:?}", a)
        };
        let fill_value_array_maybe = response_object.get("fills");
        match fill_value_array_maybe {
            Some(fill_value_array) => {
                let fill_value_vec = fill_value_array.as_array()
                    .unwrap();
                // XXX I wanted to map it, but the borrow checker won't let me,
                // as also explained in the other comment because this code has
                // been copy-pasted in contemptuous defiance of the moral law
                let mut fills = Vec::new();
                for fill_value in fill_value_vec {
                    fills.push(
                        Fill::new(
                            action,
                            fill_value.as_object().unwrap().get("qty").unwrap()
                                .as_i64().unwrap(),
                            fill_value.as_object().unwrap().get("price").unwrap()
                                .as_i64().unwrap())
                            )
                }
                fills
            },
            None => Vec::new()
        }
    }

    pub fn get_closed_order_status(&self, order_id: OrderId)
                                      -> Option<Vec<Fill>>  {
        let response_buffer = self.get(
            format!("{}/venues/{}/stocks/{}/orders/{}",
                     BASE_URL, self.venue, self.symbol, order_id));
        let response_value: serde_json::Value = serde_json::from_str(
            &response_buffer).unwrap();
        let response_object = response_value.as_object().unwrap();

        if response_object.get("open").unwrap().as_boolean().unwrap() {
            return None
        }

        let action = match response_object.get("direction").unwrap()
            .as_string().unwrap() {
                "buy" => Action::Buy,
                "sell" => Action::Sell,
                a @ _ => panic!("unexpected action {:?}", a)
        };
        let fill_value_array_maybe = response_object.get("fills");
        let fills = match fill_value_array_maybe {
            Some(fill_value_array) => {
                let fill_value_vec = fill_value_array.as_array()
                    .unwrap();
                // XXX I wanted to map it, but the borrow checker won't let me
                let mut fills = Vec::new();
                for fill_value in fill_value_vec {
                    fills.push(
                        Fill::new(
                            action,
                            fill_value.as_object().unwrap().get("qty").unwrap()
                                .as_i64().unwrap(),
                            fill_value.as_object().unwrap().get("price").unwrap()
                                .as_i64().unwrap())
                            )
                }
                fills
            },
            None => Vec::new()
        };
        Some(fills)
    }

    pub fn wait_for_closed_order_status(&self, order_id: OrderId) -> Vec<Fill> {
        loop {
            match self.get_closed_order_status(order_id) {
                Some(fills) => {
                    for fill in &fills {
                        info!("fill for order #{}: {}", order_id, fill);
                    }
                    return fills;
                },
                None => {
                    warn!("Waiting for status of order #{} ...", order_id);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Stockfighter, Action, OrderType};
    use office::parse_config;

    #[test]
    fn concerning_basic_infromation_acquisition() {
        let config = parse_config("testex_config.toml");
        let fighter = Stockfighter::new(config.api_key,
                                            config.venue,
                                            config.symbol,
                                            config.account);
        fighter.healthcheck();
        fighter.venue_healthcheck();
        fighter.get_stocks();
        fighter.get_order_book();
        fighter.get_quote();
    }

}
