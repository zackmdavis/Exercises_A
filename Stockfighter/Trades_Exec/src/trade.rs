use std::io;
use std::io::{Read, Write};

use hyper::client::Client;
use hyper::header::Headers;
use serde_json;

use book::{Fill, Quote, Cents, Shares};

type OrderId = u64;

const BASE_URL: &'static str = "https://api.stockfighter.io/ob/api";

pub enum Action {
    Buy,
    Sell
}

pub enum OrderType {
    Market,
    Limit,
    FillOrKill,
    ImmediateOrCancel
}

pub struct Stockfighter {
    venue: String,
    symbol: String,
    account: String,
    api_key: String,
}

impl Stockfighter {

    pub fn new() -> Self {
        Stockfighter {
            venue: env!("VENUE").to_owned(),
            symbol: env!("SYMBOL").to_owned(),
            account: env!("ACCOUNT").to_owned(),
            api_key: env!("API_KEY").to_owned()
        }
    }

    pub fn get(&self, path: String) -> String {
        let client = Client::new();
        let mut headers = Headers::new();
        headers.set_raw("X-Starfighter-Authorization",
                        vec![self.api_key.clone().into_bytes()]);
        let mut response = client.get(&path).headers(headers).send().unwrap();
        let mut response_buffer = String::new();
        response.read_to_string(&mut response_buffer).unwrap();
        println!("{}", response_buffer);
        response_buffer
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

    pub fn get_order_book(&self) -> String {
        self.get(format!("{}/venues/{}/stocks/{}", BASE_URL,
                         self.venue, self.symbol))
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
                println!("couldn't get quote: {:?}", err);
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
        println!("{:?}", body);
        let mut response = client.post(
            &format!("{}/venues/{}/stocks/{}/orders", BASE_URL,
                     self.venue, self.symbol))
            // XXX: format argument must be a string literal
            .body(&body)
            .headers(headers)
            .send().unwrap();
        println!("response: {:?}", response);
        let mut response_buffer = String::new();
        response.read_to_string(&mut response_buffer).unwrap();
        println!("response buffer: {:?}", response_buffer);
        let response_value: serde_json::Value = serde_json::from_str(
            &response_buffer).unwrap();
        response_value.as_object().unwrap().get("id").unwrap().as_u64().unwrap()
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
}

mod tests {
    use super::{Stockfighter, Action, OrderType};

    #[test]
    fn concerning_basic_infromation_acquisition() {
        let stockfigher = Stockfighter::new();
        stockfigher.healthcheck();
        stockfigher.venue_healthcheck();
        stockfigher.get_stocks();
        stockfigher.get_order_book();
        stockfigher.get_quote();
    }

}
