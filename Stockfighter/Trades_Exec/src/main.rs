#![feature(custom_derive, custom_attribute, plugin)]
#![plugin(serde_macros)]

extern crate hyper;
extern crate serde;
extern crate serde_json;
extern crate time;

mod book;

use std::io::Read;

use hyper::client::Client;
use hyper::header::Headers;
use serde_json::*;

use book::Quote;


const BASE_URL: &'static str = "https://api.stockfighter.io/ob/api";

/// XXX: this is super copy-pastey and DAMP while I'm just getting started
/// playing around


fn healthcheck() {
    let client = Client::new();
    let mut response = client.get(&format!("{}/heartbeat", BASE_URL))
        .send().unwrap();
    let mut response_buffer = String::new();
    response.read_to_string(&mut response_buffer).unwrap();
    println!("{}", response_buffer);
}

fn venue_healthcheck(venue: &str) {
    let client = Client::new();
    let mut response = client.get(
        &format!("{}/venues/{}/heartbeat", BASE_URL, venue)).send().unwrap();
    let mut response_buffer = String::new();
    response.read_to_string(&mut response_buffer).unwrap();
    println!("{}", response_buffer);
}

fn get_stocks(venue: &str) {
    let client = Client::new();
    let mut response = client.get(
        &format!("{}/venues/{}/stocks", BASE_URL, venue)).send().unwrap();
    let mut response_buffer = String::new();
    response.read_to_string(&mut response_buffer).unwrap();
    println!("{}", response_buffer);
}

fn get_order_book(venue: &str, symbol: &str) {
    let client = Client::new();
    let mut response = client.get(
        &format!("{}/venues/{}/stocks/{}", BASE_URL, venue, symbol))
        .send().unwrap();
    let mut response_buffer = String::new();
    response.read_to_string(&mut response_buffer).unwrap();
    println!("{}", response_buffer);
}


fn get_quote(venue: &str, symbol: &str) -> Option<Quote> {
    let client = Client::new();
    let mut response = client.get(
        &format!("{}/venues/{}/stocks/{}/quote", BASE_URL, venue, symbol))
        .send().unwrap();
    let mut response_buffer = String::new();
    response.read_to_string(&mut response_buffer).unwrap();
    println!("{}", response_buffer);
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


enum Action {
    Buy,
    Sell
}

enum OrderType {
    Market,
    Limit,
    FillOrKill,
    ImmediateOrCancel
}


const ORDER_TEMPLATE: &'static str = "\
{\
  \"account\": \"{}\",\
  \"venue\": \"{}\",\
  \"stock\": \"{}\",\
  \"qty\": {},\
  \"direction\": \"{}\",\
  \"orderType\": \"{}\"\
}";


fn order(account: &str, venue: &str, symbol: &str, price: u32, quantity: u32,
         action: Action, order_type: OrderType) {
    let client = Client::new();
    let direction_parameter = match action {
        Action::Buy => "buy",
        Action::Sell => "sell"
    };
    let mut headers = Headers::new();
    headers.set_raw("X-Starfighter-Authorization",
                    vec![format!("{}", env!("API_KEY")).into_bytes()]);
    let mut response = client.post(
        &format!("{}/venues/{}/stocks/{}/orders", BASE_URL, venue, symbol))
        // XXX: format argument must be a string literal
        .body(&format!("\
{{\
  \"account\": \"{}\",\
  \"venue\": \"{}\",\
  \"stock\": \"{}\",\
  \"qty\": {},\
  \"direction\": \"{}\",\
  \"orderType\": \"{}\"\
}}", account, venue, symbol, quantity,
                      direction_parameter, "market"))
        .headers(headers)
        .send().unwrap();
    let mut response_buffer = String::new();
    response.read_to_string(&mut response_buffer).unwrap();
    println!("{}", response_buffer);
}


fn main() {
    let mut quote: Quote;
    let mut holdings: i32 = 0;
    let mut bought_at: Option<u32> = None;
    loop {
        // this is a buggy implementation of a stupid strategy, but I don't
        // have much time to hack on this because I need to go to sleep
        quote = get_quote(env!("VENUE"), env!("SYMBOL")).unwrap();
        if quote.bid.is_some() && quote.ask.is_some() {
            if holdings < 500 {
                order(env!("ACCOUNT"), env!("VENUE"), env!("SYMBOL"),
                      quote.ask.unwrap()-2, 500, Action::Buy, OrderType::Market);
                holdings += 500;
                bought_at = Some((quote.bid.unwrap() + quote.ask.unwrap()) / 2);
            }
            if let Some(bought_price) = bought_at {
                if (quote.bid.unwrap() + quote.ask.unwrap()) / 2 > bought_price {
                    order(env!("ACCOUNT"), env!("VENUE"), env!("SYMBOL"),
                          quote.ask.unwrap()-2, 500,
                          Action::Sell, OrderType::Market);
                    holdings -= 500;
                    bought_at = None;
                }
            }
        }
        println!("purported holdings are {}", holdings);
    }
}
