extern crate hyper;

use std::io::Read;

use hyper::client::Client;
use hyper::header::Headers;


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

enum Direction {
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
         action: Direction, order_type: OrderType) {
    let client = Client::new();
    let direction_parameter = match action {
        Direction::Buy => "buy",
        Direction::Sell => "sell"
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
    healthcheck();
    venue_healthcheck(env!("VENUE"));
    get_stocks(env!("VENUE"));
    get_order_book(env!("VENUE"), env!("SYMBOL"));
    for i in 0..1000 {
        order(env!("ACCOUNT"), env!("VENUE"), env!("SYMBOL"), 0, 100,
              Direction::Buy, OrderType::Market);
    }
}
