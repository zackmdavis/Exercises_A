#![feature(custom_derive, custom_attribute, plugin, associated_consts)]
#![plugin(serde_macros)]

extern crate hyper;
extern crate serde;
extern crate serde_json;
extern crate time;

mod book;
mod trade;

use std::thread::sleep;

use book::{};
use trade::{Stockfighter, Action, OrderType};

fn main() {
    let stockfighter = Stockfighter::new();
    let order_id = stockfighter.order(
        10, 10000, Action::Buy, OrderType::Limit);
    println!("order ID was: {}", order_id);
    println!("sleeping ...");
    sleep(std::time::Duration::from_secs(2));
    let fills = stockfighter.get_closed_order_status(order_id);
    println!("Fills for the order: {:?}", fills);
}
