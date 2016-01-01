#![feature(btree_range, collections_bound, custom_attribute,
           custom_derive, iter_arith, plugin)]
#![plugin(serde_macros)]

extern crate ansi_term;
extern crate hyper;
#[macro_use] extern crate log;
extern crate serde;
extern crate serde_json;
extern crate time;
extern crate toml;

mod audit;
mod book;
mod office;
mod strategy;
mod trade;


use std::thread::sleep;

use audit::initialize_logging;
use book::{Position};
use office::parse_config;
use trade::{Stockfighter, Action, OrderType};
use strategy::{maker, poll_order_book};



fn main() {
    initialize_logging();
    let config = parse_config("my_config.toml");
    let fighter = Stockfighter::new(config.api_key,
                                    config.venue,
                                    config.symbol,
                                    config.account);
    let our_position = Position::new();
    maker(fighter, our_position);
}
