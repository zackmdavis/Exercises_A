#![feature(btree_range, collections_bound, custom_attribute,
           custom_derive, iter_arith, plugin)]
#![plugin(serde_macros)]

extern crate ansi_term;
extern crate hyper;
#[macro_use] extern crate log;
extern crate serde;
extern crate serde_json;
extern crate time;

mod audit;
mod book;
mod strategy;
mod trade;


use std::thread::sleep;

use audit::initialize_logging;
use book::{Position};
use trade::{Stockfighter, Action, OrderType};
use strategy::{maker};



fn main() {
    initialize_logging();
    let fighter = Stockfighter::new();
    let our_position = Position::new();
    maker(fighter, our_position);
}
