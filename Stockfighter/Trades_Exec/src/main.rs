#![feature(btree_range, collections_bound, custom_attribute,
           custom_derive, iter_arith, plugin)]
#![plugin(serde_macros)]

extern crate hyper;
extern crate serde;
extern crate serde_json;
extern crate time;

mod book;
mod strategy;
mod trade;


use std::thread::sleep;

use book::{};
use trade::{Stockfighter, Action, OrderType};
use strategy::{maker};


fn main() {
    maker()
}
