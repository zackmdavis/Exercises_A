// http://www.reddit.com/r/dailyprogrammer/comments/pih8x/easy_challenge_1/

// The prompt is to create a program to ask and echo back a user's
// name, age, and username, which feels so trivial as to be
// degrading. And it would be, for me, in Python. But ... I don't
// actually know how offhand how to do child's input/output in Rust,
// and what you don't know can't be degrading to do.

// XXX: old_io is probably facing deprecation if names mean anything
#![feature(old_io)]
use std::old_io;
use std::collections::HashMap;

fn main() {
    let things_to_ask_about = ["name", "age", "username"];
    let mut collected_information = HashMap::new();
    for askable in things_to_ask_about.iter() {
        println!("What is your {}?", askable);
        let input = old_io::stdin()
            .read_line()
            .ok().expect("failure message here");
        // XXX EVIDENCE OF MY IMPENDING DEATH in these moments when I
        // want to scream with the righteous fury of a person who has
        // been genuinely wronged, on the topic of what the fuck is wrong
        // with this bullshit language where you can't even trim a string
        // because "`input` does not live long enough" this and "borrowed
        // value is only valid for the block suffix following statement 1
        // at 21:48" that
        //
        // But what the fuck is wrong with this bullshit language is in
        // the map, not the territory
        //
        // on the balance of available evidence, doesn't it seem more
        // likely that the borrow checker is smarter than you, or that
        // the persons who wrote the borrow checker are smarter than you?
        //
        // and if you can't even follow their work even after several
        // scattered hours of dutifully trying to RTFM, will an
        // increasingly competitive global Economy remain interested in
        // keeping you alive and happy in the decades to come?
        //
        // I am not a person who has been genuinely wronged, just a man
        // not smart enough to know any better
        collected_information.insert(askable, input.trim());
    }

    for (askable, response) in collected_information.iter() {
        println!("You claimed that your {} is {}.", askable, response);
    }
}
