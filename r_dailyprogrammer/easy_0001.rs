// http://www.reddit.com/r/dailyprogrammer/comments/pih8x/easy_challenge_1/

// The prompt is to create a program to ask and echo back a user's
// name, age, and username, which feels so trivial as to be
// degrading. And it would be, for me, in Python. But ... I don't
// actually know how offhand how to do child's input/output in Rust,
// and what you don't know can't be degrading to do.

use std::io;
use std::collections::HashMap;

fn main() {
    let things_to_ask_about = ["name", "age", "username"];
    let mut collected_information = HashMap::new();
    for askable in things_to_ask_about.iter() {
        println!("What is your {}?", askable);
        let mut input_buffer = String::new();
        io::stdin()
            .read_line(&mut input_buffer)
            .ok().expect("failure message here");
        // with thanks to Steve Klabnik and AndR
        // (http://zackmdavis.net/blog/2015/03/xxx-ii/#comments)
        collected_information.insert(askable, input_buffer.trim().to_string());
    }
    for (askable, response) in collected_information.iter() {
        println!("You claimed that your {} is {}.", askable, response);
    }
}
