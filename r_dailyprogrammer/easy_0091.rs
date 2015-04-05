// http://reddit.com
// /r/dailyprogrammer/comments/yqydh/8242012_challenge_91_easy_sleep_sort/

use std::thread;

static TICK: u32 = 10;

fn printing_sleepsort(intervals: Vec<u32>) -> () {
    for interval in intervals.iter() {
        thread::spawn(|| {
            // XXX `borrowed value is only valid for the for at 9:8`
            // and this humble Pythonista is beginning to despair of ever
            // moving to the big city and learning Rust; there has to be
            // some way to break the loop of "can't read the theory of
            // borrowing without experience working with examples" and
            // "can't make anything that works without knowing the
            // theory" and "this is all too painful to even think about"
            thread::sleep_ms(TICK * interval);
            print!("{} ", interval)
        });
    }
}

fn main() {
    let demo_input: Vec<u32> = vec![5, 2, 7, 4, 6, 3, 8, 1];
    printing_sleepsort(demo_input);
}
