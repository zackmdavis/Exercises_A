#![feature(inclusive_range_syntax)]

// https://www.reddit.com/r/dailyprogrammer/comments/5j6ggm/20161219_challenge_296_easy_the_twelve_days_of/

// yet another exercise that should be far beneath me, the undertaking intended
// mainly to get one's fingers moving DOING SOMETHING, no matter how trivial

const NAME_ORDINALS: &'static [(&'static str, &'static str)] = &[
    ("no", "zeroth"),
    ("a", "first"),
    ("two", "second"),
    ("three", "third"),
    ("four", "fourth"),
    ("five", "fifth"),
    ("six", "sixth"),
    ("seven", "seventh"),
    ("eight", "eighth"),
    ("nine", "ninth"),
    ("ten", "tenth"),
    ("eleven", "eleventh"),
    ("twelve", "twlefth"),
];

const GIFTS: &'static [&'static str] = &[
    "nothing",
    "partridge in a pear tree",
    "turtle doves",
    "french hens",
    "calling birds",
    "golden rings",
    "geese a-laying",
    "swans a-swimming",
    "maids a-milking",
    "ladies dancing",
    "lords a-leaping",
    "pipers piping",
    "drummers drumming",
];


fn main() {
    for day in 0...12 {
        println!("On the {} day of Christmas, my true love gave to me:", NAME_ORDINALS[day].1);
        for giftdex in (0...day).rev() {
            println!(" • {} {}", NAME_ORDINALS[giftdex].0, GIFTS[giftdex]);
        }
        println!();
    }
}
