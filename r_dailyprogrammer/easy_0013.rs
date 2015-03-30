// http://www.reddit.com/r/dailyprogrammer/comments/pzo4w/2212012_challenge_13_easy/

// day number from calendar date

use std::collections::HashMap;

fn day_of_year(month: &str, day: usize) -> usize {
    let mut month_to_index: HashMap<&str, usize> = HashMap::new();
    let months = ["January", "February", "March", "April",
                  "May", "June", "July", "August",
                  "September", "October", "November", "December"];
    for (i, month) in months.iter().enumerate() {
        month_to_index.insert(month, i);
    }
    let month_lengths = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let this_months_index = match month_to_index.get(month) {
        Some(month_index) => *month_index,
        None => panic!("invalid month name")
    };
    let mut days_from_months_past = 0;
    for i in 0..(this_months_index) {
        days_from_months_past += month_lengths[i];
    }
    days_from_months_past + day
}

#[test]
fn test_known_dates() {
    assert_eq!(day_of_year("January", 1), 1);
    assert_eq!(day_of_year("December", 31), 365);
    assert_eq!(day_of_year("March", 29), 88);
    assert_eq!(day_of_year("July", 2), 183);
}
