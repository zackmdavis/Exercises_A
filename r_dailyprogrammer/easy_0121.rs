// http://www.reddit.com/r/dailyprogrammer/comments/19mn2d/030413_challenge_121_easy_bytelandian_exchange_1/

fn exchange(k: usize) -> [usize; 3] {
    return [k/2, k/3, k/4];
}

fn group_by() {
    // TODO
}

fn exchange_feeder(real_coins: &mut Vec<usize>, pennies: usize) -> usize {
    if real_coins.len() == 0 {
        pennies
    } else {
        let feed: usize = match real_coins.pop() {
            Some(coin) => coin,
            None => panic!("len() wasn't 0 so this can't actually happen; \
                           if you thought the compiler was smart enough \
                           to notice and let you get away with not \
                           destructuring the Option, then you were \
                           apparently mistaken")
        };
        let change: [usize; 3] = exchange(feed);
        0 // TODO
    }
}

#[test]
fn test_exchange() {
    assert_eq!(exchange(7), [3, 2, 1]);
}
