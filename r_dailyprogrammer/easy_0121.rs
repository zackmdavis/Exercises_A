// http://www.reddit.com/r/dailyprogrammer/comments/19mn2d/030413_challenge_121_easy_bytelandian_exchange_1/

fn exchange(k: usize) -> [usize; 3] {
    return [k/2, k/3, k/4];
}

fn exchange_feeder(real_coins: Vec<usize>, pennies: usize) {
    // TODO
}

#[test]
fn test_exchange() {
    assert_eq!(exchange(7), [3, 2, 1]);
}
