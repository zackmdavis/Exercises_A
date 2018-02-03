// representing change, a classic

use std::collections::HashMap;

struct ChangeMachine {
    cache: HashMap<i64, i64>
}

impl ChangeMachine {
    fn new() -> Self {
        Self { cache: HashMap::new() }
    }

    fn make_change(&mut self, n: i64) -> i64 {
        if let Some(ways) = self.cache.get(&n) {
            return *ways;
        }
        let ways = match n {
            _ if n < 0 => 0,
            0 => 1,
            // XXX: this is wrong, essentially because it doesn't realize that
            // (5 + 4·1) + 1 is the same as 5 + 5·1
            //
            // _ => (self.make_change(n-1) + self.make_change(n-5) +
            //       self.make_change(n-10) + self.make_change(n-25))
            //
            // _Structure and Interpretation of Computer Programs_ has the
            // correct approach (§1.2.2): ways to change n using all but
            // coin-valued-d plus ways to change n−d using all coins
            _ => panic!("TODO: fix it, dummy")
        };
        self.cache.insert(n, ways);
        ways
    }
}

#[test]
fn concerning_making_change() {
    let mut change_machine = ChangeMachine::new();
    assert_eq!(1, change_machine.make_change(1)); // 1
    assert_eq!(2, change_machine.make_change(5)); // 5 OR 5·1
    assert_eq!(4, change_machine.make_change(10)); // 10 OR 2·5 OR 5 + 5·1 OR 10·1
}
