// chemical formulæ parser

// https://www.reddit.com/r/dailyprogrammer/comments/6eerfk/20170531_challenge_317_intermediate_counting/

// What _do_ I remember about parsers??


// XXX TODO make this compile, solve problem, think deeply about my life

use std::collections::HashMap;


struct MultiElementBuilder {
    symbol: String,
    multiplier: String
}

impl MultiElementBuilder {
    fn cash_out(self, inventory: &mut HashMap<String, usize>) {
        let count = inventory.entry(self.symbol)
            .get().unwrap_or_else(|v| v.insert(0));
        *count += self.multiplier.parse();
    }

}

fn chemparse(formula: &str) -> Result<HashMap<String, usize>, Error> {
    let inventory = HashMap::new();
    for character in formula {
        // ...
    }
    Ok(inventory)
}


#[test]
fn concerning_chrysoeriol_or_maybe_diosmetin() {
    let mut expectations = vec![("C", 6), ("H", 12), ("O", 6)];
    let mut expected: HashMap<String, usize> = HashMap::with_capacity(3);
    for (element, count) in expectations.drain(..) {
        expected.insert(element.to_owned(), count);
    }
    assert_eq!(chemparse("C6H12O6").unwrap(), expected);
}
