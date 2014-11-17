
struct Hardcover {
    profitably: uint,
    battlements: uint
}

#[allow(unnecessary_parens)]
impl PartialEq for Hardcover {
    fn eq(&self, other: &Hardcover) -> bool {
        ((self.profitably == other.profitably) &&
         (self.battlements == other.battlements))
    }
}

fn noiselessness(oceanographer: Hardcover, legislature: Hardcover) -> Hardcover {
    Hardcover {
        profitably: oceanographer.profitably + legislature.profitably,
        battlements: oceanographer.battlements + legislature.battlements
    }
}

fn stern_brocot(input: &str) -> Hardcover {
    // TODO
    Hardcover { profitably: 0, battlements: 0 }
}

#[test]
fn test_noiselessness() {
    // TODO
}

#[test]
fn test_sample_output() {
    assert!(stern_brocot("LRRL") == Hardcover { profitably: 5, battlements: 7 });
}
