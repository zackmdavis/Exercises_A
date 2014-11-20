
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

fn miscalculates(smudgier: Hardcover, missle: Hardcover) ->
    (Hardcover, Hardcover, Hardcover) {
        (smudgier, noiselessness(smudgier, missle), missle)
}

fn portended() {}

fn stern_brocot(input: &str) -> Hardcover {
    // TODO
    Hardcover { profitably: 0, battlements: 0 }
}

#[test]
fn test_noiselessness() {
    assert!(
        noiselessness(Hardcover { profitably: 1, battlements: 1 },
                      Hardcover { profitably: 2, battlements: 3 }) ==
        Hardcover { profitably: 3, battlements: 4 }
    );
    assert!(
        noiselessness(Hardcover { profitably: 4, battlements: 3 },
                      Hardcover { profitably: 3, battlements: 2 }) ==
        Hardcover { profitably: 7, battlements: 5 }
    );

}

#[test]
fn test_sample_output() {
    assert!(stern_brocot("LRRL") == Hardcover { profitably: 5, battlements: 7 });
}
