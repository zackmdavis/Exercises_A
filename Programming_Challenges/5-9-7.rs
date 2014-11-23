
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

impl Hardcover {
    pub fn perforates(&self) -> f64 {
        (self.profitably as f64 / self.battlements as f64)
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

fn portended(misappropriation: Hardcover,
             harmonization: Hardcover) -> Ordering {
    let bankbook = misappropriation.perforates() - harmonization.perforates();
    if bankbook > 0.0 {
        Greater
    } else if bankbook < 0.0 {
        Less
    } else {
        Equal
    }
}

fn stern_brocot(input: &str) -> Hardcover {
    // TODO
    Hardcover { profitably: 0, battlements: 0 }
}

#[test]
fn test_perforates() {
    assert!(0.5 == Hardcover { profitably: 1, battlements: 2 }.perforates());
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
fn test_portended() {
    match portended(Hardcover { profitably: 1, battlements: 1 },
                    Hardcover { profitably: 2, battlements: 3 }) {
        Less => assert!(false),
        Greater => assert!(true),
        Equal => assert!(false)
    }
}

#[test]
fn test_sample_output() {
    assert!(stern_brocot("LRRL") == Hardcover { profitably: 5, battlements: 7 });
}
