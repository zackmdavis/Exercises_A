// http://www.reddit.com/r/dailyprogrammer/
// comments/2zyipu/20150323_challenge_207_easy_bioinformatics_1_dna/

#[derive(PartialEq, Debug, Copy, Clone)]
enum Base {
    A, T,
    C, G
}

struct Codon(Base, Base, Base);

enum AminoAcid {
    Ala, Arg, Asn, Asp, Asx, Cys, Glu, Gln, Glx, Gly, His,
    Ile, Leu, Lys, Met, Phe, Pro, Ser, Thr, Trp, Tyr, Val
}

fn base_to_complement(base: Base) -> Base {
    if base == Base::A {
        Base::T
    } else if base == Base::T {
        Base::A
    } else if base == Base::C {
        Base::G
    } else if base == Base::G {
        Base::C
    } else {
        panic!("This has never happened before")
    }
}

fn match_dna(strand: Vec<Base>) -> Vec<Base> {
    let mut complement: Vec<Base> = Vec::new();
    for &base in strand.iter() {
        complement.push(base_to_complement(base));
    }
    complement
}

#[test]
fn test_match_dna() {
    let strand = vec![Base::A, Base::A, Base::T, Base::G, Base::C,
                      Base::C, Base::T, Base::A, Base::T, Base::G,
                      Base::G, Base::C];
    let expected = vec![Base::T, Base::T, Base::A, Base::C, Base::G,
                        Base::G, Base::A, Base::T, Base::A, Base::C,
                        Base::C, Base::G];
    assert_eq!(match_dna(strand), expected);
}
