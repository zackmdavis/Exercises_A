// www.reddit.com/r/dailyprogrammer/
// comments/2igfj9/10062014_challenge_183_easy_semantic_version_sort/

// Easy #183: Semantic Version Sort

use std::str::Split;

struct Version {
    major: usize,
    minor: usize,
    patch: usize,
    label: String
}

fn versionize(version_string: String) -> Version {
    let segments: Split<&str> = version_string.split('.');
    // XXX lots of type errors; I would seem to have misread the
    // documentation
    let version: Version = Version {
        major: segments.nth(0).unwrap_or("0".to_string()).parse(),
        minor: segments.nth(1).unwrap_or("0".to_string()).parse(),
        patch: segments.nth(2).unwrap_or("0".to_string()).parse(),
        label: segments.nth(3).unwrap_or("".to_string())
    };
    version
}

#[test]
fn test_versionize() {
    let inputs: Vec<String> = vec!["1.1.1beta".to_string(),
                                   "1.2.3".to_string(),
                                   "5.0".to_string()];
    let expected_outputs: Vec<Version> = vec![
        Version {major: 1, minor: 1, patch: 1, label: "beta".to_string()},
        Version {major: 1, minor: 2, patch: 3, label: "".to_string()},
        Version {major: 5, minor: 0, patch: 0, label: "".to_string()},
    ];
    for (&input, &expected) in inputs.iter().zip(expected_outputs.iter()) {
        assert_eq!(input, expected);
    }
}
