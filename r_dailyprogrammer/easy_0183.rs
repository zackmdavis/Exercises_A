// www.reddit.com/r/dailyprogrammer/
// comments/2igfj9/10062014_challenge_183_easy_semantic_version_sort/

// Easy #183: Semantic Version Sort

use std::cmp::Ordering;

#[derive(Eq,PartialEq,PartialOrd,Debug)]
struct Version {
    major: usize,
    minor: usize,
    patch: usize,
    label: String
}

impl Ord for Version {
    fn cmp(&self, other: &Version) -> Ordering {
        let self_segments: [usize; 3] = [self.major, self.minor, self.patch];
        let other_segments: [usize; 3] = [other.major, other.minor, other.patch];
        for (&self_segment, &other_segment) in
            self_segments.iter().zip(other_segments.iter())
        {
            if self_segment > other_segment {
                return Ordering::Greater;
            } else if self_segment < other_segment {
                return Ordering::Less;
            } // else {
            //     continue;
            // }
        }
        // TODO: label
        Ordering::Equal
    }
}

fn versionize(version_string: String) -> Version {
    let segments = version_string.split('.');
    Version {major: 1, minor: 2, patch: 3, label: "".to_string()}
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
        assert_eq!(versionize(input), expected);
    }
}
