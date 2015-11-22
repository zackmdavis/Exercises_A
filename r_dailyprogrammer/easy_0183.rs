// https://www.reddit.com/r/dailyprogrammer/comments/2igfj9/10062014_challenge_183_easy_semantic_version_sort/

// Easy #183: Semantic Version Sort

use std::cmp::Ordering;
use std::error::Error;


#[derive(Debug, Eq, PartialEq, PartialOrd, Clone)]
struct Version {
    major: u16,
    minor: u16,
    patch: u16,
    label: Option<String>,
    metadata: Option<String>
}

impl Version {
    pub fn new_release(major: u16, minor: u16, patch: u16) -> Self {
        Version { major: major, minor: minor, patch: patch,
                  label: None, metadata: None }
    }

    pub fn new_prerelease(major: u16, minor: u16, patch: u16,
                          label: String) -> Self {
        Version { major: major, minor: minor, patch: patch,
                  label: Some(label), metadata: None }
    }

    pub fn new_build(major: u16, minor: u16, patch: u16,
                     label: String, metadata: String) -> Self {
        Version { major: major, minor: minor, patch: patch,
                  label: Some(label), metadata: Some(metadata) }
    }
}

impl Ord for Version {
    fn cmp(&self, other: &Version) -> Ordering {
        let self_segments: [u16; 3] = [self.major, self.minor, self.patch];
        let other_segments: [u16; 3] = [other.major, other.minor, other.patch];
        for (&self_segment, &other_segment) in self_segments.iter()
            .zip(other_segments.iter()) {
                match self_segment.cmp(&other_segment) {
                    Ordering::Equal => { continue; },
                    compared @ _ => { return compared; }
                }
        }
        self.label.clone().unwrap_or("".to_owned()).cmp(
            &other.label.clone().unwrap_or("".to_owned()))
    }
}


fn versionize(version_string: String) -> Result<Version, Box<Error>> {
    // maybe we should use regexes instead of working this all out
    // manually—oh, but how do you manually link to the Regex crate
    // (with just rustc and this one file, no Cargo)?
    let segments = version_string.split('.').collect::<Vec<_>>();
    let major = try!(segments[0].parse::<u16>());
    let minor = try!(segments[1].parse::<u16>());
    let mut tail = segments[2].split('-');
    let patch = try!(tail.next().unwrap().parse::<u16>());
    let (label, metadata) = match tail.next() {
        Some(extended_label) => {
            let mut extended_label_segments = extended_label.split('+');
            let label = extended_label_segments.next().unwrap().to_owned();
            let metadata_maybe = extended_label_segments.next()
                .map(|m| m.to_owned());
            (Some(label), metadata_maybe)
        },
        None => (None, None)
    };
    Ok(Version { major: major, minor: minor, patch: patch,
                 label: label, metadata: metadata })
}


#[ignore]  // TODO make this pass
#[test]
fn concerning_versionizing() {
    let inputs = vec!["1.1.1beta".to_owned(),
                      "1.2.3".to_owned(),
                      "5.0".to_owned()];
    let expected_outputs = vec![
        Version::new_prerelease(1, 1, 1, "beta".to_owned()),
        Version::new_release(1, 2, 3),
        Version::new_release(5, 0, 0),
    ];
    for (input, expected) in inputs.iter().zip(expected_outputs.iter()) {
        assert_eq!(versionize(input.clone()).unwrap(), *expected);
    }
}

#[ignore] // TODO make this pass
#[test]
fn concerning_version_comparison() {
    assert!(Version::new_release(5, 0, 0) > Version::new_release(4, 9, 10));
    assert!(Version::new_release(5, 0, 0) >
            Version::new_prerelease(5, 0, 0, "rc1".to_owned()));
    assert_eq!(Version::new_prerelease(5, 0, 0, "β".to_owned()),
               Version::new_build(5, 0, 0, "β".to_owned(),
                                  "8ab8581f6-2015-10-27".to_owned()))
}
