// http://www.reddit.com/r/dailyprogrammer/
// comments/3629st/20150515_challenge_214_hard_chester_the_greedy/

// People in the comments say they used a k-d tree, but let's just try
// the naive thing first.

#[derive(Copy, Clone, Debug)]
struct Site {
    meters_east: f64,
    meters_north: f64,
}

fn distance_separating_sites(here: &Site, there: &Site) -> f64 {
    ((there.meters_east - here.meters_east).powi(2) +
     (there.meters_north - here.meters_north).powi(2)).sqrt()
}

fn pop_nearest_site(here: Site, theres: &mut Vec<Site>) -> (Site, f64) {
    let mut index_of_nearest: usize = 0;
    let mut distance_to_nearest = distance_separating_sites(&here, &theres[0]);
    for i in 1..theres.len() {
        let so_far_away = distance_separating_sites(&here,
                                                    &theres[i]);
        if so_far_away < distance_to_nearest {
            index_of_nearest = i;
            distance_to_nearest = so_far_away;
        }
    }
    (theres.swap_remove(index_of_nearest), distance_to_nearest)
}

fn hungry_puppy_path_length(theres: &mut Vec<Site>) -> f64 {
    let mut odometer: f64 = 0.0;
    let mut here = Site{ meters_east: 0.5, meters_north: 0.5 };
    for _ in 0..theres.len() {
        let was_here = here;
        let (new_here, reward_meters) = pop_nearest_site(was_here, theres);
        println!("{:?}", new_here);
        odometer += reward_meters;
        here = new_here;
    }
    odometer
}

#[test]
fn test_sample_input_1() {
    let mut theres = vec![
        Site{ meters_east: 0.9, meters_north: 0.7 },
        Site{ meters_east: 0.7, meters_north: 0.7 },
        Site{ meters_east: 0.1, meters_north: 0.1 },
        Site{ meters_east: 0.4, meters_north: 0.1 },
        Site{ meters_east: 0.6, meters_north: 0.6 },
        Site{ meters_east: 0.8, meters_north: 0.8 }
    ];
    let answer = hungry_puppy_path_length(&mut theres);
    assert_eq!(answer, 1.6467103925399036);
}

#[test]
fn test_sample_input_2() {
    let mut theres = vec![
        Site{ meters_east: 0.01864427, meters_north: 0.81566369 },
        Site{ meters_east: 0.57103466, meters_north: 0.06437769 },
        Site{ meters_east: 0.08549021, meters_north: 0.71634700 },
        Site{ meters_east: 0.53715504, meters_north: 0.98376388 },
        Site{ meters_east: 0.61774160, meters_north: 0.24703295 },
        Site{ meters_east: 0.68143001, meters_north: 0.00933313 },
        Site{ meters_east: 0.97813131, meters_north: 0.67607706 },
        Site{ meters_east: 0.18616497, meters_north: 0.49921984 },
        Site{ meters_east: 0.05408843, meters_north: 0.02618551 },
        Site{ meters_east: 0.46946944, meters_north: 0.07752532 },
        Site{ meters_east: 0.40383292, meters_north: 0.35266170 },
        Site{ meters_east: 0.43228830, meters_north: 0.28751588 },
        Site{ meters_east: 0.52277396, meters_north: 0.67336453 },
        Site{ meters_east: 0.93041298, meters_north: 0.56418199 },
        Site{ meters_east: 0.89505620, meters_north: 0.44046000 },
        Site{ meters_east: 0.60054037, meters_north: 0.04439641 },
        Site{ meters_east: 0.42916610, meters_north: 0.43868593 },
        Site{ meters_east: 0.66099940, meters_north: 0.66182206 },
        Site{ meters_east: 0.65422651, meters_north: 0.44257725 },
        Site{ meters_east: 0.85856951, meters_north: 0.57848694 },
        Site{ meters_east: 0.08444465, meters_north: 0.90311460 },
        Site{ meters_east: 0.80230740, meters_north: 0.95651611 },
        Site{ meters_east: 0.04776469, meters_north: 0.99818535 },
        Site{ meters_east: 0.84977669, meters_north: 0.28901074 },
        Site{ meters_east: 0.11978126, meters_north: 0.58758278 },
        Site{ meters_east: 0.35860766, meters_north: 0.37649725 },
        Site{ meters_east: 0.11060792, meters_north: 0.78967005 },
        Site{ meters_east: 0.59278925, meters_north: 0.05049235 },
        Site{ meters_east: 0.68610276, meters_north: 0.33434298 },
        Site{ meters_east: 0.17456973, meters_north: 0.46935874 },
        Site{ meters_east: 0.90137425, meters_north: 0.06529296 },
        Site{ meters_east: 0.76283190, meters_north: 0.34241037 },
        Site{ meters_east: 0.14337678, meters_north: 0.63503574 },
        Site{ meters_east: 0.46869394, meters_north: 0.78315369 },
        Site{ meters_east: 0.44552693, meters_north: 0.40363572 },
        Site{ meters_east: 0.12695300, meters_north: 0.17982330 },
        Site{ meters_east: 0.01435011, meters_north: 0.59308364 },
        Site{ meters_east: 0.05179576, meters_north: 0.13489860 },
        Site{ meters_east: 0.83404332, meters_north: 0.25519290 },
        Site{ meters_east: 0.18036782, meters_north: 0.54688275 },
        Site{ meters_east: 0.76714278, meters_north: 0.13424719 },
        Site{ meters_east: 0.12615810, meters_north: 0.39823910 },
        Site{ meters_east: 0.31242284, meters_north: 0.94521748 },
        Site{ meters_east: 0.52096181, meters_north: 0.01891391 },
        Site{ meters_east: 0.89040569, meters_north: 0.01511875 },
        Site{ meters_east: 0.38508481, meters_north: 0.22048573 },
        Site{ meters_east: 0.25517755, meters_north: 0.01951620 },
        Site{ meters_east: 0.64797234, meters_north: 0.08250470 },
        Site{ meters_east: 0.45578978, meters_north: 0.72956846 },
        Site{ meters_east: 0.77260015, meters_north: 0.27414404 },
        Site{ meters_east: 0.48950098, meters_north: 0.56534211 },
        Site{ meters_east: 0.42934190, meters_north: 0.26788002 },
        Site{ meters_east: 0.60255176, meters_north: 0.11878041 },
        Site{ meters_east: 0.01160821, meters_north: 0.96158818 },
        Site{ meters_east: 0.46993993, meters_north: 0.73099519 },
        Site{ meters_east: 0.12613086, meters_north: 0.38784156 },
        Site{ meters_east: 0.85526663, meters_north: 0.79664950 },
        Site{ meters_east: 0.19968545, meters_north: 0.86344196 },
        Site{ meters_east: 0.87952718, meters_north: 0.69685183 },
        Site{ meters_east: 0.59740476, meters_north: 0.02545566 },
        Site{ meters_east: 0.19031424, meters_north: 0.41077753 },
        Site{ meters_east: 0.77731168, meters_north: 0.83955523 },
        Site{ meters_east: 0.94205747, meters_north: 0.97735842 },
        Site{ meters_east: 0.36100251, meters_north: 0.46765545 },
        Site{ meters_east: 0.64584984, meters_north: 0.17985025 },
        Site{ meters_east: 0.91250230, meters_north: 0.42738220 },
        Site{ meters_east: 0.61934165, meters_north: 0.46702397 },
        Site{ meters_east: 0.34651915, meters_north: 0.19205453 },
        Site{ meters_east: 0.20911229, meters_north: 0.54136790 },
        Site{ meters_east: 0.75443902, meters_north: 0.65533315 },
        Site{ meters_east: 0.91697773, meters_north: 0.22234220 },
        Site{ meters_east: 0.36521330, meters_north: 0.16897923 },
        Site{ meters_east: 0.08920742, meters_north: 0.88355655 },
        Site{ meters_east: 0.12334809, meters_north: 0.91527283 },
        Site{ meters_east: 0.13399087, meters_north: 0.70201708 },
        Site{ meters_east: 0.90378200, meters_north: 0.12468040 },
        Site{ meters_east: 0.87986309, meters_north: 0.55180819 },
        Site{ meters_east: 0.24460918, meters_north: 0.02405143 },
        Site{ meters_east: 0.69175089, meters_north: 0.98939611 },
        Site{ meters_east: 0.86760018, meters_north: 0.63574170 },
        Site{ meters_east: 0.04127189, meters_north: 0.26172388 },
        Site{ meters_east: 0.06765159, meters_north: 0.92155860 },
        Site{ meters_east: 0.92383828, meters_north: 0.51137669 },
        Site{ meters_east: 0.49086954, meters_north: 0.80284762 },
        Site{ meters_east: 0.71457543, meters_north: 0.26436647 },
        Site{ meters_east: 0.79180041, meters_north: 0.36357630 },
        Site{ meters_east: 0.23100252, meters_north: 0.47759742 },
        Site{ meters_east: 0.09887423, meters_north: 0.41375010 },
        Site{ meters_east: 0.44351628, meters_north: 0.06065774 },
        Site{ meters_east: 0.38814185, meters_north: 0.96674508 },
        Site{ meters_east: 0.24125015, meters_north: 0.62844019 },
        Site{ meters_east: 0.58788947, meters_north: 0.90764731 },
        Site{ meters_east: 0.78024072, meters_north: 0.08275363 },
        Site{ meters_east: 0.31216130, meters_north: 0.91772787 },
        Site{ meters_east: 0.91013743, meters_north: 0.61033491 },
        Site{ meters_east: 0.35309287, meters_north: 0.62662153 },
        Site{ meters_east: 0.71253793, meters_north: 0.15539307 },
        Site{ meters_east: 0.40369777, meters_north: 0.45550895 },
        Site{ meters_east: 0.00835701, meters_north: 0.03116160 },
        Site{ meters_east: 0.38856060, meters_north: 0.07434118 },
    ];
    let answer = hungry_puppy_path_length(&mut theres);
    assert_eq!(answer, 9.127777855837017);
}
