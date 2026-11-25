use crate::Lanes;

#[test]
fn tracks_dirty_ranges() {
    let mut leader = Lanes::new(16, 16, 2, 12);
    let mut follower = Lanes::new(16, 16, 2, 12);
    let init = leader.take_dirty();
    assert_eq!(init.len(), 2);
    assert!(leader.pending().is_empty());
    let _ = follower.take_dirty();

    leader.set(1, &[0, 0, 7]);
    let patches = leader.pending();
    assert_eq!(patches, vec![(10, &[7][..])]);
    let owned: Vec<(u32, Vec<u32>)> = patches.iter().map(|&(o, v)| (o, v.to_vec())).collect();
    follower.apply(&owned);
    assert_eq!(leader.take_dirty(), vec![(40, 7u32.to_le_bytes().to_vec())]);
    assert_eq!(
        follower.take_dirty(),
        vec![(40, 7u32.to_le_bytes().to_vec())]
    );

    leader.set(1, &[0, 0, 7]);
    assert!(leader.pending().is_empty());
    assert!(leader.take_dirty().is_empty());

    leader.reset();
    assert_eq!(leader.pending(), vec![(10, &[0][..])]);
}
