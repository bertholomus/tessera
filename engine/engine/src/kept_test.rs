use crate::Row;
use crate::kept::Kept;
use crate::lanes::Lanes;

fn row(slot: u32, pos: u32) -> Row {
    Row {
        slot,
        token: pos,
        position: pos,
    }
}

#[test]
fn grow_rows_matches_single() {
    let (slots, context, pages, page) = (4, 128, 32, 16);
    let mut one = Kept::paged(slots, context, pages, page);
    let mut batched = Kept::paged(slots, context, pages, page);
    let mut lanes_one = Lanes::new(page, slots * (context / page), slots, pages);
    let mut lanes_batched = Lanes::new(page, slots * (context / page), slots, pages);
    let (s0, _) = one.admit(&[1, 2], 0).unwrap();
    let (s1, _) = one.admit(&[3, 4], 0).unwrap();
    assert_eq!(batched.admit(&[1, 2], 0).unwrap().0, s0);
    assert_eq!(batched.admit(&[3, 4], 0).unwrap().0, s1);
    let mut rows: Vec<Row> = (0..40).map(|p| row(s0, p)).collect();
    rows.extend((0..20).map(|p| row(s1, p)));
    rows.push(row(s0, 40));
    for r in &rows {
        if let Some(ps) = one.grow(r.slot, r.position as usize + 1).unwrap() {
            lanes_one.set(r.slot, ps);
        }
    }
    batched.grow_rows(&rows, &mut lanes_batched).unwrap();
    assert_eq!(lanes_batched.pending(), lanes_one.pending());
    assert_eq!(batched.totals(), one.totals());
}
