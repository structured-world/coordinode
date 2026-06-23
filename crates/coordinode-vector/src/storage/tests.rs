use super::*;
use std::collections::HashMap;
use std::sync::Mutex;

/// Test backend: stores f32 in RAM under (label, property, node) keys.
pub struct InMemoryVectorTier {
    f32_store: Mutex<HashMap<(u32, u32, u64), Vec<f32>>>,
}

impl InMemoryVectorTier {
    pub fn new() -> Self {
        Self {
            f32_store: Mutex::new(HashMap::new()),
        }
    }
}

impl VectorTierStorage for InMemoryVectorTier {
    fn put_f32(&self, l: u32, p: u32, n: u64, v: &[f32]) -> Result<(), VectorTierError> {
        self.f32_store
            .lock()
            .map_err(|e| format!("poisoned: {e}"))?
            .insert((l, p, n), v.to_vec());
        Ok(())
    }

    fn multi_get_f32(
        &self,
        l: u32,
        p: u32,
        ids: &[u64],
    ) -> Result<Vec<Option<Vec<f32>>>, VectorTierError> {
        let g = self
            .f32_store
            .lock()
            .map_err(|e| format!("poisoned: {e}"))?;
        Ok(ids.iter().map(|&n| g.get(&(l, p, n)).cloned()).collect())
    }
}

#[test]
fn handle_routes_to_backend() {
    let backend = Arc::new(InMemoryVectorTier::new());
    let h = VectorTierHandle::new(backend.clone(), 7, 13);
    h.put_f32(99, &[1.0, 2.0, 3.0]).unwrap();

    let got_f32 = h.multi_get_f32(&[99, 100]).unwrap();
    assert_eq!(got_f32[0].as_deref(), Some(&[1.0, 2.0, 3.0][..]));
    assert!(got_f32[1].is_none());
}

#[test]
fn distinct_label_property_pairs_dont_collide() {
    let backend = Arc::new(InMemoryVectorTier::new());
    let h1 = VectorTierHandle::new(backend.clone(), 1, 1);
    let h2 = VectorTierHandle::new(backend.clone(), 1, 2);
    let h3 = VectorTierHandle::new(backend.clone(), 2, 1);

    h1.put_f32(42, &[1.0]).unwrap();
    h2.put_f32(42, &[2.0]).unwrap();
    h3.put_f32(42, &[3.0]).unwrap();

    assert_eq!(
        h1.multi_get_f32(&[42]).unwrap()[0].as_deref(),
        Some(&[1.0][..])
    );
    assert_eq!(
        h2.multi_get_f32(&[42]).unwrap()[0].as_deref(),
        Some(&[2.0][..])
    );
    assert_eq!(
        h3.multi_get_f32(&[42]).unwrap()[0].as_deref(),
        Some(&[3.0][..])
    );
}
