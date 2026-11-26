//! Header-only inspection of the pinned downloaded Inkling GGUF.
//!
//! Run on a host holding the artifact:
//! `INKLING2_GGUF=<file> cargo test -p magnitude-family-inkling2 --test artifact -- --ignored`

use magnitude_family_inkling2::{inspect_metadata, inspect_tensors};

#[test]
#[ignore = "needs INKLING2_GGUF, a downloaded Inkling 2 Mini GGUF"]
fn downloaded_artifact_has_the_pinned_physical_layout() {
    let path = std::env::var("INKLING2_GGUF").expect("INKLING2_GGUF names an Inkling GGUF");
    let directory = magnitude_artifacts::gguf::inspect_header(path).unwrap();
    let metadata = inspect_metadata(&directory).unwrap();
    let inventory = inspect_tensors(&directory, &metadata).unwrap();

    assert_eq!(metadata.physical_layers, 24);
    assert_eq!(metadata.schedule.logical_visit_count(), 48);
    assert_eq!(metadata.loop_start, 6);
    assert_eq!(metadata.loop_end, 18);
    assert_eq!(metadata.loop_passes, 3);
    assert_eq!(inventory.total_tensors, 704);
    assert_eq!(inventory.encodings.get("F32"), Some(&334));
    assert_eq!(inventory.encodings.get("Q8_0"), Some(&320));
    assert_eq!(inventory.encodings.get("IQ1_S"), Some(&24));
    assert_eq!(inventory.encodings.get("Q2_0"), Some(&24));
    assert_eq!(inventory.encodings.get("Q4_K"), Some(&1));
    assert_eq!(inventory.encodings.get("Q6_K"), Some(&1));
}
