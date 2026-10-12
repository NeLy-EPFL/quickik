//! Generates the C++ headers for the Diplomat bridge modules in `src/`
//! (including Diplomat's runtime header) into `include/quickik/`, next to the
//! hand-written umbrella header `include/quickik.hpp`. They're written into the
//! source tree rather than `OUT_DIR` so that C++ builds have a fixed include
//! path.

use std::path::Path;

fn main() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let generated_dir = manifest_dir.join("include/quickik");
    // Start from scratch so headers of removed types don't linger.
    if generated_dir.exists() {
        std::fs::remove_dir_all(&generated_dir).unwrap();
    }

    let mut config = diplomat_tool::config::Config::default();
    // Puts every type in namespace `quickik`, including Diplomat's runtime
    // types (as `quickik::diplomat`), so they can't clash with another
    // Diplomat-generated library.
    config.shared_config.lib_name = Some("quickik".to_string());
    let entry = manifest_dir.join("src/lib.rs");
    let docs_urls = Default::default();
    diplomat_tool::r#gen(&entry, "cpp", &generated_dir, &docs_urls, config, true).unwrap();

    println!("cargo:rerun-if-changed=src");
}
