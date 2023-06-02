fn main() {
    #[cfg(feature = "generate-bindings")]
    generate_bindings();
}

#[cfg(feature = "generate-bindings")]
fn generate_bindings() {
    use std::path::PathBuf;

    let header_path = PathBuf::from("include").join("lib_core.h");
    let output_path = PathBuf::from("src").join("bindings.rs");

    bindgen::Builder::default()
        .header(header_path.display().to_string())
        .allowlist_file(header_path.display().to_string())
        .default_enum_style(bindgen::EnumVariation::Rust {
            non_exhaustive: false,
        })
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .size_t_is_usize(true)
        .generate()
        .unwrap()
        .write_to_file(output_path)
        .unwrap();

    println!("cargo:rerun-if-changed=include");
}
