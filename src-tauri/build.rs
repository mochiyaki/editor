fn main() {
    // Expose the target triple so quant.rs can find triple-suffixed sidecar
    // binaries (e.g. quantizer-x86_64-pc-windows-msvc.exe).
    println!(
        "cargo:rustc-env=TARGET_TRIPLE={}",
        std::env::var("TARGET").unwrap_or_default()
    );
    tauri_build::build()
}
