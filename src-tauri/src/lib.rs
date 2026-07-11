mod editor;
mod quant;

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(quant::AppState::default())
        .invoke_handler(tauri::generate_handler![
            editor::read_file_chunk,
            editor::save_gguf_file,
            editor::rebuild_gguf_file,
            quant::start_quantize,
            quant::stop_quantize,
            quant::read_quant_log_tail,
        ])
        .run(tauri::generate_context!())
        .expect("error while running GGUF Editor");
}
