// GGUF editor backend: raw file access used by the Editor section to parse
// headers in the frontend and to stream-write edited files back to disk.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;
use tauri::{AppHandle, Emitter};

const SAVE_CHUNK_SIZE: usize = 8 * 1024 * 1024;

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ReadChunkResponse {
    file_size: u64,
    bytes: Vec<u8>,
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
struct SaveProgressPayload {
    written: u64,
    total: u64,
    done: bool,
}

fn emit_save_progress(app: &AppHandle, written: u64, total: u64, done: bool) {
    let _ = app.emit(
        "editor-save-progress",
        SaveProgressPayload {
            written,
            total,
            done,
        },
    );
}

/// Reads the first `length` bytes of a file (clamped to the file size) so the
/// frontend can parse the GGUF header without loading multi-GB tensor data.
#[tauri::command]
pub fn read_file_chunk(path: String, length: u64) -> Result<ReadChunkResponse, String> {
    let file_path = Path::new(&path);
    let metadata = file_path.metadata().map_err(|err| err.to_string())?;
    let file_size = metadata.len();
    let slice_len = length.min(file_size);
    let buffer_len =
        usize::try_from(slice_len).map_err(|_| "Requested header chunk is too large".to_string())?;

    let mut file = File::open(file_path).map_err(|err| err.to_string())?;
    let mut bytes = vec![0u8; buffer_len];
    file.read_exact(&mut bytes).map_err(|err| err.to_string())?;

    Ok(ReadChunkResponse { file_size, bytes })
}

/// Header-only save: writes the new (already aligned) header, then streams the
/// original tensor data section unchanged. Only valid when tensors were not
/// added, removed, or reordered.
#[tauri::command]
pub fn save_gguf_file(
    app: AppHandle,
    source_path: String,
    destination_path: String,
    header_bytes: Vec<u8>,
    tensor_data_offset: u64,
) -> Result<(), String> {
    let source = Path::new(&source_path);
    let destination = Path::new(&destination_path);

    let source_meta = source.metadata().map_err(|err| err.to_string())?;
    let source_size = source_meta.len();
    let remaining = source_size.saturating_sub(tensor_data_offset);
    let header_len =
        u64::try_from(header_bytes.len()).map_err(|_| "Header is too large".to_string())?;
    let total = header_len + remaining;

    let mut input = File::open(source).map_err(|err| err.to_string())?;
    input
        .seek(SeekFrom::Start(tensor_data_offset))
        .map_err(|err| err.to_string())?;

    let mut output = File::create(destination).map_err(|err| err.to_string())?;
    output
        .write_all(&header_bytes)
        .map_err(|err| err.to_string())?;

    let mut written = header_len;
    emit_save_progress(&app, written, total, false);

    let mut buffer = vec![0u8; SAVE_CHUNK_SIZE];
    loop {
        let read = input.read(&mut buffer).map_err(|err| err.to_string())?;
        if read == 0 {
            break;
        }
        output
            .write_all(&buffer[..read])
            .map_err(|err| err.to_string())?;
        written += read as u64;
        emit_save_progress(&app, written, total, false);
    }

    output.flush().map_err(|err| err.to_string())?;
    emit_save_progress(&app, total, total, true);
    Ok(())
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ByteSegment {
    path: String,
    offset: u64,
    length: u64,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TensorWritePlan {
    #[serde(default)]
    segments: Vec<ByteSegment>,
    #[serde(default)]
    zero_fill: u64,
}

/// Rebuilds a GGUF file from scratch: writes the (already aligned) header, then
/// streams each tensor's data as a concatenation of byte segments taken from
/// arbitrary source files (plus optional zero fill), padding every tensor to
/// `alignment`. This is what makes real tensor delete/add/merge possible — the
/// header-only save path in `save_gguf_file` can't move tensor data around.
#[tauri::command]
pub fn rebuild_gguf_file(
    app: AppHandle,
    destination_path: String,
    header_bytes: Vec<u8>,
    alignment: u64,
    tensors: Vec<TensorWritePlan>,
) -> Result<(), String> {
    if alignment == 0 {
        return Err("Alignment must be greater than zero".into());
    }

    let mut sources: HashMap<String, File> = HashMap::new();
    let mut open_source = |path: &str| -> Result<(), String> {
        if !sources.contains_key(path) {
            let file = File::open(path).map_err(|err| format!("Failed to open '{path}': {err}"))?;
            sources.insert(path.to_string(), file);
        }
        Ok(())
    };

    let header_len =
        u64::try_from(header_bytes.len()).map_err(|_| "Header is too large".to_string())?;

    let mut total = header_len;
    for plan in &tensors {
        let mut tensor_bytes = plan.zero_fill;
        for segment in &plan.segments {
            open_source(&segment.path)?;
            tensor_bytes += segment.length;
        }
        total += tensor_bytes.div_ceil(alignment) * alignment;
    }

    let destination = Path::new(&destination_path);
    let mut output = File::create(destination).map_err(|err| err.to_string())?;
    output
        .write_all(&header_bytes)
        .map_err(|err| err.to_string())?;

    let mut written = header_len;
    emit_save_progress(&app, written, total, false);

    let mut buffer = vec![0u8; SAVE_CHUNK_SIZE];
    let zeros = vec![0u8; SAVE_CHUNK_SIZE];

    for plan in &tensors {
        let mut tensor_bytes: u64 = 0;

        for segment in &plan.segments {
            let input = sources
                .get_mut(&segment.path)
                .ok_or_else(|| format!("Source file '{}' not opened", segment.path))?;
            input
                .seek(SeekFrom::Start(segment.offset))
                .map_err(|err| err.to_string())?;

            let mut remaining = segment.length;
            while remaining > 0 {
                let chunk_len = usize::try_from(remaining.min(SAVE_CHUNK_SIZE as u64))
                    .map_err(|_| "Segment chunk too large".to_string())?;
                input
                    .read_exact(&mut buffer[..chunk_len])
                    .map_err(|err| format!("Failed reading from '{}': {err}", segment.path))?;
                output
                    .write_all(&buffer[..chunk_len])
                    .map_err(|err| err.to_string())?;
                remaining -= chunk_len as u64;
                written += chunk_len as u64;
                tensor_bytes += chunk_len as u64;
                emit_save_progress(&app, written, total, false);
            }
        }

        let mut fill = plan.zero_fill;
        while fill > 0 {
            let chunk_len = usize::try_from(fill.min(SAVE_CHUNK_SIZE as u64))
                .map_err(|_| "Zero-fill chunk too large".to_string())?;
            output
                .write_all(&zeros[..chunk_len])
                .map_err(|err| err.to_string())?;
            fill -= chunk_len as u64;
            written += chunk_len as u64;
            tensor_bytes += chunk_len as u64;
            emit_save_progress(&app, written, total, false);
        }

        let padded = tensor_bytes.div_ceil(alignment) * alignment;
        let mut pad = padded - tensor_bytes;
        while pad > 0 {
            let chunk_len = usize::try_from(pad.min(SAVE_CHUNK_SIZE as u64))
                .map_err(|_| "Padding chunk too large".to_string())?;
            output
                .write_all(&zeros[..chunk_len])
                .map_err(|err| err.to_string())?;
            pad -= chunk_len as u64;
            written += chunk_len as u64;
        }
        emit_save_progress(&app, written, total, false);
    }

    output.flush().map_err(|err| err.to_string())?;
    emit_save_progress(&app, total, total, true);
    Ok(())
}
