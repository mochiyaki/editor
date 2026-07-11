// Quantizer integration: after the editor saves an edited GGUF (step 1), the
// bundled standalone quantizer binary is spawned to convert tensor precisions
// via --tensor-type-rules into a second, quantized file (step 2). Output is
// captured to a temp log file surfaced by the Logs tab.

use serde::Deserialize;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tauri::{AppHandle, Emitter, Manager};

/// Injected by build.rs — e.g. "x86_64-pc-windows-msvc" or "x86_64-unknown-linux-gnu"
const TARGET_TRIPLE: &str = env!("TARGET_TRIPLE");

// ── State ────────────────────────────────────────────────────────────────────

pub struct ManagedChild {
    pub process: Child,
    pub log_path: PathBuf,
}

#[derive(Default)]
pub struct AppState {
    pub quant_child: Arc<Mutex<Option<ManagedChild>>>,
    // Kept after the child exits so the Logs tab can still show final output.
    pub quant_log_path: Arc<Mutex<Option<PathBuf>>>,
}

fn kill_child(child_state: &Arc<Mutex<Option<ManagedChild>>>) {
    if let Ok(mut guard) = child_state.lock() {
        if let Some(mut child) = guard.take() {
            child.process.kill().ok();
            child.process.wait().ok();
        }
    }
}

// ── Platform helpers ─────────────────────────────────────────────────────────

#[cfg(windows)]
fn suppress_console(cmd: &mut Command) {
    use std::os::windows::process::CommandExt;
    const CREATE_NO_WINDOW: u32 = 0x08000000;
    cmd.creation_flags(CREATE_NO_WINDOW);
}

#[cfg(not(windows))]
fn suppress_console(_: &mut Command) {}

fn binary_name(stem: &str) -> String {
    if cfg!(windows) {
        format!("{stem}.exe")
    } else {
        stem.to_string()
    }
}

/// Plain binary name plus the target-triple-suffixed variant
/// (e.g. `quantizer-x86_64-pc-windows-msvc.exe`) that sidecar builds use.
fn candidate_binary_names(stem: &str) -> Vec<String> {
    let plain = binary_name(stem);
    let with_triple = binary_name(&format!("{stem}-{TARGET_TRIPLE}"));
    if with_triple == plain {
        vec![plain]
    } else {
        vec![plain, with_triple]
    }
}

// ── Binary resolution ────────────────────────────────────────────────────────

/// Search the resource dir, the running exe's directory, and (walking up the
/// tree) `binaries/` directories for the quantizer binary. The dev layout
/// keeps it in `src-tauri/binaries/`.
fn resolve_quantizer(app: &AppHandle) -> Option<PathBuf> {
    let names = candidate_binary_names("quantizer");

    let find_in = |base: &Path| -> Option<PathBuf> {
        names.iter().map(|name| base.join(name)).find(|c| c.exists())
    };
    let find_at = |base: &Path| -> Option<PathBuf> {
        find_in(base).or_else(|| find_in(&base.join("binaries")))
    };

    // 1. resource_dir (production bundle)
    if let Ok(dir) = app.path().resource_dir() {
        if let Some(p) = find_at(&dir) {
            return Some(p);
        }
    }

    // 2. Same directory as the running exe, then walk up checking binaries/
    //    at each level (dev builds run from src-tauri/target/{debug,release}).
    if let Ok(exe) = std::env::current_exe() {
        let mut dir = exe.parent().map(Path::to_path_buf);
        for _ in 0..10 {
            let Some(d) = dir else { break };
            if let Some(p) = find_at(&d) {
                return Some(p);
            }
            dir = d.parent().map(Path::to_path_buf);
        }
    }

    None
}

// ── Spawning ─────────────────────────────────────────────────────────────────

fn summarize_output(text: &str) -> Option<String> {
    let s = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .rev()
        .take(6)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<Vec<_>>()
        .join(" ");
    if s.is_empty() {
        None
    } else if s.chars().count() > 800 {
        Some(format!("{}…", s.chars().take(800).collect::<String>()))
    } else {
        Some(s)
    }
}

fn ensure_output_dir(output_path: &str) -> Result<(), String> {
    if let Some(parent) = Path::new(output_path).parent() {
        if !parent.as_os_str().is_empty() && !parent.exists() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create output directory: {e}"))?;
        }
    }
    Ok(())
}

/// Spawn `binary` with `args`, capturing stdout/stderr to a fresh log file
/// under the system temp dir.
fn spawn_with_args(binary: &Path, args: &[String]) -> Result<ManagedChild, String> {
    let log_path = std::env::temp_dir().join(format!(
        "gguf-editor-quant-{}.log",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0)
    ));

    let mut log_stdout = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&log_path)
        .map_err(|e| format!("Failed to create log file: {e}"))?;

    // Diagnostic header so launch failures are visible in the Logs tab even
    // when the process exits before printing anything.
    let _ = writeln!(log_stdout, "$ {} {}", binary.display(), args.join(" "));
    if let Some(parent) = binary.parent() {
        let _ = writeln!(log_stdout, "cwd: {}", parent.display());
    }
    let _ = writeln!(log_stdout, "----");

    let log_stderr = log_stdout
        .try_clone()
        .map_err(|e| format!("Failed to clone log handle: {e}"))?;

    let mut cmd = Command::new(binary);
    cmd.args(args);

    // Working dir = binary's parent so Windows finds the sibling quantizer DLLs.
    if let Some(parent) = binary.parent() {
        cmd.current_dir(parent);
    }

    cmd.stdout(Stdio::from(log_stdout))
        .stderr(Stdio::from(log_stderr))
        .stdin(Stdio::null());

    suppress_console(&mut cmd);

    let process = cmd
        .spawn()
        .map_err(|e| format!("Failed to spawn {}: {e}", binary.display()))?;

    Ok(ManagedChild { process, log_path })
}

/// Background thread: wait for the quantizer to finish, then emit
/// `quant-complete` / `quant-error`.
fn monitor_quantize(
    app: AppHandle,
    child_state: Arc<Mutex<Option<ManagedChild>>>,
    expected_output: String,
) {
    std::thread::spawn(move || {
        loop {
            std::thread::sleep(Duration::from_millis(500));

            let result: Option<(bool, Option<i32>, String)> = {
                let mut guard = child_state.lock().unwrap();
                if guard.is_none() {
                    break; // killed externally (Stop button)
                }
                let child = guard.as_mut().unwrap();
                match child.process.try_wait() {
                    Ok(None) => None, // still running
                    Ok(Some(status)) => {
                        let log = std::fs::read_to_string(&child.log_path).unwrap_or_default();
                        let success = status.success();
                        *guard = None;
                        Some((success, status.code(), log))
                    }
                    Err(_) => {
                        *guard = None;
                        Some((false, None, String::new()))
                    }
                }
            };

            if let Some((success, code, log)) = result {
                if success {
                    let _ = app.emit(
                        "quant-complete",
                        serde_json::json!({ "output_path": expected_output }),
                    );
                } else {
                    let hint = match code {
                        Some(c) => format!("quantizer exited with an error (exit code {c})."),
                        None => "quantizer exited with an error.".to_string(),
                    };
                    let detail = summarize_output(&log)
                        .map(|m| format!(" {m}"))
                        .unwrap_or_default();
                    let _ = app.emit(
                        "quant-error",
                        serde_json::json!({
                            "error": format!("{hint}{detail} Check the Logs tab for details.")
                        }),
                    );
                }
                break;
            }
        }
    });
}

// ── Tauri commands ───────────────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct QuantizeArgs {
    input_path: String,
    output_path: String,
    #[serde(default)]
    weight_type: Option<String>,
    #[serde(default)]
    tensor_rules: Option<Vec<String>>,
    #[serde(default)]
    threads: Option<u32>,
}

fn build_quantize_args(args: &QuantizeArgs) -> Vec<String> {
    let mut out = vec![
        "-m".to_string(),
        args.input_path.clone(),
        "-o".to_string(),
        args.output_path.clone(),
    ];

    if let Some(weight_type) = &args.weight_type {
        if !weight_type.is_empty() {
            out.push("--type".to_string());
            out.push(weight_type.clone());
        }
    }

    if let Some(rules) = &args.tensor_rules {
        let valid: Vec<&str> = rules
            .iter()
            .map(String::as_str)
            .filter(|r| !r.is_empty() && r.contains('='))
            .collect();
        if !valid.is_empty() {
            out.push("--tensor-type-rules".to_string());
            out.push(valid.join(","));
        }
    }

    if let Some(t) = args.threads {
        if t > 0 {
            out.push("-t".to_string());
            out.push(t.to_string());
        }
    }

    out
}

#[tauri::command]
pub fn start_quantize(
    app: AppHandle,
    state: tauri::State<AppState>,
    args: QuantizeArgs,
) -> Result<(), String> {
    kill_child(&state.quant_child);

    let has_type = args.weight_type.as_deref().is_some_and(|t| !t.is_empty());
    let has_rules = args
        .tensor_rules
        .as_ref()
        .is_some_and(|r| r.iter().any(|rule| rule.contains('=')));
    if !has_type && !has_rules {
        return Err("Nothing to quantize: no weight type or tensor rules given.".to_string());
    }

    if args.input_path.is_empty() || !Path::new(&args.input_path).exists() {
        return Err(format!("Input file not found: {}", args.input_path));
    }
    if Path::new(&args.input_path) == Path::new(&args.output_path) {
        return Err("Input and output paths must be different.".to_string());
    }

    let binary = resolve_quantizer(&app).ok_or_else(|| {
        format!(
            "quantizer binary not found. Looked for '{}' next to the app and in src-tauri/binaries/.",
            binary_name("quantizer")
        )
    })?;

    ensure_output_dir(&args.output_path)?;

    let output_path = args.output_path.clone();
    let child = spawn_with_args(&binary, &build_quantize_args(&args))?;
    *state.quant_log_path.lock().unwrap() = Some(child.log_path.clone());
    *state.quant_child.lock().unwrap() = Some(child);

    monitor_quantize(app, Arc::clone(&state.quant_child), output_path);

    Ok(())
}

#[tauri::command]
pub fn stop_quantize(state: tauri::State<AppState>) -> Result<(), String> {
    kill_child(&state.quant_child);
    Ok(())
}

#[tauri::command]
pub fn read_quant_log_tail(state: tauri::State<AppState>, max_bytes: Option<u64>) -> String {
    let log_path = {
        let guard = match state.quant_log_path.lock() {
            Ok(g) => g,
            Err(_) => return String::new(),
        };
        match guard.as_ref() {
            Some(path) => path.clone(),
            None => return String::new(),
        }
    };

    let limit = max_bytes.unwrap_or(200_000) as usize;
    match std::fs::read(&log_path) {
        Ok(data) => {
            let start = data.len().saturating_sub(limit);
            String::from_utf8_lossy(&data[start..]).into_owned()
        }
        Err(_) => String::new(),
    }
}
