import React, { useState, useEffect, useRef, useCallback, useMemo } from "react";
import {
  AlertCircle,
  Braces,
  Check,
  ChevronDown,
  Combine,
  Database,
  FilePen,
  FilePlus,
  FolderOpen,
  GripVertical,
  Import,
  Loader2,
  Moon,
  Plus,
  Redo2,
  RefreshCw,
  Replace,
  ReplaceAll,
  RotateCcw,
  Save,
  Search,
  Square,
  Sun,
  Terminal,
  Trash2,
  Undo2,
  X,
  Zap,
} from "lucide-react";
import {
  buildGGUFHeader,
  buildGGUFHeaderWithTensors,
  formatValue,
  GGUFValueType,
  GGUFValueTypeName,
  parseGGUF,
  quantizationName,
  resolveAlignment,
  tensorByteSize,
  typeTraits,
  ZERO_TENSOR_DTYPES,
  type FinalTensor,
  type GGUFArrayValue,
  type NewMetaRow,
  type ParsedGGUF,
} from "./ggufParser";

// ─── Types ───────────────────────────────────────────────────────────────────

type Theme = "dark" | "light";
type ViewId = "editor" | "logs";
type View = "landing" | "loading" | "data";

type QuantStatus = "idle" | "running" | "completed" | "error";

// One reconciled entry of the quantize plan: an existing (non-deleted,
// non-merged-away) tensor whose precision the user changed via the Precision
// dropdown, resolved to its final name in the edited file.
interface QuantPlanEntry {
  key: string;
  name: string;
  from: string;
  to: string;
}

interface TensorSegment {
  path: string;
  offset: number;
  length: number;
}

interface TensorWritePlan {
  segments: TensorSegment[];
  zeroFill: number;
}

type AddedTensorSource =
  | { kind: "external"; path: string; offset: number }
  | { kind: "zeros" };

interface AddedTensor {
  id: number;
  name: string;
  shape: number[];
  dtype: number;
  size: number;
  source: AddedTensorSource;
}

interface TensorMerge {
  id: number;
  name: string;
  dtype: number;
  shape: number[];
  size: number;
  parts: number[];
}

// Row keys identify every entry of the unified tensor list: "t:<index>" for
// tensors from the opened file, "m:<id>" for merges, "a:<id>" for added ones.
type DisplayRow =
  | { kind: "orig"; key: string; index: number; name: string; shape: string; dtype: string }
  | { kind: "merge"; key: string; mergeIndex: number; merge: TensorMerge }
  | { kind: "added"; key: string; addedIndex: number; added: AddedTensor };

// One planned rename produced by the tensor Find & Replace drawer.
interface FindReplaceTarget {
  scope: "orig" | "merge" | "added";
  id: number;
  from: string;
  to: string;
}

interface ImportModalTensor {
  name: string;
  shape: number[];
  dtype: number;
  size: number | null;
  absOffset: number;
  checked: boolean;
}

interface ImportModalState {
  path: string;
  fileName: string;
  tensors: ImportModalTensor[];
  filter: string;
}

interface ZeroModalState {
  name: string;
  dtype: number;
  shape: string;
}

interface SaveOverlayState {
  active: boolean;
  progress: number;
  label: string;
}

interface EditSnapshot {
  editedMetadata: Record<string, string>;
  deletedMetaKeys: Set<string>;
  newMetaRows: NewMetaRow[];
  editedTensorNames: Record<number, string>;
  deletedTensors: Set<number>;
  selectedTensors: Set<number>;
  addedTensors: AddedTensor[];
  merges: TensorMerge[];
  tensorOrder: string[] | null;
  precisionOverrides: Record<string, string>;
}

// ─── Constants ────────────────────────────────────────────────────────────────

const HEADER_READ_INITIAL = 4 * 1024 * 1024;
const HEADER_READ_MAX = 1024 * 1024 * 1024;
const MAX_ARRAY_ELEMENTS = 30;

// Scalar GGUF type ids, plus 100 + elemType pseudo-ids for array rows.
const EDITABLE_TYPES: [number, string][] = [
  [0, "UINT8"],
  [1, "INT8"],
  [2, "UINT16"],
  [3, "INT16"],
  [4, "UINT32"],
  [5, "INT32"],
  [6, "FLOAT32"],
  [7, "BOOL"],
  [8, "STRING"],
  [10, "UINT64"],
  [11, "INT64"],
  [12, "FLOAT64"],
  [104, "[UINT32]"],
  [105, "[INT32]"],
  [106, "[FLOAT32]"],
  [112, "[FLOAT64]"],
  [108, "[STRING]"],
  [110, "[UINT64]"],
];

// Target types the bundled standalone quantizer accepts (its --type /
// --tensor-type-rules values), mapped to their ggml dtype ids. Keep in sync
// with quantizer.cpp's supported-types list.
const QUANT_TARGETS: { id: number; name: string }[] = [
  { id: 0, name: "f32" },
  { id: 1, name: "f16" },
  { id: 30, name: "bf16" },
  { id: 8, name: "q8_0" },
  { id: 2, name: "q4_0" },
  { id: 3, name: "q4_1" },
  { id: 6, name: "q5_0" },
  { id: 7, name: "q5_1" },
  { id: 41, name: "q1_0" },
  { id: 10, name: "q2_k" },
  { id: 11, name: "q3_k" },
  { id: 12, name: "q4_k" },
  { id: 13, name: "q5_k" },
  { id: 14, name: "q6_k" },
  { id: 19, name: "iq1_s" },
  { id: 29, name: "iq1_m" },
  { id: 16, name: "iq2_xxs" },
  { id: 17, name: "iq2_xs" },
  { id: 22, name: "iq2_s" },
  { id: 18, name: "iq3_xxs" },
  { id: 21, name: "iq3_s" },
  { id: 20, name: "iq4_nl" },
  { id: 23, name: "iq4_xs" },
  { id: 34, name: "tq1_0" },
  { id: 35, name: "tq2_0" },
  { id: 39, name: "mxfp4" },
  { id: 40, name: "nvfp4" },
];

// dtypes the quantizer can read (dequantize) as a conversion source — same
// set as the targets. Tensors of any other dtype keep a static badge.
const QUANT_SOURCE_IDS = new Set(QUANT_TARGETS.map((target) => target.id));

// ─── Helpers ──────────────────────────────────────────────────────────────────

function cn(...classes: (string | boolean | undefined | null)[]): string {
  return classes.filter(Boolean).join(" ");
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function filename(path: string): string {
  if (!path) return "";
  return path.replace(/\\/g, "/").split("/").pop() ?? path;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1048576) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1073741824) return `${(bytes / 1048576).toFixed(1)} MB`;
  return `${(bytes / 1073741824).toFixed(2)} GB`;
}

function uniqueName(base: string, taken: Set<string>): string {
  let name = base;
  let counter = 2;
  while (taken.has(name)) {
    name = `${base}_${counter}`;
    counter += 1;
  }
  return name;
}

function shouldUseTextarea(key: string, type: number, displayValue: string): boolean {
  return (
    type === GGUFValueType.STRING &&
    (key === "tokenizer.chat_template" ||
      displayValue.includes("\n") ||
      displayValue.length > 200)
  );
}

// ─── Small UI pieces ──────────────────────────────────────────────────────────

function TypeBadge({ label }: { label: string }) {
  return (
    <span className="inline-flex text-[10px] font-bold font-mono bg-emerald-100 text-emerald-700 dark:bg-emerald-900/60 dark:text-emerald-300 px-2 py-0.5 rounded-full whitespace-nowrap">
      {label}
    </span>
  );
}

function CountPill({ value }: { value: number }) {
  return (
    <span className="text-[10px] font-bold bg-emerald-100 text-emerald-700 dark:bg-emerald-900/60 dark:text-emerald-300 px-2 py-0.5 rounded-full">
      {value}
    </span>
  );
}

function RowBadge({
  label,
  tone,
  title,
}: {
  label: string;
  tone: "merge" | "add";
  title?: string;
}) {
  return (
    <span
      title={title}
      className={cn(
        "text-xs px-1.5 py-0.5 rounded whitespace-nowrap flex-shrink-0 border",
        tone === "merge"
          ? "bg-sky-100 text-sky-700 border-sky-200 dark:bg-sky-950/60 dark:text-sky-300 dark:border-sky-900"
          : "bg-emerald-100 text-emerald-700 border-emerald-200 dark:bg-emerald-950/60 dark:text-emerald-300 dark:border-emerald-900"
      )}
    >
      {label}
    </span>
  );
}

// Precision pulldown: pick a quantizer target type for one tensor. Only
// offers types whose block size divides the tensor's first dimension, so the
// resulting --tensor-type-rules never hit the quantizer's block-size guard.
function PrecisionSelect({
  currentDtype,
  ne0,
  value,
  disabled,
  onChange,
}: {
  currentDtype: number;
  ne0: number;
  value: string;
  disabled?: boolean;
  onChange: (value: string) => void;
}) {
  const currentLabel = quantizationName(currentDtype);

  const options = QUANT_TARGETS.filter((target) => {
    if (target.id === currentDtype) return false;
    const blockSize = typeTraits(target.id)?.[0] ?? 1;
    return blockSize <= 1 || (ne0 > 0 && ne0 % blockSize === 0);
  });

  if (!QUANT_SOURCE_IDS.has(currentDtype) || options.length === 0) {
    return <TypeBadge label={currentLabel} />;
  }

  return (
    <select
      value={value}
      disabled={disabled}
      onChange={(event) => onChange(event.target.value)}
      title={
        value
          ? `Quantized from ${currentLabel} to ${value.toUpperCase()} after saving (step 2)`
          : "Pick a new precision — converted by the quantizer after saving"
      }
      className={cn(
        "w-full px-1.5 py-1 rounded-lg border text-xs font-mono focus:outline-none cursor-pointer",
        value
          ? "bg-amber-100 border-amber-300 text-amber-700 dark:bg-amber-950/50 dark:border-amber-800 dark:text-amber-300"
          : "bg-zinc-800 border-zinc-700 text-zinc-300",
        disabled && "opacity-40 cursor-not-allowed"
      )}
    >
      <option value="">{currentLabel}</option>
      {options.map((target) => (
        <option key={target.id} value={target.name}>
          {target.name.toUpperCase()}
        </option>
      ))}
    </select>
  );
}

function ModalShell({
  title,
  wide,
  children,
}: {
  title: string;
  wide?: boolean;
  children: React.ReactNode;
}) {
  return (
    <div className="fixed inset-0 z-50 bg-black/70 flex items-center justify-center p-6">
      <div
        className={cn(
          "w-full rounded-xl border border-zinc-800 bg-zinc-900 text-zinc-100 flex flex-col max-h-[85vh] shadow-2xl shadow-black/50",
          wide ? "max-w-2xl" : "max-w-md"
        )}
      >
        <div className="px-5 py-3.5 border-b border-zinc-800 font-semibold text-sm">
          {title}
        </div>
        {children}
      </div>
    </div>
  );
}

// ─── Logs panel ───────────────────────────────────────────────────────────────

// Quantizer output viewer (step 2 of Save & Quantize). Content comes from the
// backend's temp log file via read_quant_log_tail polling.
function LogsPanel({
  logs,
  quantStatus,
  onRefresh,
  onClear,
  onStop,
}: {
  logs: string;
  quantStatus: QuantStatus;
  onRefresh: () => void;
  onClear: () => void;
  onStop: () => void;
}) {
  const [autoScroll, setAutoScroll] = useState(true);
  const logRef = useRef<HTMLPreElement>(null);

  useEffect(() => {
    if (autoScroll && logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [logs, autoScroll]);

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-2">
          <span className="text-xs text-zinc-500">Auto-scroll</span>
          <button
            onClick={() => setAutoScroll((current) => !current)}
            className={cn(
              "relative inline-flex h-4 w-7 items-center rounded-full transition-colors",
              autoScroll ? "bg-emerald-500" : "bg-zinc-700"
            )}
          >
            <span
              className={cn(
                "inline-block h-3 w-3 rounded-full bg-white shadow transition-transform",
                autoScroll ? "translate-x-3.5" : "translate-x-0.5"
              )}
            />
          </button>
        </div>
        <button
          onClick={onRefresh}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-zinc-800 hover:bg-zinc-700 text-zinc-400 hover:text-zinc-200 text-xs transition-colors"
        >
          <RefreshCw size={11} className={quantStatus === "running" ? "animate-spin" : ""} />
          Refresh
        </button>
        <button
          onClick={onClear}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-zinc-800 hover:bg-zinc-700 text-zinc-400 hover:text-zinc-200 text-xs transition-colors"
        >
          <Trash2 size={11} />
          Clear
        </button>
        {quantStatus === "running" && (
          <>
            <button
              onClick={onStop}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-rose-900/60 hover:bg-rose-800/80 text-rose-300 text-xs transition-colors"
            >
              <Square size={10} />
              Stop
            </button>
            <div className="flex items-center gap-1.5">
              <div className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
              <span className="text-xs text-emerald-400">Quantizing…</span>
            </div>
          </>
        )}
        {quantStatus === "completed" && (
          <span className="text-xs text-emerald-400">Quantization complete</span>
        )}
        {quantStatus === "error" && (
          <span className="text-xs text-rose-400">Quantization failed</span>
        )}
      </div>
      <pre
        ref={logRef}
        className="w-full overflow-auto rounded-xl bg-zinc-900 border border-zinc-800 p-4 text-xs font-mono text-zinc-300 leading-relaxed whitespace-pre-wrap"
        style={{ height: "calc(100vh - 250px)" }}
      >
        {logs || (
          <span className="text-zinc-600 italic">
            {quantStatus === "running"
              ? "Waiting for output…"
              : "No log output yet. Save with a precision change to start a quantize job."}
          </span>
        )}
      </pre>
    </div>
  );
}

// ─── Main App ─────────────────────────────────────────────────────────────────

export function EditorApp({
  theme,
  visible,
  onToggleTheme,
}: {
  theme: Theme;
  visible: boolean;
  onToggleTheme?: () => void;
}) {
  // Metadata and tensors live on one page ("editor"). The Logs view only
  // becomes reachable once a quantization has been initiated this session.
  const [activeView, setActiveView] = useState<ViewId>("editor");
  const [logsAvailable, setLogsAvailable] = useState(false);
  const [view, setView] = useState<View>("landing");

  const [filePath, setFilePath] = useState("");
  const [fileName, setFileName] = useState("");
  const [fileSize, setFileSize] = useState(0);
  const [parsedData, setParsedData] = useState<ParsedGGUF | null>(null);
  const [error, setError] = useState("");
  const [search, setSearch] = useState("");
  const [toast, setToast] = useState("");
  const [dropActive, setDropActive] = useState(false);
  const [saveOverlay, setSaveOverlay] = useState<SaveOverlayState>({
    active: false,
    progress: 0,
    label: "Starting…",
  });

  const [editedMetadata, setEditedMetadata] = useState<Record<string, string>>({});
  const [deletedMetaKeys, setDeletedMetaKeys] = useState<Set<string>>(() => new Set());
  const [newMetaRows, setNewMetaRows] = useState<NewMetaRow[]>([]);
  const [editedTensorNames, setEditedTensorNames] = useState<Record<number, string>>({});
  const [deletedTensors, setDeletedTensors] = useState<Set<number>>(() => new Set());
  const [selectedTensors, setSelectedTensors] = useState<Set<number>>(() => new Set());
  const [addedTensors, setAddedTensors] = useState<AddedTensor[]>([]);
  const [merges, setMerges] = useState<TensorMerge[]>([]);
  const [importModal, setImportModal] = useState<ImportModalState | null>(null);
  const [zeroModal, setZeroModal] = useState<ZeroModalState | null>(null);
  // Pending precision changes, keyed by row key ("t:<i>", "m:<id>", "a:<id>"),
  // valued with a quantizer type string (e.g. "q4_k"). Applied in step 2.
  const [precisionOverrides, setPrecisionOverrides] = useState<Record<string, string>>({});
  const [quantStatus, setQuantStatus] = useState<QuantStatus>("idle");
  const [quantLogs, setQuantLogs] = useState("");
  const [quantModalOpen, setQuantModalOpen] = useState(false);
  const [frOpen, setFrOpen] = useState(false);
  const [frFind, setFrFind] = useState("");
  const [frReplace, setFrReplace] = useState("");
  const [frMatchCase, setFrMatchCase] = useState(false);
  const [frRegex, setFrRegex] = useState(false);

  // User-defined tensor order as a list of row keys; null = original order.
  const [tensorOrder, setTensorOrder] = useState<string[] | null>(null);
  const [dragKey, setDragKey] = useState<string | null>(null);
  const [dragOverKey, setDragOverKey] = useState<string | null>(null);
  const rowIdRef = useRef(0);
  const mainRef = useRef<HTMLElement | null>(null);

  const toastTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const rootRef = useRef<HTMLDivElement | null>(null);
  const visibleRef = useRef(visible);
  visibleRef.current = visible;
  const undoStackRef = useRef<EditSnapshot[]>([]);
  const redoStackRef = useRef<EditSnapshot[]>([]);
  const [historyRevision, setHistoryRevision] = useState(0);

  const isDark = theme === "dark";

  // ── derived state ──

  const mergedSourceIndices = useMemo(() => {
    const indices = new Set<number>();
    merges.forEach((merge) => merge.parts.forEach((index) => indices.add(index)));
    return indices;
  }, [merges]);

  const defaultKeys = useMemo(() => {
    if (!parsedData) return [] as string[];
    return [
      ...parsedData.tensorInfos.map((_, index) => `t:${index}`),
      ...merges.map((merge) => `m:${merge.id}`),
      ...addedTensors.map((added) => `a:${added.id}`),
    ];
  }, [parsedData, merges, addedTensors]);

  // Reconcile the saved order with the current row set: rows removed since the
  // last drag are dropped, rows created since are appended in default order.
  const orderedKeys = useMemo(() => {
    if (!tensorOrder) return defaultKeys;
    const valid = new Set(defaultKeys);
    const kept = tensorOrder.filter((key) => valid.has(key));
    const keptSet = new Set(kept);
    return [...kept, ...defaultKeys.filter((key) => !keptSet.has(key))];
  }, [defaultKeys, tensorOrder]);

  const orderChanged = orderedKeys.some((key, index) => key !== defaultKeys[index]);

  const structuralChanges =
    deletedTensors.size > 0 || addedTensors.length > 0 || merges.length > 0 || orderChanged;
  const hasPendingEdits =
    structuralChanges ||
    Object.keys(editedMetadata).length > 0 ||
    Object.keys(editedTensorNames).length > 0 ||
    Object.keys(precisionOverrides).length > 0 ||
    deletedMetaKeys.size > 0 ||
    newMetaRows.length > 0;
  const canUndo = historyRevision >= 0 && undoStackRef.current.length > 0;
  const canRedo = historyRevision >= 0 && redoStackRef.current.length > 0;

  const lcSearch = search.trim().toLowerCase();

  const filteredMetadata = useMemo(() => {
    if (!parsedData) return [];
    const rows: { key: string; type: number; displayValue: string; typeLabel: string }[] = [];
    Object.entries(parsedData.metadata).forEach(([key, entry]) => {
      const displayValue =
        editedMetadata[key] ?? formatValue(entry.type, entry.value, MAX_ARRAY_ELEMENTS);
      if (
        lcSearch &&
        !key.toLowerCase().includes(lcSearch) &&
        !displayValue.toLowerCase().includes(lcSearch)
      ) {
        return;
      }
      const typeLabel =
        entry.type === GGUFValueType.ARRAY &&
        (entry.value as GGUFArrayValue)?._isArray
          ? `[${GGUFValueTypeName[(entry.value as GGUFArrayValue).elemType] ?? (entry.value as GGUFArrayValue).elemType}]`
          : GGUFValueTypeName[entry.type] ?? String(entry.type);
      rows.push({ key, type: entry.type, displayValue, typeLabel });
    });
    return rows;
  }, [parsedData, editedMetadata, lcSearch]);

  const displayRows = useMemo(() => {
    if (!parsedData) return [] as DisplayRow[];
    const mergesById = new Map(merges.map((merge, mergeIndex) => [merge.id, { merge, mergeIndex }]));
    const addedById = new Map(addedTensors.map((added, addedIndex) => [added.id, { added, addedIndex }]));
    const rows: DisplayRow[] = [];

    orderedKeys.forEach((key) => {
      const id = Number(key.slice(2));
      if (key.startsWith("t:")) {
        const tensor = parsedData.tensorInfos[id];
        if (!tensor) return;
        const dtype = quantizationName(tensor.dtype);
        const shape = tensor.shape.join(" × ") || "(scalar)";
        const name = editedTensorNames[id] ?? tensor.name;
        if (
          !lcSearch ||
          name.toLowerCase().includes(lcSearch) ||
          dtype.toLowerCase().includes(lcSearch) ||
          shape.toLowerCase().includes(lcSearch)
        ) {
          rows.push({ kind: "orig", key, index: id, name, shape, dtype });
        }
      } else if (key.startsWith("m:")) {
        const entry = mergesById.get(id);
        if (!entry) return;
        if (!lcSearch || entry.merge.name.toLowerCase().includes(lcSearch)) {
          rows.push({ kind: "merge", key, ...entry });
        }
      } else {
        const entry = addedById.get(id);
        if (!entry) return;
        if (!lcSearch || entry.added.name.toLowerCase().includes(lcSearch)) {
          rows.push({ kind: "added", key, ...entry });
        }
      }
    });
    return rows;
  }, [parsedData, orderedKeys, merges, addedTensors, editedTensorNames, lcSearch]);

  // Tensor-name Find & Replace: a bulk-rename tool, deliberately independent
  // of the search box above (which only filters what's shown). Operates on
  // every renamable tensor in the file regardless of the current filter.
  const frPreview = useMemo(() => {
    const empty = {
      targets: [] as FindReplaceTarget[],
      targetMap: new Map<string, string>(),
      regexError: null as string | null,
      duplicates: [] as string[],
      hasEmptyResult: false,
    };
    if (!parsedData || !frFind) return empty;

    let toName: (name: string) => string | null;
    if (frRegex) {
      let re: RegExp;
      try {
        re = new RegExp(frFind, frMatchCase ? "g" : "gi");
      } catch (err) {
        return { ...empty, regexError: err instanceof Error ? err.message : "Invalid pattern" };
      }
      toName = (name) => {
        re.lastIndex = 0;
        if (!re.test(name)) return null;
        re.lastIndex = 0;
        return name.replace(re, frReplace);
      };
    } else {
      const needle = frMatchCase ? frFind : frFind.toLowerCase();
      const literalRe = new RegExp(escapeRegExp(frFind), frMatchCase ? "g" : "gi");
      toName = (name) => {
        const haystack = frMatchCase ? name : name.toLowerCase();
        if (!haystack.includes(needle)) return null;
        return name.replace(literalRe, frReplace);
      };
    }

    const targets: FindReplaceTarget[] = [];
    const targetMap = new Map<string, string>();

    parsedData.tensorInfos.forEach((tensor, index) => {
      if (deletedTensors.has(index) || mergedSourceIndices.has(index)) return;
      const current = editedTensorNames[index] ?? tensor.name;
      const next = toName(current);
      if (next !== null && next !== current) {
        targets.push({ scope: "orig", id: index, from: current, to: next });
        targetMap.set(`orig:${index}`, next);
      }
    });
    merges.forEach((merge, mergeIndex) => {
      const next = toName(merge.name);
      if (next !== null && next !== merge.name) {
        targets.push({ scope: "merge", id: mergeIndex, from: merge.name, to: next });
        targetMap.set(`merge:${mergeIndex}`, next);
      }
    });
    addedTensors.forEach((added, addedIndex) => {
      const next = toName(added.name);
      if (next !== null && next !== added.name) {
        targets.push({ scope: "added", id: addedIndex, from: added.name, to: next });
        targetMap.set(`added:${addedIndex}`, next);
      }
    });

    const finalNames: string[] = [];
    parsedData.tensorInfos.forEach((tensor, index) => {
      if (deletedTensors.has(index) || mergedSourceIndices.has(index)) return;
      const current = editedTensorNames[index] ?? tensor.name;
      finalNames.push(targetMap.get(`orig:${index}`) ?? current);
    });
    merges.forEach((merge, mergeIndex) => {
      finalNames.push(targetMap.get(`merge:${mergeIndex}`) ?? merge.name);
    });
    addedTensors.forEach((added, addedIndex) => {
      finalNames.push(targetMap.get(`added:${addedIndex}`) ?? added.name);
    });

    const counts = new Map<string, number>();
    finalNames.forEach((name) => counts.set(name, (counts.get(name) ?? 0) + 1));
    const duplicates = [...counts.entries()].filter(([, count]) => count > 1).map(([name]) => name);
    const hasEmptyResult = targets.some((target) => !target.to.trim());

    return { targets, targetMap, regexError: null, duplicates, hasEmptyResult };
  }, [
    parsedData,
    frFind,
    frReplace,
    frMatchCase,
    frRegex,
    editedTensorNames,
    deletedTensors,
    mergedSourceIndices,
    merges,
    addedTensors,
  ]);

  // Reconciled quantize plan: every precision override that still points at a
  // tensor that will exist in the edited file, resolved to its final name.
  // Overrides on deleted / merged-away / removed rows are dropped here, so the
  // rule list sent to the quantizer always matches the saved file.
  const quantPlan = useMemo(() => {
    const entries: QuantPlanEntry[] = [];
    const skipped: string[] = [];
    if (!parsedData) return { entries, skipped };

    const mergesById = new Map(merges.map((merge) => [merge.id, merge]));
    const addedById = new Map(addedTensors.map((added) => [added.id, added]));

    orderedKeys.forEach((key) => {
      const target = precisionOverrides[key];
      if (!target) return;
      const id = Number(key.slice(2));

      let name: string | null = null;
      let dtype = -1;
      if (key.startsWith("t:")) {
        const tensor = parsedData.tensorInfos[id];
        if (!tensor || deletedTensors.has(id) || mergedSourceIndices.has(id)) return;
        name = editedTensorNames[id] ?? tensor.name;
        dtype = tensor.dtype;
      } else if (key.startsWith("m:")) {
        const merge = mergesById.get(id);
        if (!merge) return;
        name = merge.name;
        dtype = merge.dtype;
      } else {
        const added = addedById.get(id);
        if (!added) return;
        name = added.name;
        dtype = added.dtype;
      }

      if (!name || quantizationName(dtype).toLowerCase() === target) return;
      // Rule syntax splits entries on "," and pattern/type on "=" — names
      // containing either can't be expressed as a rule.
      if (name.includes(",") || name.includes("=")) {
        skipped.push(name);
        return;
      }
      entries.push({ key, name, from: quantizationName(dtype), to: target });
    });

    return { entries, skipped };
  }, [
    parsedData,
    orderedKeys,
    precisionOverrides,
    deletedTensors,
    mergedSourceIndices,
    editedTensorNames,
    merges,
    addedTensors,
  ]);

  const frDisabled =
    !frFind ||
    frPreview.targets.length === 0 ||
    !!frPreview.regexError ||
    frPreview.duplicates.length > 0 ||
    frPreview.hasEmptyResult;

  const pendingBadges: string[] = [];
  if (deletedTensors.size > 0) pendingBadges.push(`${deletedTensors.size} deleted`);
  if (merges.length > 0) pendingBadges.push(`${merges.length} merge${merges.length > 1 ? "s" : ""}`);
  if (addedTensors.length > 0) pendingBadges.push(`${addedTensors.length} added`);
  if (orderChanged) pendingBadges.push("reordered");
  if (quantPlan.entries.length > 0)
    pendingBadges.push(
      `${quantPlan.entries.length} precision change${quantPlan.entries.length > 1 ? "s" : ""}`
    );

  // ── infrastructure ──

  function showToast(message: string) {
    setToast(message);
    if (toastTimerRef.current) clearTimeout(toastTimerRef.current);
    toastTimerRef.current = setTimeout(() => setToast(""), 2500);
  }

  function captureEditSnapshot(): EditSnapshot {
    return {
      editedMetadata: { ...editedMetadata },
      deletedMetaKeys: new Set(deletedMetaKeys),
      newMetaRows: newMetaRows.map((row) => ({ ...row })),
      editedTensorNames: { ...editedTensorNames },
      deletedTensors: new Set(deletedTensors),
      selectedTensors: new Set(selectedTensors),
      addedTensors: addedTensors.map((tensor) => ({
        ...tensor,
        shape: [...tensor.shape],
        source: { ...tensor.source },
      })),
      merges: merges.map((merge) => ({
        ...merge,
        shape: [...merge.shape],
        parts: [...merge.parts],
      })),
      tensorOrder: tensorOrder ? [...tensorOrder] : null,
      precisionOverrides: { ...precisionOverrides },
    };
  }

  function restoreEditSnapshot(snapshot: EditSnapshot) {
    setEditedMetadata({ ...snapshot.editedMetadata });
    setDeletedMetaKeys(new Set(snapshot.deletedMetaKeys));
    setNewMetaRows(snapshot.newMetaRows.map((row) => ({ ...row })));
    setEditedTensorNames({ ...snapshot.editedTensorNames });
    setDeletedTensors(new Set(snapshot.deletedTensors));
    setSelectedTensors(new Set(snapshot.selectedTensors));
    setAddedTensors(
      snapshot.addedTensors.map((tensor) => ({
        ...tensor,
        shape: [...tensor.shape],
        source: { ...tensor.source },
      }))
    );
    setMerges(
      snapshot.merges.map((merge) => ({
        ...merge,
        shape: [...merge.shape],
        parts: [...merge.parts],
      }))
    );
    setTensorOrder(snapshot.tensorOrder ? [...snapshot.tensorOrder] : null);
    setPrecisionOverrides({ ...snapshot.precisionOverrides });
    setDragKey(null);
    setDragOverKey(null);
  }

  function recordHistory() {
    undoStackRef.current.push(captureEditSnapshot());
    redoStackRef.current = [];
    setHistoryRevision((revision) => revision + 1);
  }

  function clearHistory() {
    undoStackRef.current = [];
    redoStackRef.current = [];
    setHistoryRevision((revision) => revision + 1);
  }

  function handleUndo() {
    const previous = undoStackRef.current.pop();
    if (!previous) return;
    redoStackRef.current.push(captureEditSnapshot());
    restoreEditSnapshot(previous);
    setError("");
    setHistoryRevision((revision) => revision + 1);
  }

  function handleRedo() {
    const next = redoStackRef.current.pop();
    if (!next) return;
    undoStackRef.current.push(captureEditSnapshot());
    restoreEditSnapshot(next);
    setError("");
    setHistoryRevision((revision) => revision + 1);
  }

  function resetEditState({ clearHistoryStacks = true }: { clearHistoryStacks?: boolean } = {}) {
    setEditedMetadata({});
    setDeletedMetaKeys(new Set());
    setNewMetaRows([]);
    setEditedTensorNames({});
    setDeletedTensors(new Set());
    setSelectedTensors(new Set());
    setAddedTensors([]);
    setMerges([]);
    setTensorOrder(null);
    setPrecisionOverrides({});
    setDragKey(null);
    setDragOverKey(null);
    if (clearHistoryStacks) clearHistory();
  }

  function nextRowId(): number {
    rowIdRef.current += 1;
    return rowIdRef.current;
  }

  // Listen for streaming save progress from the Rust backend.
  useEffect(() => {
    let cleanup: (() => void) | undefined;
    let disposed = false;

    import("@tauri-apps/api/event")
      .then(({ listen }) =>
        listen<{ written: number; total: number; done: boolean }>(
          "editor-save-progress",
          (event) => {
            const payload = event.payload;
            const pct =
              payload.total > 0 ? Math.round((payload.written / payload.total) * 100) : 0;
            setSaveOverlay({
              active: !payload.done,
              progress: payload.done ? 0 : pct,
              label: payload.done
                ? "Starting…"
                : `Writing tensor data… ${pct}% (${formatBytes(payload.written)} / ${formatBytes(payload.total)})`,
            });
          }
        )
      )
      .then((unlisten) => {
        if (disposed) unlisten();
        else cleanup = unlisten;
      })
      .catch(() => {
        // Not running inside Tauri (browser dev mode).
      });

    return () => {
      disposed = true;
      if (toastTimerRef.current) clearTimeout(toastTimerRef.current);
      cleanup?.();
    };
  }, []);

  // Listen for quantizer completion/error events from the Rust backend.
  useEffect(() => {
    const cleanups: (() => void)[] = [];
    let disposed = false;

    (async () => {
      try {
        const { listen } = await import("@tauri-apps/api/event");

        const unlistenComplete = await listen<{ output_path: string }>(
          "quant-complete",
          (event) => {
            setQuantStatus("completed");
            // The editor is still showing the intermediate edited file from
            // step 1 — swap it for the finished quantized output.
            openGGUFPath(event.payload.output_path);
            showToast(`Quantized file created: ${filename(event.payload.output_path)}`);
          }
        );
        if (disposed) {
          unlistenComplete();
          return;
        }
        cleanups.push(unlistenComplete);

        const unlistenError = await listen<{ error: string }>("quant-error", (event) => {
          setQuantStatus("error");
          setError(`Quantization failed: ${event.payload.error}`);
          setLogsAvailable(true);
          setActiveView("logs");
        });
        if (disposed) {
          unlistenError();
          return;
        }
        cleanups.push(unlistenError);
      } catch {
        // Not running inside Tauri (browser dev mode).
      }
    })();

    return () => {
      disposed = true;
      cleanups.forEach((cleanup) => cleanup());
    };
    // showToast only touches refs/setState, and openGGUFPath is stable (its
    // useCallback dependency chain is empty) — safe from a stale closure.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const fetchQuantLogs = useCallback(async () => {
    try {
      const { invoke } = await import("@tauri-apps/api/core");
      const content = await invoke<string>("read_quant_log_tail", { maxBytes: 200000 });
      if (content) setQuantLogs(content);
    } catch {
      // Not running inside Tauri.
    }
  }, []);

  // Poll the quantizer log file while it runs or while the Logs view is open.
  useEffect(() => {
    if (activeView !== "logs" && quantStatus !== "running") return;
    let cancelled = false;
    (async () => {
      while (!cancelled) {
        await fetchQuantLogs();
        await new Promise((resolve) => setTimeout(resolve, 800));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [activeView, quantStatus, fetchQuantLogs]);

  async function handleStopQuantize() {
    try {
      const { invoke } = await import("@tauri-apps/api/core");
      await invoke("stop_quantize");
    } catch {
      // ignore
    }
    setQuantStatus("idle");
  }

  const parseGGUFFromPath = useCallback(async (path: string) => {
    const { invoke } = await import("@tauri-apps/api/core");
    let size = HEADER_READ_INITIAL;

    // The header size is unknown up front: read a chunk, retry with a bigger
    // one whenever the parser runs past the end of the buffer.
    for (;;) {
      const response = await invoke<{ fileSize: number; bytes: number[] }>("read_file_chunk", {
        path,
        length: size,
      });
      const bytes = new Uint8Array(response.bytes);
      const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);

      try {
        return { parsed: parseGGUF(buffer), fileSize: Number(response.fileSize) };
      } catch (err) {
        if (err instanceof RangeError) {
          const sliceEnd = Math.min(size, Number(response.fileSize));
          if (sliceEnd >= Number(response.fileSize))
            throw new Error("File appears truncated or malformed");
          if (size >= HEADER_READ_MAX)
            throw new Error(`GGUF header exceeds ${HEADER_READ_MAX / 1024 / 1024} MB`);
          size = Math.min(size * 2, HEADER_READ_MAX);
          continue;
        }
        throw err;
      }
    }
  }, []);

  const openGGUFPath = useCallback(
    async (selected: string) => {
      if (!selected) return;

      if (!selected.toLowerCase().endsWith(".gguf")) {
        setError("Please select a .gguf file.");
        return;
      }

      setError("");
      setView("loading");
      resetEditState();

      try {
        const { parsed, fileSize: size } = await parseGGUFFromPath(selected);
        setParsedData(parsed);
        setFilePath(selected);
        setFileName(filename(selected));
        setFileSize(size);
        setView("data");
      } catch (err) {
        setError(`Failed to parse GGUF file: ${err instanceof Error ? err.message : err}`);
        setParsedData(null);
        setFilePath("");
        setFileName("");
        setFileSize(0);
        setView("landing");
      }
    },
    [parseGGUFFromPath]
  );

  // Window-level drag & drop: accept a .gguf dropped anywhere on the Editor
  // section. The rect check keeps drops meant for other sections untouched.
  useEffect(() => {
    let cleanup: (() => void) | undefined;
    let disposed = false;

    import("@tauri-apps/api/window")
      .then(({ getCurrentWindow }) =>
        getCurrentWindow().onDragDropEvent(({ payload }) => {
          const insideEditor = (position: { x: number; y: number }) => {
            if (!visibleRef.current) return false;
            const element = rootRef.current;
            if (!element) return false;
            const rect = element.getBoundingClientRect();
            if (rect.width === 0 || rect.height === 0) return false;
            const scale = window.devicePixelRatio || 1;
            const x = position.x / scale;
            const y = position.y / scale;
            return x >= rect.left && x <= rect.right && y >= rect.top && y <= rect.bottom;
          };

          if (payload.type === "enter" || payload.type === "over") {
            setDropActive(insideEditor(payload.position));
            return;
          }

          if (payload.type === "drop") {
            const inside = insideEditor(payload.position);
            setDropActive(false);
            if (!inside) return;
            const ggufPath = payload.paths.find((path) =>
              path.toLowerCase().endsWith(".gguf")
            );
            if (!ggufPath) {
              setError("Drop a .gguf file to open it.");
              return;
            }
            openGGUFPath(ggufPath);
            return;
          }

          setDropActive(false);
        })
      )
      .then((unlisten) => {
        if (disposed) unlisten();
        else cleanup = unlisten;
      })
      .catch(() => {
        // Browser dev mode falls back to the HTML drop handlers below.
      });

    return () => {
      disposed = true;
      cleanup?.();
    };
  }, [openGGUFPath]);

  const handleBrowserDrop = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDropActive(false);
    const file = event.dataTransfer.files[0] as (File & { path?: string }) | undefined;
    const path = file?.path || file?.name;
    if (path) openGGUFPath(path);
  };

  async function handleOpenFile() {
    try {
      const { open } = await import("@tauri-apps/plugin-dialog");
      const selected = await open({
        multiple: false,
        filters: [{ name: "GGUF Model File", extensions: ["gguf"] }],
        title: "Open GGUF File",
      });
      if (!selected || Array.isArray(selected)) return;
      openGGUFPath(selected);
    } catch (err) {
      console.error("File dialog error:", err);
    }
  }

  // ── metadata edits ──

  function updateMetaValue(key: string, value: string) {
    recordHistory();
    setEditedMetadata((current) => ({ ...current, [key]: value }));
  }

  function toggleDeleteMeta(key: string) {
    recordHistory();
    setDeletedMetaKeys((current) => {
      const next = new Set(current);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  }

  function addMetaRow() {
    recordHistory();
    setNewMetaRows((current) => [
      ...current,
      { key: "", value: "", type: GGUFValueType.STRING },
    ]);
  }

  function removeNewMetaRow(index: number) {
    recordHistory();
    setNewMetaRows((current) => current.filter((_, rowIndex) => rowIndex !== index));
  }

  function updateNewMetaRow(index: number, patch: Partial<NewMetaRow>) {
    recordHistory();
    setNewMetaRows((current) =>
      current.map((row, rowIndex) => (rowIndex === index ? { ...row, ...patch } : row))
    );
  }

  // ── tensor edits ──

  function updateTensorName(index: number, value: string) {
    recordHistory();
    setEditedTensorNames((current) => ({ ...current, [index]: value }));
  }

  function updatePrecision(key: string, value: string) {
    recordHistory();
    setPrecisionOverrides((current) => {
      const next = { ...current };
      if (!value) delete next[key];
      else next[key] = value;
      return next;
    });
  }

  function applyFindReplace() {
    if (frDisabled) return;

    recordHistory();

    const origUpdates: Record<number, string> = {};
    frPreview.targets.forEach((target) => {
      if (target.scope === "orig") origUpdates[target.id] = target.to;
    });
    if (Object.keys(origUpdates).length > 0) {
      setEditedTensorNames((current) => ({ ...current, ...origUpdates }));
    }

    if (frPreview.targets.some((target) => target.scope === "merge")) {
      setMerges((current) =>
        current.map((merge, index) => {
          const next = frPreview.targetMap.get(`merge:${index}`);
          return next ? { ...merge, name: next } : merge;
        })
      );
    }

    if (frPreview.targets.some((target) => target.scope === "added")) {
      setAddedTensors((current) =>
        current.map((added, index) => {
          const next = frPreview.targetMap.get(`added:${index}`);
          return next ? { ...added, name: next } : added;
        })
      );
    }

    const count = frPreview.targets.length;
    showToast(`Renamed ${count} tensor${count === 1 ? "" : "s"}`);
  }

  function toggleDeleteTensor(index: number) {
    recordHistory();
    setDeletedTensors((current) => {
      const next = new Set(current);
      if (next.has(index)) next.delete(index);
      else next.add(index);
      return next;
    });
    setSelectedTensors((current) => {
      const next = new Set(current);
      next.delete(index);
      return next;
    });
  }

  function toggleSelectTensor(index: number) {
    setSelectedTensors((current) => {
      const next = new Set(current);
      if (next.has(index)) next.delete(index);
      else next.add(index);
      return next;
    });
  }

  // ── reorder ──

  const canReorder = !lcSearch;

  function moveRow(fromKey: string, toKey: string) {
    if (fromKey === toKey) return;
    const keys = [...orderedKeys];
    const from = keys.indexOf(fromKey);
    const to = keys.indexOf(toKey);
    if (from < 0 || to < 0) return;
    keys.splice(from, 1);
    keys.splice(to, 0, fromKey);
    recordHistory();
    setTensorOrder(keys);
  }

  // Custom mouse-driven row drag. HTML5 drag & drop can't be used here:
  // Tauri's native drag-drop handler (registered above to open dropped .gguf
  // files) swallows the webview's dragover/drop events on Windows, so an
  // in-page drag starts but never delivers a drop.
  useEffect(() => {
    if (!dragKey) return;

    const rowKeyAt = (x: number, y: number): string | null => {
      const element = document.elementFromPoint(x, y);
      const rowElement = element?.closest("[data-row-key]") as HTMLElement | null;
      return rowElement?.dataset.rowKey ?? null;
    };

    const handleMove = (event: MouseEvent) => {
      setDragOverKey(rowKeyAt(event.clientX, event.clientY));
      const main = mainRef.current;
      if (main) {
        const rect = main.getBoundingClientRect();
        if (event.clientY < rect.top + 56) main.scrollTop -= 16;
        else if (event.clientY > rect.bottom - 56) main.scrollTop += 16;
      }
    };

    const handleUp = (event: MouseEvent) => {
      const target = rowKeyAt(event.clientX, event.clientY);
      if (target) moveRow(dragKey, target);
      setDragKey(null);
      setDragOverKey(null);
    };

    const previousUserSelect = document.body.style.userSelect;
    const previousCursor = document.body.style.cursor;
    document.body.style.userSelect = "none";
    document.body.style.cursor = "grabbing";
    window.addEventListener("mousemove", handleMove);
    window.addEventListener("mouseup", handleUp);
    return () => {
      document.body.style.userSelect = previousUserSelect;
      document.body.style.cursor = previousCursor;
      window.removeEventListener("mousemove", handleMove);
      window.removeEventListener("mouseup", handleUp);
    };
    // orderedKeys can't change mid-drag, so the moveRow closure stays valid.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dragKey]);

  function rowDragCls(key: string): string | false {
    if (dragKey === key) return "opacity-40";
    if (!dragKey || dragOverKey !== key) return false;
    const from = orderedKeys.indexOf(dragKey);
    const to = orderedKeys.indexOf(key);
    return to > from
      ? "border-b-2 border-b-emerald-500"
      : "border-t-2 border-t-emerald-500";
  }

  function gripCell(key: string) {
    return (
      <td className="py-2 pr-1">
        <span
          title={canReorder ? "Drag to reorder" : "Clear the search to reorder"}
          onMouseDown={(event) => {
            if (!canReorder || event.button !== 0) return;
            event.preventDefault();
            setDragKey(key);
            setDragOverKey(key);
          }}
          className={cn(
            "inline-flex text-zinc-600",
            canReorder ? "cursor-grab hover:text-zinc-300" : "cursor-not-allowed opacity-40"
          )}
        >
          <GripVertical size={13} />
        </span>
      </td>
    );
  }

  function collectTakenNames(): Set<string> {
    const taken = new Set<string>();
    if (parsedData) {
      parsedData.tensorInfos.forEach((tensor, index) => {
        if (deletedTensors.has(index) || mergedSourceIndices.has(index)) return;
        taken.add(editedTensorNames[index] ?? tensor.name);
      });
    }
    merges.forEach((merge) => taken.add(merge.name));
    addedTensors.forEach((added) => taken.add(added.name));
    return taken;
  }

  // ── merge ──

  function handleMergeSelected() {
    if (!parsedData) return;
    const indices = [...selectedTensors].sort((a, b) => a - b);
    if (indices.length < 2) {
      setError("Select at least two tensors to merge.");
      return;
    }

    const tensors = indices.map((index) => parsedData.tensorInfos[index]);
    const dtype = tensors[0].dtype;
    if (!tensors.every((tensor) => tensor.dtype === dtype)) {
      setError("Merge failed: all selected tensors must have the same precision/type.");
      return;
    }

    const nDims = tensors[0].shape.length;
    if (nDims === 0 || !tensors.every((tensor) => tensor.shape.length === nDims)) {
      setError("Merge failed: all selected tensors must have the same number of dimensions.");
      return;
    }

    for (let d = 0; d < nDims - 1; d += 1) {
      if (!tensors.every((tensor) => tensor.shape[d] === tensors[0].shape[d])) {
        setError(
          "Merge failed: tensors must match on every dimension except the last (merge concatenates along the last axis)."
        );
        return;
      }
    }

    let size = 0;
    try {
      tensors.forEach((tensor) => {
        size += tensorByteSize(tensor.dtype, tensor.shape);
      });
    } catch (err) {
      setError(`Merge failed: ${err instanceof Error ? err.message : err}`);
      return;
    }

    const shape = [...tensors[0].shape];
    shape[nDims - 1] = tensors.reduce((sum, tensor) => sum + tensor.shape[nDims - 1], 0);

    const baseName = (editedTensorNames[indices[0]] ?? tensors[0].name) + ".merged";
    const name = uniqueName(baseName, collectTakenNames());

    recordHistory();
    setMerges((current) => [
      ...current,
      { id: nextRowId(), name, dtype, shape, size, parts: indices },
    ]);
    setSelectedTensors(new Set());
    setError("");
    showToast(`Merged ${indices.length} tensors into "${name}"`);
  }

  function removeMerge(mergeIndex: number) {
    recordHistory();
    setMerges((current) => current.filter((_, index) => index !== mergeIndex));
  }

  function updateMergeName(mergeIndex: number, value: string) {
    recordHistory();
    setMerges((current) =>
      current.map((merge, index) => (index === mergeIndex ? { ...merge, name: value } : merge))
    );
  }

  // ── add (import / zeros) ──

  async function handleImportFromGGUF() {
    try {
      const { open } = await import("@tauri-apps/plugin-dialog");
      const selected = await open({
        multiple: false,
        filters: [{ name: "GGUF Model File", extensions: ["gguf"] }],
        title: "Import Tensors from GGUF",
      });
      if (!selected || Array.isArray(selected)) return;

      const { parsed } = await parseGGUFFromPath(selected);
      const tensors: ImportModalTensor[] = parsed.tensorInfos.map((tensor) => {
        let size: number | null = null;
        try {
          size = tensorByteSize(tensor.dtype, tensor.shape);
        } catch {
          size = null;
        }
        return {
          name: tensor.name,
          shape: tensor.shape,
          dtype: tensor.dtype,
          size,
          absOffset: parsed.tensorDataOffset + Number(tensor.offset),
          checked: false,
        };
      });
      setImportModal({
        path: selected,
        fileName: filename(selected),
        tensors,
        filter: "",
      });
    } catch (err) {
      setError(`Failed to read GGUF file: ${err instanceof Error ? err.message : err}`);
    }
  }

  function confirmImport() {
    if (!importModal) return;
    const taken = collectTakenNames();
    const imported: AddedTensor[] = [];

    importModal.tensors.forEach((tensor) => {
      if (!tensor.checked || tensor.size == null) return;
      const name = uniqueName(tensor.name, taken);
      taken.add(name);
      imported.push({
        id: nextRowId(),
        name,
        shape: tensor.shape,
        dtype: tensor.dtype,
        size: tensor.size,
        source: { kind: "external", path: importModal.path, offset: tensor.absOffset },
      });
    });

    if (imported.length > 0) {
      recordHistory();
      setAddedTensors((current) => [...current, ...imported]);
      showToast(
        `Added ${imported.length} tensor${imported.length > 1 ? "s" : ""} from ${importModal.fileName}`
      );
    }
    setImportModal(null);
  }

  function confirmZeroTensor() {
    if (!zeroModal) return;
    const name = zeroModal.name.trim();
    if (!name) {
      setError("New tensor needs a name.");
      return;
    }
    if (collectTakenNames().has(name)) {
      setError(`A tensor named "${name}" already exists.`);
      return;
    }

    const dims = zeroModal.shape
      .split(/[,×x]/)
      .map((part) => part.trim())
      .filter(Boolean)
      .map(Number);
    if (dims.length === 0 || dims.some((dim) => !Number.isInteger(dim) || dim <= 0)) {
      setError('Shape must be positive integers, e.g. "4096, 32".');
      return;
    }

    let size: number;
    try {
      size = tensorByteSize(zeroModal.dtype, dims);
    } catch (err) {
      setError(`Cannot create tensor: ${err instanceof Error ? err.message : err}`);
      return;
    }

    recordHistory();
    setAddedTensors((current) => [
      ...current,
      { id: nextRowId(), name, shape: dims, dtype: zeroModal.dtype, size, source: { kind: "zeros" } },
    ]);
    setZeroModal(null);
    setError("");
    showToast(`Added zero tensor "${name}" (${formatBytes(size)})`);
  }

  function removeAddedTensor(addedIndex: number) {
    recordHistory();
    setAddedTensors((current) => current.filter((_, index) => index !== addedIndex));
  }

  function updateAddedTensorName(addedIndex: number, value: string) {
    recordHistory();
    setAddedTensors((current) =>
      current.map((added, index) => (index === addedIndex ? { ...added, name: value } : added))
    );
  }

  // ── save ──

  function computeFinalTensors(data: ParsedGGUF): {
    finals: FinalTensor[];
    plans: TensorWritePlan[];
    alignment: number;
  } {
    const alignment = resolveAlignment(data);
    const finals: FinalTensor[] = [];
    const plans: TensorWritePlan[] = [];
    let running = 0;

    const push = (
      name: string,
      shape: number[],
      dtype: number,
      size: number,
      plan: TensorWritePlan
    ) => {
      finals.push({ name, shape, dtype, offset: running });
      plans.push(plan);
      running += Math.ceil(size / alignment) * alignment;
    };

    const mergesById = new Map(merges.map((merge) => [merge.id, merge]));
    const addedById = new Map(addedTensors.map((added) => [added.id, added]));

    // Walk the unified row order so the saved file lays tensors out in the
    // sequence the user arranged in the UI.
    orderedKeys.forEach((key) => {
      const id = Number(key.slice(2));

      if (key.startsWith("t:")) {
        const tensor = data.tensorInfos[id];
        if (!tensor || deletedTensors.has(id) || mergedSourceIndices.has(id)) return;
        const size = tensorByteSize(tensor.dtype, tensor.shape);
        push(editedTensorNames[id] ?? tensor.name, tensor.shape, tensor.dtype, size, {
          segments: [
            { path: filePath, offset: data.tensorDataOffset + Number(tensor.offset), length: size },
          ],
          zeroFill: 0,
        });
        return;
      }

      if (key.startsWith("m:")) {
        const merge = mergesById.get(id);
        if (!merge) return;
        const segments = merge.parts.map((index) => {
          const tensor = data.tensorInfos[index];
          const size = tensorByteSize(tensor.dtype, tensor.shape);
          return {
            path: filePath,
            offset: data.tensorDataOffset + Number(tensor.offset),
            length: size,
          };
        });
        push(merge.name, merge.shape, merge.dtype, merge.size, { segments, zeroFill: 0 });
        return;
      }

      const added = addedById.get(id);
      if (!added) return;
      const plan: TensorWritePlan =
        added.source.kind === "external"
          ? {
              segments: [
                { path: added.source.path, offset: added.source.offset, length: added.size },
              ],
              zeroFill: 0,
            }
          : { segments: [], zeroFill: added.size };
      push(added.name, added.shape, added.dtype, added.size, plan);
    });

    const names = new Set<string>();
    for (const tensor of finals) {
      if (!tensor.name.trim()) throw new Error("Every tensor needs a non-empty name.");
      if (names.has(tensor.name))
        throw new Error(`Duplicate tensor name "${tensor.name}" — rename before saving.`);
      names.add(tensor.name);
    }

    return { finals, plans, alignment };
  }

  // Save button entry point. With pending precision changes this becomes a
  // two-step process: the plan modal confirms the reconciled tensor list
  // first, then handleSave writes the edited file (step 1) and hands it to
  // the quantizer (step 2).
  function handleSaveClick() {
    if (!parsedData || !filePath) return;
    if (quantPlan.entries.length > 0 || quantPlan.skipped.length > 0) {
      setQuantModalOpen(true);
      return;
    }
    handleSave();
  }

  async function handleSave() {
    if (!parsedData || !filePath) return;

    // Snapshot the reconciled quantize rules now — openGGUFPath resets all
    // pending-edit state after step 1.
    const quantRules = quantPlan.entries.map(
      (entry) => `^${escapeRegExp(entry.name)}$=${entry.to}`
    );

    const newMetaToAdd = newMetaRows.filter((row) => row.key.trim());

    try {
      const { save } = await import("@tauri-apps/plugin-dialog");
      const destination = await save({
        defaultPath: fileName.replace(/\.gguf$/i, "_edited.gguf"),
        filters: [{ name: "GGUF Model File", extensions: ["gguf"] }],
      });

      if (!destination) return;

      if (destination === filePath) {
        setError(
          "Saving over the file that is currently open is not supported — the save streams tensor data from it. Choose a different destination."
        );
        return;
      }

      setError("");
      setSaveOverlay({ active: true, progress: 0, label: "Starting…" });

      const { invoke } = await import("@tauri-apps/api/core");
      if (structuralChanges) {
        const { finals, plans, alignment } = computeFinalTensors(parsedData);
        const headerBytes = buildGGUFHeaderWithTensors(
          parsedData,
          editedMetadata,
          deletedMetaKeys,
          newMetaToAdd,
          finals
        );
        await invoke("rebuild_gguf_file", {
          destinationPath: destination,
          headerBytes: Array.from(headerBytes),
          alignment,
          tensors: plans,
        });
      } else {
        const editedTensorNameList = parsedData.tensorInfos.map(
          (tensor, index) => editedTensorNames[index] ?? tensor.name
        );
        const headerBytes = buildGGUFHeader(
          parsedData,
          editedMetadata,
          editedTensorNameList,
          deletedTensors,
          deletedMetaKeys,
          newMetaToAdd
        );
        await invoke("save_gguf_file", {
          sourcePath: filePath,
          destinationPath: destination,
          headerBytes: Array.from(headerBytes),
          tensorDataOffset: parsedData.tensorDataOffset,
        });
      }
      setSaveOverlay({ active: false, progress: 0, label: "Starting…" });
      // Reopen the edited file (this also resets all pending-edit state).
      await openGGUFPath(destination);

      // Step 2: quantize the freshly saved file into a second output.
      if (quantRules.length > 0) {
        const quantDestination = destination.replace(/\.gguf$/i, "") + "_quantized.gguf";
        try {
          setQuantLogs("");
          await invoke("start_quantize", {
            args: {
              input_path: destination,
              output_path: quantDestination,
              weight_type: null,
              tensor_rules: quantRules,
              threads: null,
            },
          });
          setQuantStatus("running");
          setLogsAvailable(true);
          setActiveView("logs");
          showToast(`Saved ${filename(destination)} — quantizing…`);
        } catch (err) {
          setQuantStatus("error");
          setError(
            `File saved, but quantization failed to start: ${err instanceof Error ? err.message : err}`
          );
        }
      } else {
        showToast("File saved successfully!");
      }
    } catch (err) {
      setSaveOverlay({ active: false, progress: 0, label: "Starting…" });
      setError(`Save failed: ${err instanceof Error ? err.message : err}`);
    }
  }

  function handleResetChanges() {
    if (!hasPendingEdits) return;
    resetEditState();
    setError("");
    showToast("Pending changes reset.");
  }

  // ── render ──

  const inputCls =
    "w-full px-2.5 py-1.5 rounded-lg bg-zinc-800 border border-zinc-700 text-zinc-200 text-xs font-mono focus:outline-none focus:border-emerald-500 placeholder:text-zinc-600 disabled:opacity-50 read-only:text-zinc-500";
  const smallBtnCls =
    "flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-zinc-800 hover:bg-zinc-700 text-zinc-300 text-xs font-medium transition-colors disabled:opacity-40 disabled:cursor-not-allowed";
  const dangerBtnCls =
    "px-2 py-1 rounded text-xs font-medium transition-colors bg-zinc-800 text-zinc-500 hover:bg-rose-100 hover:text-rose-600 dark:hover:bg-rose-900/40 dark:hover:text-rose-400";
  const restoreBtnCls =
    "px-2 py-1 rounded text-xs font-medium transition-colors bg-emerald-100 text-emerald-700 hover:bg-emerald-200 dark:bg-emerald-950/50 dark:text-emerald-400 dark:hover:bg-emerald-900/50";
  const theadCls = "text-left text-xs text-zinc-500 border-b border-zinc-800";
  // Buttons sitting on the green top bar: translucent white over the accent.
  const headerBtnCls =
    "flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-medium bg-white/15 hover:bg-white/25 border border-white/25 transition-colors whitespace-nowrap disabled:opacity-40 disabled:cursor-not-allowed flex-shrink-0";
  const headerIconBtnCls =
    "inline-flex h-7 w-7 items-center justify-center rounded-lg bg-white/10 hover:bg-white/20 transition-colors disabled:opacity-40 disabled:cursor-not-allowed flex-shrink-0";
  const viewTabCls =
    "flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors";
  const viewTabActiveCls =
    "bg-emerald-100 text-emerald-800 dark:bg-emerald-900/60 dark:text-emerald-200";
  const viewTabIdleCls = "text-zinc-500 hover:text-zinc-300 hover:bg-zinc-800/60";
  const sectionHeaderCls =
    "sticky top-0 z-10 flex items-center gap-2 px-4 py-2 bg-zinc-950 border-b border-zinc-800 flex-wrap";
  const sectionTitleCls = "text-xs font-semibold uppercase tracking-wider text-zinc-500";
  const statValueCls = "font-semibold text-zinc-200 ml-1";

  return (
    <div
      ref={rootRef}
      onDragEnter={(event) => {
        event.preventDefault();
        setDropActive(true);
      }}
      onDragOver={(event) => {
        event.preventDefault();
        setDropActive(true);
      }}
      onDragLeave={(event) => {
        if (!event.currentTarget.contains(event.relatedTarget as Node)) {
          setDropActive(false);
        }
      }}
      onDrop={handleBrowserDrop}
      className={cn(
        "flex-1 flex flex-col overflow-hidden relative",
        !visible && "hidden",
        isDark ? "dark bg-zinc-950 text-zinc-100" : "light bg-white text-zinc-900"
      )}
    >
      {/* ── Top bar (green, like the classic gguf-editor) ── */}
      <div
        className={cn(
          "flex items-center gap-2 px-3.5 py-2 flex-shrink-0",
          isDark ? "bg-emerald-900 text-emerald-100" : "bg-emerald-500 text-white"
        )}
      >
        <span className="flex items-center gap-2 whitespace-nowrap flex-shrink-0">
          <FilePen size={16} />
          <span className="font-bold text-[15px] tracking-wide">GGUF Editor</span>
          <span className="text-xs font-normal opacity-70">desktop</span>
        </span>
        <button onClick={handleOpenFile} className={headerBtnCls} title="Open a .gguf file">
          <FolderOpen size={12} />
          Open File
        </button>
        <span className="flex-1 min-w-0 text-xs font-mono opacity-80 truncate" title={filePath}>
          {fileName || "No file selected"}
        </span>
        {hasPendingEdits && (
          <span className="text-xs bg-white/15 border border-white/30 px-1.5 py-0.5 rounded whitespace-nowrap flex-shrink-0">
            unsaved edits
          </span>
        )}
        <button onClick={handleUndo} disabled={!canUndo} title="Undo" className={headerIconBtnCls}>
          <Undo2 size={13} />
        </button>
        <button onClick={handleRedo} disabled={!canRedo} title="Redo" className={headerIconBtnCls}>
          <Redo2 size={13} />
        </button>
        <button
          onClick={handleResetChanges}
          disabled={!hasPendingEdits}
          title="Reset pending changes"
          className={headerBtnCls}
        >
          <RotateCcw size={12} />
          Reset
        </button>
        <button
          onClick={handleSaveClick}
          disabled={!parsedData}
          title={
            quantPlan.entries.length > 0
              ? "Two steps: save the edited file, then quantize it to a second file"
              : undefined
          }
          className="flex items-center gap-1.5 px-3 py-1 rounded-lg bg-white hover:bg-emerald-50 text-emerald-700 text-xs font-semibold transition-colors shadow-sm disabled:opacity-50 disabled:cursor-not-allowed flex-shrink-0 whitespace-nowrap"
        >
          {quantPlan.entries.length > 0 ? <Zap size={12} /> : <Save size={12} />}
          {quantPlan.entries.length > 0 ? "Save Changes & Quantize" : "Save Changes"}
        </button>
        {onToggleTheme && (
          <button
            onClick={onToggleTheme}
            title={isDark ? "Switch to light mode" : "Switch to dark mode"}
            className={headerIconBtnCls}
          >
            {isDark ? <Sun size={14} /> : <Moon size={14} />}
          </button>
        )}
      </div>

      {/* ── Toolbar: view switch (appears once quantization ran) + unified search ── */}
      <div className="flex items-center gap-2 px-3 py-2 bg-zinc-900 border-b border-zinc-800 flex-shrink-0">
        {logsAvailable && (
          <div className="flex items-center gap-0.5 flex-shrink-0">
            <button
              onClick={() => setActiveView("editor")}
              className={cn(
                viewTabCls,
                activeView === "editor" ? viewTabActiveCls : viewTabIdleCls
              )}
            >
              <Database size={13} />
              Editor
            </button>
            <button
              onClick={() => setActiveView("logs")}
              className={cn(
                viewTabCls,
                "relative",
                activeView === "logs" ? viewTabActiveCls : viewTabIdleCls
              )}
            >
              <Terminal size={13} />
              Logs
              {quantStatus === "running" && (
                <span className="absolute top-0.5 right-0.5 w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
              )}
            </button>
          </div>
        )}
        {activeView === "editor" ? (
          <div className="relative flex-1">
            <Search
              size={14}
              className="absolute left-3 top-1/2 -translate-y-1/2 text-zinc-500 pointer-events-none"
            />
            <input
              type="text"
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              placeholder="Search metadata keys, values, tensor names…"
              className="w-full pl-9 pr-9 py-1.5 rounded-lg bg-zinc-800 border border-zinc-700 text-zinc-100 text-sm focus:outline-none focus:border-emerald-500 placeholder:text-zinc-600"
            />
            {search && (
              <button
                onClick={() => setSearch("")}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-zinc-500 hover:text-zinc-300"
              >
                <X size={13} />
              </button>
            )}
          </div>
        ) : (
          <div className="flex-1" />
        )}
      </div>

      {/* ── Stats strip ── */}
      {view === "data" && parsedData && (
        <div className="flex items-center gap-4 px-4 py-1.5 bg-zinc-800/40 border-b border-zinc-800 text-[11px] text-zinc-500 flex-shrink-0 flex-wrap">
          <span>
            Metadata
            <strong className={statValueCls}>
              {Object.keys(parsedData.metadata).length.toLocaleString()}
            </strong>
          </span>
          <span>
            Tensors
            <strong className={statValueCls}>
              {parsedData.tensorInfos.length.toLocaleString()}
            </strong>
          </span>
          <span>
            File size<strong className={statValueCls}>{formatBytes(fileSize)}</strong>
          </span>
          <span>
            Alignment<strong className={statValueCls}>{parsedData.alignment} B</strong>
          </span>
          {pendingBadges.length > 0 && (
            <span
              className="bg-amber-100 text-amber-800 border border-amber-300 dark:bg-amber-950/60 dark:text-amber-400 dark:border-amber-900 px-1.5 py-0.5 rounded"
              title="Structural changes are applied when you save — the file is rebuilt with tensor data rewritten"
            >
              {pendingBadges.join(" · ")}
            </span>
          )}
          <div className="flex-1" />
          <span className="text-[10px] font-bold bg-emerald-100 text-emerald-700 dark:bg-emerald-900/60 dark:text-emerald-300 px-2 py-0.5 rounded-full">
            GGUF v{parsedData.version}
          </span>
        </div>
      )}

      {/* ── Error banner ── */}
      {error && (
        <div className="flex items-start gap-2.5 px-4 py-2.5 bg-rose-500/10 border-b border-rose-500/30 flex-shrink-0">
          <AlertCircle size={15} className="text-rose-500 mt-0.5 flex-shrink-0" />
          <p className="text-sm text-rose-600 dark:text-rose-300 leading-relaxed flex-1">
            {error}
          </p>
          <button
            onClick={() => setError("")}
            className="text-rose-500/70 hover:text-rose-500 flex-shrink-0"
          >
            <X size={14} />
          </button>
        </div>
      )}

      {/* ── Content ── */}
      <main ref={mainRef} className="flex-1 overflow-auto">
        {activeView === "logs" && (
          <div className="px-4 py-4">
            <LogsPanel
              logs={quantLogs}
              quantStatus={quantStatus}
              onRefresh={fetchQuantLogs}
              onClear={() => setQuantLogs("")}
              onStop={handleStopQuantize}
            />
          </div>
        )}

        {activeView === "editor" && view === "landing" && (
          <div
            className={cn(
              "h-full flex flex-col items-center justify-center text-center gap-4 p-10 transition-colors",
              dropActive && "bg-emerald-50 dark:bg-emerald-950/30"
            )}
          >
            <div className="w-16 h-16 rounded-2xl bg-emerald-100 dark:bg-emerald-900/50 flex items-center justify-center">
              <FilePen size={30} className="text-emerald-600 dark:text-emerald-300" />
            </div>
            <h2 className="text-base font-semibold text-zinc-100">
              {dropActive ? "Drop the GGUF file to open it" : "Open a GGUF file to get started"}
            </h2>
            <p className="text-sm text-zinc-500 max-w-md leading-relaxed">
              Select or drop a <span className="font-mono text-zinc-400">.gguf</span> model
              file to preview and edit its metadata — and add, remove, merge, or reorder
              tensors.
            </p>
            <button
              onClick={handleOpenFile}
              className="mt-2 flex items-center gap-2 px-5 py-2.5 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white text-sm font-semibold transition-colors shadow-lg shadow-emerald-900/20"
            >
              <FolderOpen size={14} />
              Choose File…
            </button>
          </div>
        )}

        {activeView === "editor" && view === "loading" && (
          <div className="h-full flex flex-col items-center justify-center gap-3">
            <Loader2 size={28} className="text-emerald-500 animate-spin" />
            <span className="text-sm text-zinc-400">Parsing GGUF file…</span>
          </div>
        )}

        {activeView === "editor" && view === "data" && parsedData && (
          <>
            {/* ── Metadata section ── */}
            <div className={sectionHeaderCls}>
              <Braces size={13} className="text-emerald-600 dark:text-emerald-400" />
              <h2 className={sectionTitleCls}>Metadata</h2>
              <CountPill value={filteredMetadata.length + newMetaRows.length} />
              <div className="flex-1" />
              <button onClick={addMetaRow} className={smallBtnCls}>
                <Plus size={12} />
                Add Row
              </button>
            </div>
            <div className="px-4 pb-5">
              <table className="w-full table-fixed">
                    <thead>
                      <tr className={theadCls}>
                        <th className="py-2 pr-3 font-medium w-[240px]">Key</th>
                        <th className="py-2 pr-3 font-medium">Value</th>
                        <th className="py-2 pr-3 font-medium w-[90px] text-center">Type</th>
                        <th className="py-2 font-medium w-[80px] text-right">Actions</th>
                      </tr>
                    </thead>
                    <tbody>
                      {filteredMetadata.map(({ key, type, displayValue, typeLabel }) => {
                        const isDeleted = deletedMetaKeys.has(key);
                        const value = editedMetadata[key] ?? displayValue;
                        const multiline = shouldUseTextarea(key, type, value);
                        return (
                          <tr
                            key={key}
                            className={cn(
                              "border-b border-zinc-800/60 align-top hover:bg-emerald-50 dark:hover:bg-emerald-950/20",
                              isDeleted && "opacity-40"
                            )}
                          >
                            <td
                              className={cn(
                                "py-2 pr-3 text-xs font-mono text-zinc-300 break-all",
                                isDeleted && "line-through"
                              )}
                              title={key}
                            >
                              {key}
                            </td>
                            <td className="py-2 pr-3">
                              {multiline ? (
                                <textarea
                                  className={cn(inputCls, "leading-relaxed resize-y")}
                                  rows={4}
                                  value={value}
                                  readOnly={isDeleted}
                                  onChange={(event) => updateMetaValue(key, event.target.value)}
                                />
                              ) : (
                                <input
                                  className={inputCls}
                                  type="text"
                                  value={value}
                                  readOnly={isDeleted}
                                  title={
                                    type === GGUFValueType.ARRAY
                                      ? "Comma-separated values, e.g.: 1.0, 2.0, 3.0"
                                      : ""
                                  }
                                  onChange={(event) => updateMetaValue(key, event.target.value)}
                                />
                              )}
                            </td>
                            <td className="py-2 pr-3 text-center">
                              <TypeBadge label={typeLabel} />
                            </td>
                            <td className="py-2 text-right">
                              <button
                                onClick={() => toggleDeleteMeta(key)}
                                className={isDeleted ? restoreBtnCls : dangerBtnCls}
                              >
                                {isDeleted ? "Restore" : "Delete"}
                              </button>
                            </td>
                          </tr>
                        );
                      })}

                      {newMetaRows.map((row, index) => {
                        const isArrayType = row.type >= 100;
                        return (
                          <tr
                            key={`new-${index}`}
                            className="border-b border-zinc-800/60 align-top bg-emerald-50 dark:bg-emerald-950/20"
                          >
                            <td className="py-2 pr-3">
                              <input
                                type="text"
                                className={inputCls}
                                placeholder="new.key"
                                value={row.key}
                                onChange={(event) =>
                                  updateNewMetaRow(index, { key: event.target.value })
                                }
                              />
                            </td>
                            <td className="py-2 pr-3">
                              <input
                                type="text"
                                className={inputCls}
                                placeholder={isArrayType ? "1.0, 2.0, 3.0" : "value"}
                                title={
                                  isArrayType
                                    ? "Comma-separated values, e.g.: 1.0, 2.0, 3.0"
                                    : ""
                                }
                                value={row.value}
                                onChange={(event) =>
                                  updateNewMetaRow(index, { value: event.target.value })
                                }
                              />
                            </td>
                            <td className="py-2 pr-3 text-center">
                              <select
                                className="px-1.5 py-1.5 rounded-lg bg-zinc-800 border border-zinc-700 text-zinc-200 text-xs font-mono focus:outline-none focus:border-emerald-500 cursor-pointer w-full"
                                value={row.type}
                                onChange={(event) =>
                                  updateNewMetaRow(index, { type: Number(event.target.value) })
                                }
                              >
                                {EDITABLE_TYPES.map(([typeValue, name]) => (
                                  <option key={typeValue} value={typeValue}>
                                    {name}
                                  </option>
                                ))}
                              </select>
                            </td>
                            <td className="py-2 text-right">
                              <button
                                onClick={() => removeNewMetaRow(index)}
                                className={dangerBtnCls}
                              >
                                Remove
                              </button>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>

              {filteredMetadata.length + newMetaRows.length === 0 && (
                <p className="text-sm text-zinc-600 italic text-center py-6">
                  No metadata matches your search.
                </p>
              )}
            </div>

            {/* ── Tensors section ── */}
            <div className={sectionHeaderCls}>
              <Database size={13} className="text-emerald-600 dark:text-emerald-400" />
              <h2 className={sectionTitleCls}>Tensors</h2>
              <CountPill value={displayRows.length} />
              <div className="flex-1" />
              <button
                onClick={handleImportFromGGUF}
                className={smallBtnCls}
                title="Copy tensors from another GGUF file into this one"
              >
                <Import size={12} />
                Import from GGUF…
              </button>
              <button
                onClick={() => setZeroModal({ name: "", dtype: 0, shape: "" })}
                className={smallBtnCls}
                title="Add a new zero-filled tensor"
              >
                <FilePlus size={12} />
                Zero Tensor…
              </button>
              <button
                onClick={handleMergeSelected}
                disabled={selectedTensors.size < 2}
                className={smallBtnCls}
                title="Concatenate the selected tensors along their last axis into one tensor"
              >
                <Combine size={12} />
                Merge Selected{selectedTensors.size > 0 ? ` (${selectedTensors.size})` : ""}
              </button>
            </div>
            <div className="px-4 pb-6">
              {/* ── Find & Replace drawer ── */}
              <div className="mt-3 mb-3">
                    <button
                      onClick={() => setFrOpen((current) => !current)}
                      className={cn(
                        "flex items-center gap-1.5 w-full px-2.5 py-1.5 rounded-lg text-xs font-medium transition-colors",
                        frOpen
                          ? "bg-zinc-800 text-zinc-200"
                          : "bg-zinc-800/60 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200"
                      )}
                    >
                      <Replace size={12} />
                      Find &amp; Replace in Names
                      {!frOpen && frFind && (
                        <span className="text-xs bg-zinc-700/60 text-zinc-400 px-1.5 py-0.5 rounded">
                          {frPreview.targets.length}
                        </span>
                      )}
                      <div className="flex-1" />
                      <ChevronDown
                        size={13}
                        className={cn("transition-transform", frOpen && "rotate-180")}
                      />
                    </button>

                    {frOpen && (
                      <div className="mt-2 rounded-lg border border-zinc-800 bg-zinc-950/40 p-3 space-y-2.5">
                        <p className="text-xs text-zinc-600 italic">
                          Renames every matching tensor across the file — independent of the
                          search box above.
                        </p>
                        <div className="grid grid-cols-2 gap-2">
                          <input
                            type="text"
                            value={frFind}
                            onChange={(event) => setFrFind(event.target.value)}
                            placeholder="Find in tensor names…"
                            className={inputCls}
                            autoFocus
                          />
                          <input
                            type="text"
                            value={frReplace}
                            onChange={(event) => setFrReplace(event.target.value)}
                            placeholder="Replace with…"
                            className={inputCls}
                          />
                        </div>
                        <div className="flex items-center justify-between gap-3 flex-wrap">
                          <div className="flex items-center gap-3 text-xs text-zinc-400">
                            <label className="flex items-center gap-1.5 cursor-pointer select-none">
                              <input
                                type="checkbox"
                                checked={frMatchCase}
                                onChange={(event) => setFrMatchCase(event.target.checked)}
                                className="accent-emerald-600 cursor-pointer"
                              />
                              Match case
                            </label>
                            <label className="flex items-center gap-1.5 cursor-pointer select-none">
                              <input
                                type="checkbox"
                                checked={frRegex}
                                onChange={(event) => setFrRegex(event.target.checked)}
                                className="accent-emerald-600 cursor-pointer"
                              />
                              Regex
                            </label>
                          </div>
                          <div className="flex items-center gap-2">
                            {frFind && (
                              <button
                                onClick={() => {
                                  setFrFind("");
                                  setFrReplace("");
                                }}
                                className="text-xs text-zinc-500 hover:text-zinc-300"
                              >
                                Clear
                              </button>
                            )}
                            <button
                              onClick={applyFindReplace}
                              disabled={frDisabled}
                              className={smallBtnCls}
                              title="Rename every matching tensor"
                            >
                              <ReplaceAll size={12} />
                              Replace All
                              {frPreview.targets.length > 0 ? ` (${frPreview.targets.length})` : ""}
                            </button>
                          </div>
                        </div>
                        {frFind && (
                          <p
                            className={cn(
                              "text-xs",
                              frPreview.regexError ||
                                frPreview.duplicates.length > 0 ||
                                frPreview.hasEmptyResult
                                ? "text-amber-400"
                                : "text-zinc-500"
                            )}
                          >
                            {frPreview.regexError
                              ? `Invalid pattern: ${frPreview.regexError}`
                              : frPreview.duplicates.length > 0
                                ? `Would create duplicate name${
                                    frPreview.duplicates.length > 1 ? "s" : ""
                                  }: "${frPreview.duplicates.slice(0, 3).join('", "')}"${
                                    frPreview.duplicates.length > 3 ? "…" : ""
                                  }`
                                : frPreview.hasEmptyResult
                                  ? "Replacement would leave a tensor with an empty name."
                                  : frPreview.targets.length > 0
                                    ? `${frPreview.targets.length} tensor name${
                                        frPreview.targets.length === 1 ? "" : "s"
                                      } will change`
                                    : "No tensor names match."}
                          </p>
                        )}
                      </div>
                    )}
                  </div>

                  <table className="w-full table-fixed">
                    <thead>
                      <tr className={theadCls}>
                        <th className="py-2 pr-1 w-[22px]"></th>
                        <th className="py-2 pr-2 w-[28px]"></th>
                        <th className="py-2 pr-3 font-medium">Name</th>
                        <th className="py-2 pr-3 font-medium w-[160px]">Shape</th>
                        <th className="py-2 pr-3 font-medium w-[110px] text-center">Precision</th>
                        <th className="py-2 font-medium w-[80px] text-right">Actions</th>
                      </tr>
                    </thead>
                    <tbody>
                      {displayRows.map((row) => {
                        if (row.kind === "orig") {
                          const { index, name, shape } = row;
                          const isDeleted = deletedTensors.has(index);
                          const inMerge = mergedSourceIndices.has(index);
                          const info = parsedData.tensorInfos[index];
                          return (
                            <tr
                              key={row.key}
                              data-row-key={row.key}
                              className={cn(
                                "border-b border-zinc-800/60 hover:bg-emerald-50 dark:hover:bg-emerald-950/20",
                                isDeleted && "opacity-40",
                                inMerge && "opacity-60 bg-sky-50/60 dark:bg-sky-950/10",
                                rowDragCls(row.key)
                              )}
                            >
                              {gripCell(row.key)}
                              <td className="py-2 pr-2">
                                <input
                                  type="checkbox"
                                  checked={selectedTensors.has(index)}
                                  disabled={isDeleted || inMerge}
                                  title="Select for merge"
                                  onChange={() => toggleSelectTensor(index)}
                                  className="accent-emerald-600 cursor-pointer disabled:cursor-not-allowed"
                                />
                              </td>
                              <td className="py-2 pr-3">
                                <div className="flex items-center gap-2">
                                  <input
                                    type="text"
                                    className={cn(inputCls, isDeleted && "line-through")}
                                    value={name}
                                    disabled={isDeleted || inMerge}
                                    onChange={(event) =>
                                      updateTensorName(index, event.target.value)
                                    }
                                  />
                                  {inMerge && <RowBadge label="in merge" tone="merge" />}
                                </div>
                              </td>
                              <td className="py-2 pr-3">
                                <TypeBadge label={shape} />
                              </td>
                              <td className="py-2 pr-3 text-center">
                                <PrecisionSelect
                                  currentDtype={info.dtype}
                                  ne0={info.shape[0] ?? 0}
                                  value={precisionOverrides[row.key] ?? ""}
                                  disabled={isDeleted || inMerge}
                                  onChange={(value) => updatePrecision(row.key, value)}
                                />
                              </td>
                              <td className="py-2 text-right">
                                {!inMerge && (
                                  <button
                                    onClick={() => toggleDeleteTensor(index)}
                                    className={isDeleted ? restoreBtnCls : dangerBtnCls}
                                  >
                                    {isDeleted ? "Restore" : "Delete"}
                                  </button>
                                )}
                              </td>
                            </tr>
                          );
                        }

                        if (row.kind === "merge") {
                          const { merge, mergeIndex } = row;
                          return (
                            <tr
                              key={row.key}
                              data-row-key={row.key}
                              className={cn(
                                "border-b border-zinc-800/60 bg-sky-50 dark:bg-sky-950/20",
                                rowDragCls(row.key)
                              )}
                            >
                              {gripCell(row.key)}
                              <td className="py-2 pr-2"></td>
                              <td className="py-2 pr-3">
                                <div className="flex items-center gap-2">
                                  <input
                                    type="text"
                                    className={inputCls}
                                    value={merge.name}
                                    onChange={(event) =>
                                      updateMergeName(mergeIndex, event.target.value)
                                    }
                                  />
                                  <RowBadge
                                    label={`merged ×${merge.parts.length}`}
                                    tone="merge"
                                    title={`Concatenation of ${merge.parts.length} tensors along the last axis`}
                                  />
                                </div>
                              </td>
                              <td className="py-2 pr-3">
                                <TypeBadge label={merge.shape.join(" × ")} />
                              </td>
                              <td className="py-2 pr-3 text-center">
                                <PrecisionSelect
                                  currentDtype={merge.dtype}
                                  ne0={merge.shape[0] ?? 0}
                                  value={precisionOverrides[row.key] ?? ""}
                                  onChange={(value) => updatePrecision(row.key, value)}
                                />
                              </td>
                              <td className="py-2 text-right">
                                <button
                                  onClick={() => removeMerge(mergeIndex)}
                                  className={dangerBtnCls}
                                >
                                  Unmerge
                                </button>
                              </td>
                            </tr>
                          );
                        }

                        const { added, addedIndex } = row;
                        return (
                          <tr
                            key={row.key}
                            data-row-key={row.key}
                            className={cn(
                              "border-b border-zinc-800/60 bg-emerald-50 dark:bg-emerald-950/20",
                              rowDragCls(row.key)
                            )}
                          >
                            {gripCell(row.key)}
                            <td className="py-2 pr-2"></td>
                            <td className="py-2 pr-3">
                              <div className="flex items-center gap-2">
                                <input
                                  type="text"
                                  className={inputCls}
                                  value={added.name}
                                  onChange={(event) =>
                                    updateAddedTensorName(addedIndex, event.target.value)
                                  }
                                />
                                <RowBadge
                                  label={added.source.kind === "external" ? "imported" : "zeros"}
                                  tone="add"
                                  title={
                                    added.source.kind === "external"
                                      ? `Imported from ${added.source.path}`
                                      : "Zero-filled tensor"
                                  }
                                />
                              </div>
                            </td>
                            <td className="py-2 pr-3">
                              <TypeBadge label={added.shape.join(" × ")} />
                            </td>
                            <td className="py-2 pr-3 text-center">
                              <PrecisionSelect
                                currentDtype={added.dtype}
                                ne0={added.shape[0] ?? 0}
                                value={precisionOverrides[row.key] ?? ""}
                                onChange={(value) => updatePrecision(row.key, value)}
                              />
                            </td>
                            <td className="py-2 text-right">
                              <button
                                onClick={() => removeAddedTensor(addedIndex)}
                                className={dangerBtnCls}
                              >
                                Remove
                              </button>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>

              {displayRows.length === 0 && (
                <p className="text-sm text-zinc-600 italic text-center py-6">
                  No tensors match your search.
                </p>
              )}
            </div>
          </>
        )}
      </main>

      {/* ── Status bar ── */}
      <footer
        className={cn(
          "flex items-center gap-4 px-5 py-1.5 border-t text-xs flex-shrink-0",
          isDark
            ? "bg-zinc-900 border-zinc-800 text-zinc-600"
            : "bg-white border-gray-200 text-gray-400"
        )}
      >
        <span className="font-mono truncate">{fileName || "(no file)"}</span>
        {parsedData && (
          <>
            <span className={isDark ? "text-zinc-800" : "text-gray-200"}>|</span>
            <span className="font-mono">
              {Object.keys(parsedData.metadata).length} keys · {parsedData.tensorInfos.length}{" "}
              tensors
            </span>
          </>
        )}
        <div className="flex-1" />
        {quantStatus !== "idle" && (
          <span
            className={cn(
              "font-medium",
              quantStatus === "running"
                ? "text-zinc-300"
                : quantStatus === "completed"
                  ? "text-emerald-400"
                  : "text-rose-400"
            )}
          >
            {quantStatus === "running"
              ? "Quantizing…"
              : quantStatus === "completed"
                ? "Quantization done"
                : "Quantization error"}
          </span>
        )}
        <span className={cn("font-medium", hasPendingEdits ? "text-amber-400" : undefined)}>
          {hasPendingEdits ? "Unsaved edits" : "GGUF Editor"}
        </span>
      </footer>

      {/* ── Full-window drop hint (file already open) ── */}
      {dropActive && view === "data" && (
        <div className="absolute inset-0 z-40 bg-zinc-950/80 flex items-center justify-center pointer-events-none">
          <div className="rounded-xl border-2 border-dashed border-emerald-500 bg-zinc-900 px-10 py-8 flex flex-col items-center gap-3">
            <FilePen size={30} className="text-emerald-400" />
            <span className="text-sm font-semibold text-zinc-100">Drop GGUF file to open</span>
          </div>
        </div>
      )}

      {/* ── Import modal ── */}
      {importModal && (
        <ModalShell title={`Import tensors from ${importModal.fileName}`} wide>
          <div className="px-5 py-3 flex items-center gap-2 border-b border-zinc-800">
            <input
              type="text"
              className={inputCls}
              placeholder="Filter tensors…"
              value={importModal.filter}
              onChange={(event) =>
                setImportModal({ ...importModal, filter: event.target.value })
              }
            />
            <button
              className={cn(smallBtnCls, "flex-shrink-0")}
              onClick={() => {
                const lcFilter = importModal.filter.trim().toLowerCase();
                const isVisible = (tensor: ImportModalTensor) =>
                  !lcFilter || tensor.name.toLowerCase().includes(lcFilter);
                const allChecked = importModal.tensors
                  .filter(isVisible)
                  .every((tensor) => tensor.checked);
                setImportModal({
                  ...importModal,
                  tensors: importModal.tensors.map((tensor) =>
                    isVisible(tensor) ? { ...tensor, checked: !allChecked } : tensor
                  ),
                });
              }}
            >
              Toggle All
            </button>
          </div>
          <div className="flex-1 overflow-auto px-5">
            <table className="w-full">
              <tbody>
                {importModal.tensors.map((tensor, index) => {
                  const lcFilter = importModal.filter.trim().toLowerCase();
                  if (lcFilter && !tensor.name.toLowerCase().includes(lcFilter)) return null;
                  const unsupported = tensor.size == null;
                  return (
                    <tr
                      key={index}
                      className={cn(
                        "border-b border-zinc-800/60",
                        unsupported && "opacity-40"
                      )}
                    >
                      <td className="py-1.5 pr-2 w-[28px]">
                        <input
                          type="checkbox"
                          checked={tensor.checked}
                          disabled={unsupported}
                          className="accent-emerald-600 cursor-pointer disabled:cursor-not-allowed"
                          onChange={() => {
                            const tensors = [...importModal.tensors];
                            tensors[index] = { ...tensor, checked: !tensor.checked };
                            setImportModal({ ...importModal, tensors });
                          }}
                        />
                      </td>
                      <td className="py-1.5 pr-3 text-xs font-mono text-zinc-300 break-all">
                        {tensor.name}
                      </td>
                      <td className="py-1.5 pr-3 text-xs font-mono text-zinc-500 whitespace-nowrap">
                        {tensor.shape.join(" × ") || "(scalar)"}
                      </td>
                      <td className="py-1.5 pr-3">
                        <TypeBadge label={quantizationName(tensor.dtype)} />
                      </td>
                      <td className="py-1.5 text-xs text-zinc-500 whitespace-nowrap text-right">
                        {unsupported ? "unsupported" : formatBytes(tensor.size as number)}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <div className="px-5 py-3.5 border-t border-zinc-800 flex items-center gap-2">
            <span className="text-xs text-zinc-500 flex-1">
              {importModal.tensors.filter((tensor) => tensor.checked).length} selected — data is
              copied when you save
            </span>
            <button className={smallBtnCls} onClick={() => setImportModal(null)}>
              Cancel
            </button>
            <button
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
              disabled={!importModal.tensors.some((tensor) => tensor.checked)}
              onClick={confirmImport}
            >
              <Check size={12} />
              Add Selected
            </button>
          </div>
        </ModalShell>
      )}

      {/* ── Zero tensor modal ── */}
      {zeroModal && (
        <ModalShell title="New zero-filled tensor">
          <div className="px-5 py-4 space-y-4">
            <div className="space-y-1.5">
              <label className="text-xs text-zinc-500">Name</label>
              <input
                type="text"
                className={inputCls}
                placeholder="my.new.tensor"
                value={zeroModal.name}
                onChange={(event) => setZeroModal({ ...zeroModal, name: event.target.value })}
              />
            </div>
            <div className="space-y-1.5">
              <label className="text-xs text-zinc-500">Precision</label>
              <select
                className="w-full px-2.5 py-1.5 rounded-lg bg-zinc-800 border border-zinc-700 text-zinc-200 text-xs font-mono focus:outline-none focus:border-emerald-500 cursor-pointer"
                value={zeroModal.dtype}
                onChange={(event) =>
                  setZeroModal({ ...zeroModal, dtype: Number(event.target.value) })
                }
              >
                {ZERO_TENSOR_DTYPES.map((dtype) => (
                  <option key={dtype} value={dtype}>
                    {quantizationName(dtype)}
                  </option>
                ))}
              </select>
            </div>
            <div className="space-y-1.5">
              <label className="text-xs text-zinc-500">Shape</label>
              <input
                type="text"
                className={inputCls}
                placeholder="4096, 32"
                title={
                  (typeTraits(zeroModal.dtype)?.[0] ?? 1) > 1
                    ? `First dimension must be a multiple of ${typeTraits(zeroModal.dtype)?.[0]} for ${quantizationName(zeroModal.dtype)}`
                    : "Comma-separated dimensions"
                }
                value={zeroModal.shape}
                onChange={(event) => setZeroModal({ ...zeroModal, shape: event.target.value })}
              />
              {(typeTraits(zeroModal.dtype)?.[0] ?? 1) > 1 && (
                <p className="text-xs text-zinc-600">
                  First dimension must be a multiple of {typeTraits(zeroModal.dtype)?.[0]} for{" "}
                  {quantizationName(zeroModal.dtype)}.
                </p>
              )}
            </div>
          </div>
          <div className="px-5 py-3.5 border-t border-zinc-800 flex items-center justify-end gap-2">
            <button className={smallBtnCls} onClick={() => setZeroModal(null)}>
              Cancel
            </button>
            <button
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold transition-colors"
              onClick={confirmZeroTensor}
            >
              <Plus size={12} />
              Add Tensor
            </button>
          </div>
        </ModalShell>
      )}

      {/* ── Quantize plan modal ── */}
      {quantModalOpen && (
        <ModalShell title="Save Changes & Quantize" wide>
          <div className="px-5 py-4 space-y-3 overflow-auto">
            <p className="text-xs text-zinc-400 leading-relaxed">
              This is a two-step process producing two files: first your edits are saved to an{" "}
              <span className="font-mono text-zinc-300">edited</span> GGUF, then the quantizer
              converts the tensors below into a second{" "}
              <span className="font-mono text-zinc-300">*_quantized.gguf</span> file using{" "}
              <span className="font-mono text-zinc-300">--tensor-type-rules</span>. Progress is
              shown in the Logs tab.
            </p>

            {quantPlan.entries.length > 0 ? (
              <div className="rounded-lg border border-zinc-800 overflow-hidden">
                <table className="w-full">
                  <thead>
                    <tr className="text-left text-xs text-zinc-500 border-b border-zinc-800 bg-zinc-950/40">
                      <th className="py-1.5 px-3 font-medium">Tensor</th>
                      <th className="py-1.5 px-3 font-medium w-[180px] text-right">Conversion</th>
                    </tr>
                  </thead>
                  <tbody>
                    {quantPlan.entries.map((entry) => (
                      <tr key={entry.key} className="border-b border-zinc-800/60 last:border-b-0">
                        <td className="py-1.5 px-3 text-xs font-mono text-zinc-300 break-all">
                          {entry.name}
                        </td>
                        <td className="py-1.5 px-3 text-xs font-mono text-right whitespace-nowrap">
                          <span className="text-zinc-500">{entry.from}</span>
                          <span className="text-zinc-600"> → </span>
                          <span className="text-amber-300">{entry.to.toUpperCase()}</span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className="text-xs text-amber-400">
                No quantizable tensors remain — every precision change points at a tensor that was
                deleted, merged away, or removed. Continuing performs a normal save only.
              </p>
            )}

            {quantPlan.skipped.length > 0 && (
              <p className="text-xs text-amber-400 leading-relaxed">
                Skipped (name contains &quot;,&quot; or &quot;=&quot;, which the rule syntax cannot
                express): {quantPlan.skipped.join(", ")}
              </p>
            )}

            <p className="text-xs text-zinc-600">
              Precision changes on deleted tensors are dropped automatically — those tensors will
              not exist in the edited file.
            </p>
          </div>
          <div className="px-5 py-3.5 border-t border-zinc-800 flex items-center justify-end gap-2">
            <button className={smallBtnCls} onClick={() => setQuantModalOpen(false)}>
              Cancel
            </button>
            <button
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold transition-colors"
              onClick={() => {
                setQuantModalOpen(false);
                handleSave();
              }}
            >
              <Zap size={12} />
              {quantPlan.entries.length > 0
                ? `Save & Quantize ${quantPlan.entries.length} Tensor${quantPlan.entries.length > 1 ? "s" : ""}`
                : "Save Only"}
            </button>
          </div>
        </ModalShell>
      )}

      {/* ── Save progress overlay ── */}
      {saveOverlay.active && (
        <div className="fixed inset-0 z-50 bg-black/80 flex items-center justify-center p-6">
          <div className="w-full max-w-md rounded-xl border border-zinc-800 bg-zinc-900 p-6 space-y-4 text-center shadow-2xl shadow-black/50">
            <div className="flex items-center justify-center gap-2.5">
              <Loader2 size={18} className="text-zinc-300 animate-spin" />
              <span className="text-sm font-semibold text-zinc-100">Saving file…</span>
            </div>
            <div className="h-2 rounded-full bg-zinc-800 overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-emerald-500 to-emerald-300 transition-all"
                style={{ width: `${saveOverlay.progress}%` }}
              />
            </div>
            <p className="text-xs font-mono text-zinc-400">{saveOverlay.label}</p>
            <div className="flex items-start gap-2 text-left bg-amber-950/40 border border-amber-900/60 rounded-lg px-3 py-2">
              <AlertCircle size={13} className="text-amber-400 mt-0.5 flex-shrink-0" />
              <p className="text-xs text-amber-300 leading-relaxed">
                Do not close this window — closing will cancel the save and your file will be
                incomplete.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* ── Toast ── */}
      {toast && (
        <div className="fixed bottom-10 right-6 z-50 flex items-center gap-2 rounded-lg border border-zinc-700 bg-zinc-800 px-4 py-2.5 text-sm text-zinc-100 shadow-xl shadow-black/40">
          <Check size={14} className="text-emerald-400" />
          {toast}
        </div>
      )}

    </div>
  );
}
