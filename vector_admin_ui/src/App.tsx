import { ChangeEvent, DragEvent, useEffect, useMemo, useRef, useState } from "react";
import {
  API_DISPLAY,
  DocCategory,
  DocItem,
  IngestProgressEvent,
  deleteDocuments,
  fetchDocuments,
  inferDocCategory,
  uploadDocumentsWithProgress
} from "./api";

type Notice = {
  type: "success" | "error" | "info";
  text: string;
};

const CATEGORIES: DocCategory[] = ["frequent", "infrequent"];

const CATEGORY_META: Record<
  DocCategory,
  { title: string; description: string }
> = {
  frequent: {
    title: "经常更新",
    description: "通知公告、课表安排、办事流程等会定期替换的文档"
  },
  infrequent: {
    title: "不经常更新",
    description: "规章制度、培养方案、办事手册等长期有效的参考文档"
  }
};

function formatChunk(count: number): string {
  return `${count.toLocaleString()} chunks`;
}

type ProgressState = {
  active: boolean;
  message: string;
  percent: number;
  file?: string;
  fileIndex?: number;
  fileTotal?: number;
  stage?: string;
};

const EMPTY_PROGRESS: ProgressState = {
  active: false,
  message: "",
  percent: 0
};

function calcProgressPercent(
  event: IngestProgressEvent,
  uploadPercent: number
): number {
  if (uploadPercent < 100) {
    return Math.max(1, Math.round(uploadPercent * 0.08));
  }

  if (event.stage === "complete") {
    return 100;
  }

  const fileTotal = event.file_total || 1;
  const fileIndex = Math.max(0, (event.file_index || 1) - 1);
  const fileSpan = 92 / fileTotal;
  const fileBase = 8 + fileIndex * fileSpan;

  if (event.stage === "batch_start" || event.stage === "saving") {
    return Math.round(fileBase + fileSpan * 0.05);
  }
  if (event.stage === "loading" || event.stage === "chunking") {
    return Math.round(fileBase + fileSpan * 0.2);
  }
  if (
    (event.stage === "embedding" || event.stage === "writing") &&
    event.total
  ) {
    const inner = Math.min(1, (event.done || 0) / event.total);
    return Math.round(fileBase + fileSpan * (0.25 + inner * 0.7));
  }
  if (event.stage === "file_done") {
    return Math.round(fileBase + fileSpan * 0.95);
  }
  return Math.round(fileBase);
}

function toProgressState(
  event: IngestProgressEvent,
  uploadPercent: number
): ProgressState {
  return {
    active: event.stage !== "complete",
    message: event.message || "处理中...",
    percent: calcProgressPercent(event, uploadPercent),
    file: event.file,
    fileIndex: event.file_index,
    fileTotal: event.file_total,
    stage: event.stage
  };
}

function mergePendingFiles(old: File[], incoming: File[]): File[] {
  const map = new Map<string, File>();
  for (const f of old) {
    map.set(f.name.toLowerCase(), f);
  }
  for (const f of incoming) {
    map.set(f.name.toLowerCase(), f);
  }
  return Array.from(map.values());
}

function existingNamesInCategory(
  docs: DocItem[],
  category: DocCategory
): Set<string> {
  const names = new Set<string>();
  for (const doc of docs) {
    if (inferDocCategory(doc.source_path) === category && doc.source) {
      names.add(doc.source.toLowerCase());
    }
  }
  return names;
}

type UploadPanelProps = {
  category: DocCategory;
  pendingFiles: File[];
  dragOver: boolean;
  uploading: boolean;
  deleting: boolean;
  progress: ProgressState;
  onFilesChosen: (category: DocCategory, files: File[]) => void;
  onUpload: (category: DocCategory) => void;
  onClearPending: (category: DocCategory) => void;
  onRemovePending: (category: DocCategory, key: string) => void;
  onDragOver: (category: DocCategory) => void;
  onDragLeave: () => void;
};

function UploadPanel({
  category,
  pendingFiles,
  dragOver,
  uploading,
  deleting,
  progress,
  onFilesChosen,
  onUpload,
  onClearPending,
  onRemovePending,
  onDragOver,
  onDragLeave
}: UploadPanelProps) {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const meta = CATEGORY_META[category];

  const onDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    onDragLeave();
    const files: File[] = e.dataTransfer.files ? Array.from(e.dataTransfer.files) : [];
    onFilesChosen(category, files);
  };

  const onFileInputChange = (e: ChangeEvent<HTMLInputElement>) => {
    const files: File[] = e.target.files ? Array.from(e.target.files) : [];
    e.target.value = "";
    onFilesChosen(category, files);
  };

  return (
    <section className={`card uploader uploader-${category}`}>
      <div className="section-head">
        <h2>{meta.title}</h2>
        <p className="section-desc">{meta.description}</p>
      </div>
      <div
        className={`dropzone ${dragOver ? "drag-over" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          onDragOver(category);
        }}
        onDragLeave={onDragLeave}
        onDrop={onDrop}
      >
        <p>拖拽文件到这里，或点击按钮选择多个文件</p>
        <p className="hint">支持: pdf/docx/txt/md/xlsx/xls/png/jpg/jpeg/webp</p>
        <button
          type="button"
          disabled={uploading || deleting}
          onClick={() => fileInputRef.current?.click()}
        >
          选择文件
        </button>
        <input
          ref={fileInputRef}
          type="file"
          multiple
          hidden
          onChange={onFileInputChange}
        />
      </div>

      <div className="pending-panel">
        <div className="pending-head">
          <strong>待上传文件</strong>
          <span>{pendingFiles.length} 个</span>
        </div>
        {!pendingFiles.length ? (
          <div className="pending-empty">尚未添加文件</div>
        ) : (
          <ul className="pending-list">
            {pendingFiles.map((f) => {
              const key = f.name.toLowerCase();
              return (
                <li key={key}>
                  <span title={f.name}>{f.name}</span>
                  <button
                    type="button"
                    className="link-btn"
                    onClick={() => onRemovePending(category, key)}
                  >
                    移除
                  </button>
                </li>
              );
            })}
          </ul>
        )}
        <div className="pending-actions">
            <button
              type="button"
              onClick={() => onUpload(category)}
              disabled={uploading || !pendingFiles.length}
            >
              {uploading ? "入库中..." : "开始上传"}
            </button>
            <button
              type="button"
              className="ghost"
              onClick={() => onClearPending(category)}
              disabled={!pendingFiles.length || uploading}
            >
              清空列表
            </button>
          </div>
        </div>

        {progress.active && (
          <div className="progress-panel">
            <div className="progress-head">
              <strong>入库进度</strong>
              <span>{progress.percent}%</span>
            </div>
            <div className="progress-track">
              <div
                className="progress-fill"
                style={{ width: `${progress.percent}%` }}
              />
            </div>
            <p className="progress-message">{progress.message}</p>
            {progress.file && (
              <p className="progress-detail">
                当前文件: {progress.file}
                {progress.fileIndex && progress.fileTotal
                  ? ` (${progress.fileIndex}/${progress.fileTotal})`
                  : ""}
              </p>
            )}
          </div>
        )}
      </section>
  );
}

type DocTableProps = {
  category: DocCategory;
  docs: DocItem[];
  selected: Record<string, boolean>;
  loadingDocs: boolean;
  onToggleSelect: (sourcePath: string, checked: boolean) => void;
};

function DocTable({
  category,
  docs,
  selected,
  loadingDocs,
  onToggleSelect
}: DocTableProps) {
  const meta = CATEGORY_META[category];

  return (
    <section className={`card table-wrap table-${category}`}>
      <div className="section-head compact">
        <h2>
          {meta.title}
          <span className="section-count">{docs.length} 个文档</span>
        </h2>
      </div>
      {!docs.length && !loadingDocs ? (
        <div className="empty">当前版块暂无文档，请在上方对应区域上传。</div>
      ) : (
        <table>
          <thead>
            <tr>
              <th>选择</th>
              <th>文件名</th>
              <th>路径</th>
              <th>Chunk 数</th>
            </tr>
          </thead>
          <tbody>
            {docs.map((doc) => (
              <tr key={doc.source_path}>
                <td>
                  <input
                    type="checkbox"
                    checked={Boolean(selected[doc.source_path])}
                    onChange={(e) => onToggleSelect(doc.source_path, e.target.checked)}
                  />
                </td>
                <td title={doc.source}>{doc.source || "-"}</td>
                <td title={doc.source_path} className="path">
                  {doc.source_path}
                </td>
                <td>{formatChunk(doc.chunk_count)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  );
}

export default function App() {
  const [docs, setDocs] = useState<DocItem[]>([]);
  const [pendingByCategory, setPendingByCategory] = useState<
    Record<DocCategory, File[]>
  >({
    frequent: [],
    infrequent: []
  });
  const [uploadingCategory, setUploadingCategory] = useState<DocCategory | null>(
    null
  );
  const [progressByCategory, setProgressByCategory] = useState<
    Record<DocCategory, ProgressState>
  >({
    frequent: { ...EMPTY_PROGRESS },
    infrequent: { ...EMPTY_PROGRESS }
  });
  const [selected, setSelected] = useState<Record<string, boolean>>({});
  const [dragOverCategory, setDragOverCategory] = useState<DocCategory | null>(
    null
  );
  const [query, setQuery] = useState("");
  const [removeFiles, setRemoveFiles] = useState(false);
  const [loadingDocs, setLoadingDocs] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [notice, setNotice] = useState<Notice | null>(null);

  const selectedPaths = useMemo(
    () => Object.keys(selected).filter((p) => selected[p]),
    [selected]
  );

  const filteredDocs = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return docs;
    return docs.filter(
      (d) =>
        d.source.toLowerCase().includes(q) ||
        d.source_path.toLowerCase().includes(q)
    );
  }, [docs, query]);

  const docsByCategory = useMemo(() => {
    const grouped: Record<DocCategory, DocItem[]> = {
      frequent: [],
      infrequent: []
    };
    for (const doc of filteredDocs) {
      grouped[inferDocCategory(doc.source_path)].push(doc);
    }
    return grouped;
  }, [filteredDocs]);

  const categoryCounts = useMemo(() => {
    const counts: Record<DocCategory, number> = {
      frequent: 0,
      infrequent: 0
    };
    for (const doc of docs) {
      counts[inferDocCategory(doc.source_path)] += 1;
    }
    return counts;
  }, [docs]);

  const totalChunks = useMemo(
    () => docs.reduce((acc, d) => acc + d.chunk_count, 0),
    [docs]
  );

  const clearNoticeLater = () => {
    window.setTimeout(() => setNotice(null), 3500);
  };

  useEffect(() => {
    void refreshDocs();
  }, []);

  const refreshDocs = async () => {
    setLoadingDocs(true);
    try {
      const items = await fetchDocuments();
      setDocs(items);
      setSelected((old) => {
        const next: Record<string, boolean> = {};
        for (const item of items) {
          if (old[item.source_path]) next[item.source_path] = true;
        }
        return next;
      });
    } catch (err) {
      setNotice({ type: "error", text: `加载文档失败: ${String(err)}` });
      clearNoticeLater();
    } finally {
      setLoadingDocs(false);
    }
  };

  const onFilesChosen = (category: DocCategory, files: File[]) => {
    if (!files.length) {
      return;
    }

    const existing = existingNamesInCategory(docs, category);
    const accepted: File[] = [];
    const alreadyPending: string[] = [];
    const alreadyInLibrary: string[] = [];

    const currentPendingNames = new Set(
      pendingByCategory[category].map((f) => f.name.toLowerCase())
    );

    for (const file of files) {
      const lowerName = file.name.toLowerCase();
      if (currentPendingNames.has(lowerName)) {
        alreadyPending.push(file.name);
        continue;
      }
      if (existing.has(lowerName)) {
        alreadyInLibrary.push(file.name);
        continue;
      }
      accepted.push(file);
    }

    if (!accepted.length) {
      const parts = [
        alreadyPending.length
          ? `待上传列表中已有：${alreadyPending.join("、")}`
          : "",
        alreadyInLibrary.length
          ? `库中已存在：${alreadyInLibrary.join("、")}`
          : ""
      ].filter(Boolean);
      setNotice({
        type: "info",
        text: parts.join("；") || "没有可添加的新文件。"
      });
      clearNoticeLater();
      return;
    }

    setPendingByCategory((old) => ({
      ...old,
      [category]: mergePendingFiles(old[category], accepted)
    }));

    const parts = [
      `已添加 ${accepted.length} 个文件到待上传列表`,
      alreadyPending.length
        ? `跳过待上传重复：${alreadyPending.join("、")}`
        : "",
      alreadyInLibrary.length
        ? `跳过库中已有：${alreadyInLibrary.join("、")}`
        : ""
    ].filter(Boolean);

    setNotice({
      type: "info",
      text: parts.join("；") + "，请点击“开始上传”提交。"
    });
    clearNoticeLater();
  };

  const onUploadPending = async (category: DocCategory) => {
    let pendingFiles = pendingByCategory[category];
    if (!pendingFiles.length) {
      setNotice({
        type: "info",
        text: `请先在「${CATEGORY_META[category].title}」选择文件，再点击上传。`
      });
      clearNoticeLater();
      return;
    }

    const existing = existingNamesInCategory(docs, category);
    const duplicateInLibrary = pendingFiles
      .filter((f) => existing.has(f.name.toLowerCase()))
      .map((f) => f.name);
    if (duplicateInLibrary.length) {
      pendingFiles = pendingFiles.filter(
        (f) => !existing.has(f.name.toLowerCase())
      );
      setPendingByCategory((old) => ({ ...old, [category]: pendingFiles }));
      if (!pendingFiles.length) {
        setNotice({
          type: "info",
          text: `以下文件已在库中，无需重复上传：${duplicateInLibrary.join("、")}`
        });
        clearNoticeLater();
        return;
      }
      setNotice({
        type: "info",
        text: `已跳过库中已有文件：${duplicateInLibrary.join("、")}`
      });
    }

    setUploadingCategory(category);
    let uploadPercent = 0;
    setProgressByCategory((old) => ({
      ...old,
      [category]: {
        active: true,
        message: "正在上传文件...",
        percent: 1,
        stage: "uploading"
      }
    }));
    try {
      const result = await uploadDocumentsWithProgress(
        pendingFiles,
        category,
        (event) => {
          setProgressByCategory((old) => ({
            ...old,
            [category]: toProgressState(event, uploadPercent)
          }));
          if (event.stage === "complete" || event.stage === "file_done") {
            void refreshDocs();
          }
        },
        (percent) => {
          uploadPercent = percent;
          setProgressByCategory((old) => ({
            ...old,
            [category]: {
              active: true,
              message: `正在上传文件... ${percent}%`,
              percent: calcProgressPercent({ stage: "uploading" }, percent),
              stage: "uploading"
            }
          }));
        }
      );
      const msg = [
        `「${CATEGORY_META[category].title}」上传成功 ${result.uploaded_files.length} 个文件`,
        `新增 ${result.ingested_chunks} 个 chunk`,
        result.skipped_files.length ? `跳过 ${result.skipped_files.length} 个` : "",
        result.errors.length ? `失败 ${result.errors.length} 个` : ""
      ]
        .filter(Boolean)
        .join("，");
      setNotice({
        type: result.errors.length ? "error" : "success",
        text: msg
      });
      setPendingByCategory((old) => ({ ...old, [category]: [] }));
      setProgressByCategory((old) => ({
        ...old,
        [category]: {
          active: true,
          message: "入库完成",
          percent: 100,
          stage: "complete"
        }
      }));
      await refreshDocs();
      window.setTimeout(() => {
        setProgressByCategory((old) => ({
          ...old,
          [category]: { ...EMPTY_PROGRESS }
        }));
      }, 2000);
    } catch (err) {
      setNotice({ type: "error", text: `上传失败: ${String(err)}` });
      setProgressByCategory((old) => ({
        ...old,
        [category]: { ...EMPTY_PROGRESS }
      }));
    } finally {
      setUploadingCategory(null);
      clearNoticeLater();
    }
  };

  const onClearPending = (category: DocCategory) => {
    setPendingByCategory((old) => ({ ...old, [category]: [] }));
  };

  const onRemovePending = (category: DocCategory, key: string) => {
    setPendingByCategory((old) => ({
      ...old,
      [category]: old[category].filter((f) => f.name.toLowerCase() !== key)
    }));
  };

  const toggleSelectAllFiltered = () => {
    const allSelected = filteredDocs.every((d) => selected[d.source_path]);
    const next = { ...selected };
    for (const d of filteredDocs) {
      next[d.source_path] = !allSelected;
    }
    setSelected(next);
  };

  const onDeleteSelected = async () => {
    if (!selectedPaths.length) return;
    const ok = window.confirm(
      `确认删除 ${selectedPaths.length} 个文档对应向量？此操作不可撤销。`
    );
    if (!ok) return;

    setDeleting(true);
    try {
      const result = await deleteDocuments(selectedPaths, removeFiles);
      setNotice({
        type: "success",
        text: `已删除 ${result.deleted_chunks} 条向量记录`
      });
      setSelected({});
      await refreshDocs();
    } catch (err) {
      setNotice({ type: "error", text: `删除失败: ${String(err)}` });
    } finally {
      setDeleting(false);
      clearNoticeLater();
    }
  };

  return (
    <div className="page">
      <header className="hero">
        <div>
          <h1>向量库文档管理台</h1>
          <p>
            按更新频率分区上传与管理文档，支持按文档多选删除并同步清理向量。
          </p>
        </div>
        <div className="hero-meta">
          <span>API: {API_DISPLAY}</span>
        </div>
      </header>

      {notice && <div className={`notice ${notice.type}`}>{notice.text}</div>}

      <div className="dual-grid upload-grid">
        {CATEGORIES.map((category) => (
          <UploadPanel
            key={category}
            category={category}
            pendingFiles={pendingByCategory[category]}
            dragOver={dragOverCategory === category}
            uploading={uploadingCategory === category}
            deleting={deleting}
            progress={progressByCategory[category]}
            onFilesChosen={onFilesChosen}
            onUpload={onUploadPending}
            onClearPending={onClearPending}
            onRemovePending={onRemovePending}
            onDragOver={setDragOverCategory}
            onDragLeave={() => setDragOverCategory(null)}
          />
        ))}
      </div>

      <section className="card toolbar">
        <div className="stats">
          <span>文档总数: {docs.length}</span>
          <span>经常更新: {categoryCounts.frequent}</span>
          <span>不经常更新: {categoryCounts.infrequent}</span>
          <span>向量块总数: {totalChunks.toLocaleString()}</span>
          <span>已选中: {selectedPaths.length}</span>
        </div>
        <div className="actions">
          <input
            type="text"
            placeholder="按文件名或路径搜索..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <button type="button" onClick={refreshDocs} disabled={loadingDocs}>
            {loadingDocs ? "刷新中..." : "刷新列表"}
          </button>
          <button type="button" onClick={toggleSelectAllFiltered}>
            全选/反选当前筛选
          </button>
          <label className="switch">
            <input
              type="checkbox"
              checked={removeFiles}
              onChange={(e) => setRemoveFiles(e.target.checked)}
            />
            同步删除磁盘文件
          </label>
          <button
            type="button"
            className="danger"
            disabled={!selectedPaths.length || deleting}
            onClick={onDeleteSelected}
          >
            {deleting ? "删除中..." : "删除选中文档"}
          </button>
        </div>
      </section>

      <div className="dual-grid docs-grid">
        {CATEGORIES.map((category) => (
          <DocTable
            key={category}
            category={category}
            docs={docsByCategory[category]}
            selected={selected}
            loadingDocs={loadingDocs}
            onToggleSelect={(sourcePath, checked) =>
              setSelected((old) => ({
                ...old,
                [sourcePath]: checked
              }))
            }
          />
        ))}
      </div>
    </div>
  );
}
