export type DocCategory = "frequent" | "infrequent";

export type DocItem = {
  source: string;
  source_path: string;
  chunk_count: number;
};

export function inferDocCategory(sourcePath: string): DocCategory {
  const normalized = sourcePath.replace(/\\/g, "/").toLowerCase();
  if (normalized.includes("/infrequent/")) {
    return "infrequent";
  }
  return "frequent";
}

export type UploadResult = {
  uploaded_files: string[];
  ingested_chunks: number;
  skipped_files: string[];
  errors: string[];
};

export type DeleteResult = {
  deleted_chunks: number;
  deleted_files: string[];
  not_found_files: string[];
};

export type IngestProgressEvent = {
  stage: string;
  message?: string;
  file?: string;
  file_index?: number;
  file_total?: number;
  done?: number;
  total?: number;
  chunks?: number;
  result?: UploadResult;
};

declare global {
  interface ImportMetaEnv {
    readonly VITE_API_BASE?: string;
  }
  interface ImportMeta {
    readonly env: ImportMetaEnv;
  }
}

const API_BASE =
  (import.meta.env.VITE_API_BASE as string | undefined)?.trim() || "";

export const API_DISPLAY = API_BASE || "同源 /admin → 6006";

function buildApiUrl(path: string): string {
  return `${API_BASE}${path}`;
}

function parseJsonText<T>(text: string, status: number): T {
  const trimmed = text.trim();
  if (
    trimmed.toLowerCase().startsWith("<!doctype") ||
    trimmed.startsWith("<")
  ) {
    throw new Error(
      "API 返回了网页而不是 JSON。请确认后端 gateway 已启动（6006），并重启前端 dev server 以启用 /admin 代理。"
    );
  }
  try {
    return JSON.parse(text) as T;
  } catch {
    throw new Error(trimmed || `请求失败: ${status}`);
  }
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(buildApiUrl(path), init);
  const text = await res.text();
  if (!res.ok) {
    throw new Error(text || `请求失败: ${res.status}`);
  }
  return parseJsonText<T>(text, res.status);
}

export async function fetchDocuments(): Promise<DocItem[]> {
  const data = await requestJson<{ items: DocItem[] }>("/admin/documents");
  return data.items;
}

function parseStreamLines(
  chunk: string,
  onProgress: (event: IngestProgressEvent) => void
): { result: UploadResult | null; remainder: string } {
  let result: UploadResult | null = null;
  const lines = chunk.split("\n");
  const remainder = chunk.endsWith("\n") ? "" : lines.pop() || "";

  for (const line of lines) {
    if (!line.trim()) continue;
    try {
      const event = JSON.parse(line) as IngestProgressEvent;
      onProgress(event);
      if (event.stage === "complete" && event.result) {
        result = event.result;
      }
    } catch {
      // 忽略未完整的一行
    }
  }
  return { result, remainder };
}

export function uploadDocumentsWithProgress(
  files: File[],
  category: DocCategory,
  onProgress: (event: IngestProgressEvent) => void,
  onUploadPercent?: (percent: number) => void
): Promise<UploadResult> {
  return new Promise((resolve, reject) => {
    const form = new FormData();
    form.append("category", category);
    for (const file of files) {
      form.append("files", file);
    }

    const xhr = new XMLHttpRequest();
    xhr.open("POST", buildApiUrl("/admin/upload-stream"));

    let lastLen = 0;
    let finalResult: UploadResult | null = null;
    let bufferedText = "";
    const consumeNewResponse = () => {
      const text = bufferedText + xhr.responseText.substring(lastLen);
      lastLen = xhr.responseText.length;
      const parsed = parseStreamLines(text, onProgress);
      bufferedText = parsed.remainder;
      const result = parsed.result;
      if (result) {
        finalResult = result;
      }
      return result;
    };

    xhr.upload.onprogress = (e) => {
      if (!onUploadPercent || !e.lengthComputable) return;
      onUploadPercent(Math.round((e.loaded / e.total) * 100));
    };

    xhr.onprogress = () => {
      consumeNewResponse();
    };

    xhr.onload = () => {
      if (xhr.status < 200 || xhr.status >= 300) {
        try {
          reject(new Error(parseJsonText<{ detail?: string }>(xhr.responseText, xhr.status).detail || xhr.responseText));
        } catch (err) {
          reject(err instanceof Error ? err : new Error(xhr.responseText || `请求失败: ${xhr.status}`));
        }
        return;
      }

      consumeNewResponse();
      if (bufferedText.trim()) {
        const parsed = parseStreamLines(bufferedText + "\n", onProgress);
        bufferedText = parsed.remainder;
        if (parsed.result) {
          finalResult = parsed.result;
        }
      }
      if (finalResult) {
        resolve(finalResult);
      } else {
        reject(new Error("入库进度流结束，但未收到 complete 事件"));
      }
    };

    xhr.onerror = () => {
      reject(new Error("网络错误：无法连接后端 API，请确认 gateway 是否在运行"));
    };

    xhr.send(form);
  });
}

export async function deleteDocuments(
  sourcePaths: string[],
  removeFiles: boolean
): Promise<DeleteResult> {
  return requestJson<DeleteResult>("/admin/delete", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ source_paths: sourcePaths, remove_files: removeFiles })
  });
}

export { API_BASE };
