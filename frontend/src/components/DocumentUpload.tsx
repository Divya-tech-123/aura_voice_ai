import React, { useState, useRef, useEffect } from 'react';
import { Upload, FileText, Check, AlertCircle, Loader2, X, RefreshCw } from 'lucide-react';
import { uploadDocument, fetchDocuments, DocumentItem, DocumentUploadResponse } from '../services/api';

interface DocumentUploadProps {
  onUploadSuccess?: (response: DocumentUploadResponse) => void;
  className?: string;
  triggerRef?: React.MutableRefObject<(() => void) | null>;
}

/**
 * Strips any directory components from filename to prevent exposing server filesystem paths.
 */
function getSafeBasename(filepath: string): string {
  if (!filepath) return 'document';
  return filepath.replace(/^.*[\\/]/, '');
}

export const DocumentUpload: React.FC<DocumentUploadProps> = ({
  onUploadSuccess,
  className = '',
  triggerRef,
}) => {
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [isLoadingDocs, setIsLoadingDocs] = useState<boolean>(false);
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [uploadSuccess, setUploadSuccess] = useState<{ filename: string } | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // Load documents from backend
  const loadDocuments = async () => {
    setIsLoadingDocs(true);
    try {
      const docs = await fetchDocuments();
      setDocuments(docs);
    } catch (err) {
      console.warn('Failed to load documents:', err);
    } finally {
      setIsLoadingDocs(false);
    }
  };

  useEffect(() => {
    loadDocuments();
  }, []);

  // Expose trigger to parent if ref provided (e.g. from composer attachment button)
  useEffect(() => {
    if (triggerRef) {
      triggerRef.current = () => {
        fileInputRef.current?.click();
      };
    }
  }, [triggerRef]);

  const handleFilePickerClick = () => {
    setUploadError(null);
    fileInputRef.current?.click();
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // Reset input so the same file can be re-selected if desired
    e.target.value = '';

    await handleUpload(file);
  };

  const handleUpload = async (file: File) => {
    setIsUploading(true);
    setUploadError(null);
    setUploadSuccess(null);

    try {
      const result = await uploadDocument(file);
      const safeName = getSafeBasename(result.filename || file.name);
      setUploadSuccess({ filename: safeName });
      if (onUploadSuccess) {
        onUploadSuccess(result);
      }
      await loadDocuments();
    } catch (err: any) {
      const errorMsg = err?.message || 'Failed to upload document. Please check the file and try again.';
      setUploadError(errorMsg);
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className={`flex flex-col gap-3 ${className}`}>
      {/* Hidden file input supporting PDF, DOCX, TXT */}
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileChange}
        accept=".pdf,.docx,.txt"
        className="hidden"
        aria-label="Upload document file"
      />

      {/* Upload Action Button */}
      <button
        type="button"
        onClick={handleFilePickerClick}
        disabled={isUploading}
        aria-label="Upload document to knowledge base"
        className="w-full flex items-center justify-center gap-2 py-2 px-3 rounded-xl bg-slate-800 hover:bg-slate-700/90 border border-slate-700/80 hover:border-cyan-500/50 text-cyan-300 text-xs sm:text-sm font-medium transition-all shadow-sm focus:outline-none focus:ring-2 focus:ring-cyan-500/40 disabled:opacity-60 disabled:cursor-not-allowed cursor-pointer"
      >
        {isUploading ? (
          <>
            <Loader2 className="w-4 h-4 animate-spin text-cyan-400" />
            <span>Indexing document...</span>
          </>
        ) : (
          <>
            <Upload className="w-4 h-4 text-cyan-400" />
            <span>Upload Document</span>
          </>
        )}
      </button>

      {/* Supported formats hint */}
      <div className="flex items-center justify-between text-[11px] text-slate-400 px-0.5">
        <span>PDF, DOCX, TXT</span>
        <span>Max 10 MB</span>
      </div>

      {/* Uploading Progress Indicator */}
      {isUploading && (
        <div className="flex items-center gap-2.5 p-2.5 rounded-xl bg-cyan-950/40 border border-cyan-500/40 text-xs text-cyan-200 animate-pulse">
          <Loader2 className="w-4 h-4 animate-spin text-cyan-400 shrink-0" />
          <div className="flex-1 min-w-0">
            <p className="font-medium truncate">Indexing document...</p>
            <p className="text-[10px] text-cyan-300/80">Parsing text &amp; generating vector embeddings</p>
          </div>
        </div>
      )}

      {/* Success Notification Box */}
      {uploadSuccess && (
        <div className="relative flex items-start gap-2.5 p-2.5 rounded-xl bg-emerald-950/50 border border-emerald-500/40 text-emerald-200 text-xs shadow-sm animate-fade-in">
          <div className="w-4 h-4 rounded-full bg-emerald-500/20 text-emerald-400 flex items-center justify-center shrink-0 mt-0.5">
            <Check className="w-3 h-3 stroke-[2.5]" />
          </div>
          <div className="flex-1 min-w-0 pr-4">
            <p className="font-semibold text-emerald-300 truncate">
              {uploadSuccess.filename}
            </p>
            <p className="text-[11px] text-emerald-400 mt-0.5 leading-snug">
              Indexed and ready for retrieval
            </p>
          </div>
          <button
            type="button"
            onClick={() => setUploadSuccess(null)}
            className="absolute top-2 right-2 text-emerald-400 hover:text-emerald-200 p-0.5 rounded transition-colors"
            title="Dismiss"
            aria-label="Dismiss success message"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      )}

      {/* Error Notification Box */}
      {uploadError && (
        <div className="relative flex items-start gap-2.5 p-2.5 rounded-xl bg-rose-950/50 border border-rose-500/40 text-rose-200 text-xs shadow-sm animate-fade-in">
          <AlertCircle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
          <div className="flex-1 min-w-0 pr-4">
            <p className="font-semibold text-rose-300">Upload failed</p>
            <p className="text-[11px] text-rose-400 mt-0.5 leading-snug break-words">
              {uploadError}
            </p>
          </div>
          <button
            type="button"
            onClick={() => setUploadError(null)}
            className="absolute top-2 right-2 text-rose-400 hover:text-rose-200 p-0.5 rounded transition-colors"
            title="Dismiss"
            aria-label="Dismiss error message"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      )}

      {/* Document List Section */}
      <div className="mt-1">
        <div className="flex items-center justify-between px-0.5 pb-1.5 text-[11px] font-semibold uppercase tracking-wider text-slate-400">
          <span>Documents ({documents.length})</span>
          <button
            type="button"
            onClick={loadDocuments}
            disabled={isLoadingDocs}
            className="hover:text-cyan-400 p-0.5 rounded transition-colors text-slate-400 focus:outline-none focus:ring-1 focus:ring-cyan-500/40"
            title="Refresh document list"
            aria-label="Refresh document list"
          >
            <RefreshCw className={`w-3 h-3 ${isLoadingDocs ? 'animate-spin text-cyan-400' : ''}`} />
          </button>
        </div>

        {documents.length === 0 ? (
          <div className="px-3 py-4 rounded-xl border border-dashed border-slate-800 text-center bg-slate-900/40">
            <FileText className="w-5 h-5 mx-auto text-slate-600 mb-1.5 stroke-[1.5]" />
            <p className="text-xs text-slate-300 font-medium">No documents yet</p>
            <p className="text-[11px] text-slate-400 mt-0.5">
              Upload files to enable RAG answers
            </p>
          </div>
        ) : (
          <div className="flex flex-col gap-1.5 max-h-48 overflow-y-auto pr-0.5">
            {documents.map((doc) => {
              const safeName = getSafeBasename(doc.filename);
              return (
                <div
                  key={doc.filename}
                  className="flex items-center justify-between gap-2 p-2 rounded-xl bg-slate-800/60 border border-slate-700/50 hover:border-slate-600/70 transition-colors text-xs"
                >
                  <div className="flex items-center gap-2 min-w-0">
                    <FileText className="w-3.5 h-3.5 text-cyan-400 shrink-0" />
                    <span className="truncate text-slate-200 font-medium" title={safeName}>
                      {safeName}
                    </span>
                  </div>
                  <span className="shrink-0 inline-flex items-center gap-1 text-[10px] font-medium px-1.5 py-0.5 rounded-md bg-emerald-950/80 text-emerald-400 border border-emerald-800/40">
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-400"></span>
                    <span>{doc.status || 'ready'}</span>
                  </span>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
};

export default DocumentUpload;
