import { useEffect, useRef, useState } from "react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";

export default function App() {
  const [activeTab, setActiveTab] = useState("Explorer");
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [files, setFiles] = useState([]);
  const [uploadStatus, setUploadStatus] = useState("");
  const [uploading, setUploading] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [workspace, setWorkspace] = useState("Research Workspace");
  const fileInput = useRef(null);
  const [documents, setDocuments] = useState([]);
  const [selectedDocument, setSelectedDocument] = useState(null);
  const [documentSearch, setDocumentSearch] = useState("");
  const [documentsLoading, setDocumentsLoading] = useState(false);
  const [documentsError, setDocumentsError] = useState("");
  const [chatMessages, setChatMessages] = useState([]);
  const [chatQuestion, setChatQuestion] = useState("");
  const [chatLoading, setChatLoading] = useState(false);
  const [chatError, setChatError] = useState("");

  async function loadDocuments() {
    setDocumentsLoading(true);
    setDocumentsError("");

    try {
      const response = await fetch(`${API_URL}/documents`);

      if (!response.ok) {
        throw new Error("Unable to load documents.");
      }

      const data = await response.json();
      setDocuments(data.documents ?? []);
    } catch (err) {
      setDocumentsError(err.message || "Could not connect to the backend.");
    } finally {
      setDocumentsLoading(false);
    }
  }

  useEffect(() => {
    if (activeTab === "Collection") {
      loadDocuments();
    }
  }, [activeTab]);
  const filteredDocuments = documents.filter((doc) =>
    doc.name.toLowerCase().includes(documentSearch.trim().toLowerCase()),
  );
  const totalStorage = documents.reduce(
    (total, doc) => total + (doc.size_bytes || 0),
    0,
  );

  const totalStorageMB = (totalStorage / (1024 * 1024)).toFixed(2);

  const fileTypes = new Set(
    documents.map((doc) => doc.type?.toUpperCase() || "UNKNOWN"),
  ).size;

  async function askQuestion(event) {
    event.preventDefault();

    if (!question.trim()) return;

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch(`${API_URL}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: question.trim() }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "The request failed.");
      }

      setResult(data);
    } catch (err) {
      setError(
        err.message.includes("fetch")
          ? "Cannot connect to the backend. Check that FastAPI is running."
          : err.message,
      );
    } finally {
      setLoading(false);
    }
  }
  async function sendChatMessage(event) {
    event.preventDefault();

    const question = chatQuestion.trim();
    if (!question || chatLoading) return;

    setChatMessages((previous) => [
      ...previous,
      { role: "user", content: question },
    ]);
    setChatQuestion("");
    setChatLoading(true);
    setChatError("");

    try {
      const response = await fetch(`${API_URL}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },

        body: JSON.stringify({
          question,
          history: chatMessages.map((message) => ({
            role: message.role,
            content: message.content,
          })),
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Could not answer the question.");
      }

      setChatMessages((previous) => [
        ...previous,
        {
          role: "assistant",
          content: data.answer,
          citations: data.citations || [],
          evidence: data.evidence || [],
          traceId: data.trace_id,
        },
      ]);
    } catch (error) {
      setChatError(error.message || "Could not connect to the backend.");
    } finally {
      setChatLoading(false);
    }
  }

  async function handleFiles(eventOrFiles) {
    const selected = Array.from(
      eventOrFiles?.target?.files ?? eventOrFiles ?? [],
    );

    if (eventOrFiles?.target) {
      eventOrFiles.target.value = "";
    }

    if (selected.length === 0 || uploading) return;

    const validFiles = [];
    const rejected = [];

    for (const file of selected) {
      const extension = file.name.split(".").pop()?.toLowerCase();
      const isSupported = extension === "pdf" || extension === "txt";

      if (!isSupported) {
        rejected.push(`${file.name}: only PDF and TXT files are supported.`);
        continue;
      }

      if (file.size === 0) {
        rejected.push(`${file.name}: the file is empty.`);
        continue;
      }

      const duplicate =
        files.some(
          (item) =>
            item.name.toLowerCase() === file.name.toLowerCase() &&
            item.size === file.size &&
            item.status !== "Failed",
        ) ||
        validFiles.some(
          (item) =>
            item.name.toLowerCase() === file.name.toLowerCase() &&
            item.size === file.size,
        );

      if (duplicate) {
        rejected.push(`${file.name}: already added.`);
        continue;
      }

      validFiles.push(file);
    }

    if (validFiles.length === 0) {
      setUploadStatus(rejected.join(" "));
      return;
    }

    const pendingFiles = validFiles.map((file) => ({
      id: crypto.randomUUID(),
      name: file.name,
      size: file.size,
      status: "Uploading",
      documentId: null,
    }));

    setFiles((previous) => [...previous, ...pendingFiles]);
    setUploading(true);
    setUploadStatus(rejected.join(" "));
    setError("");

    const failures = [];

    try {
      for (let i = 0; i < validFiles.length; i++) {
        const file = validFiles[i];
        const pending = pendingFiles[i];

        try {
          const formData = new FormData();
          formData.append("file", file);

          const response = await fetch(`${API_URL}/upload`, {
            method: "POST",
            body: formData,
          });

          const data = await response.json();

          if (!response.ok) {
            throw new Error(
              typeof data.detail === "string" ? data.detail : "Upload failed.",
            );
          }

          setFiles((previous) =>
            previous.map((item) =>
              item.id === pending.id
                ? {
                    ...item,
                    status: "Indexed",
                    documentId: data.document_id,
                  }
                : item,
            ),
          );
        } catch (err) {
          failures.push(`${file.name}: ${err.message}`);

          setFiles((previous) =>
            previous.map((item) =>
              item.id === pending.id ? { ...item, status: "Failed" } : item,
            ),
          );
        }
      }

      const successful = validFiles.length - failures.length;
      const messages = [...rejected];

      if (successful > 0) {
        messages.push(`Successfully indexed ${successful} file(s).`);
      }

      if (failures.length > 0) {
        messages.push(`Upload failed: ${failures.join(" | ")}`);
      }

      setUploadStatus(messages.join(" "));
    } finally {
      setUploading(false);
    }
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">R</div>
          <div>
            <h1>ResearchPilot</h1>
            <p>Agentic research workspace</p>
          </div>
        </div>

        <button
          className="workspace-button"
          onClick={() => {
            const name = window.prompt("Enter a workspace name:");
            if (name?.trim()) setWorkspace(name.trim());
          }}
        >
          <span className="workspace-icon">▦</span>
          <span>
            <small>WORKSPACE</small>
            <strong>{workspace}</strong>
          </span>
          <span className="muted">⌄</span>
        </button>

        <p className="nav-label">RESEARCH</p>

        {["Upload", "Collection", "Explorer", "Chat"].map((tab) => (
          <button
            key={tab}
            className={`nav-item ${activeTab === tab ? "active" : ""}`}
            onClick={() => setActiveTab(tab)}
          >
            <span className="nav-icon">
              {tab === "Upload"
                ? "↑"
                : tab === "Collection"
                  ? "▤"
                  : tab === "Explorer"
                    ? "⌕"
                    : "◌"}
            </span>
            {tab}
            {tab === "Upload" && files.length > 0 && (
              <span className="nav-count">{files.length}</span>
            )}
          </button>
        ))}

        <div className="sidebar-bottom">
          <div className="status-dot" />
          <span>Research API</span>
          <span className="status-label">Local</span>
        </div>
      </aside>

      <main className="main-area">
        <header className="topbar">
          <div>
            <p className="eyebrow">RESEARCH WORKSPACE</p>
            <h2>{activeTab}</h2>
          </div>
          <div className="topbar-badge">
            <span className="status-dot" />
            Agentic RAG
          </div>
        </header>
        {activeTab === "Upload" && (
          <section className="page-content">
            <div className="section-heading">
              <div>
                <p className="eyebrow">KNOWLEDGE BASE</p>
                <h2>Add research documents</h2>
                <p>Select PDF or text files to prepare your collection.</p>
              </div>
            </div>
            <div
              className={`upload-zone ${isDragging ? "dragging" : ""}`}
              role="button"
              tabIndex={uploading ? -1 : 0}
              aria-label="Choose research files or drag and drop them"
              onClick={() => !uploading && fileInput.current?.click()}
              onKeyDown={(event) => {
                if (
                  !uploading &&
                  (event.key === "Enter" || event.key === " ")
                ) {
                  event.preventDefault();
                  fileInput.current?.click();
                }
              }}
              onDragEnter={(event) => {
                event.preventDefault();
                if (!uploading) setIsDragging(true);
              }}
              onDragOver={(event) => event.preventDefault()}
              onDragLeave={(event) => {
                if (!event.currentTarget.contains(event.relatedTarget)) {
                  setIsDragging(false);
                }
              }}
              onDrop={(event) => {
                event.preventDefault();
                setIsDragging(false);
                if (!uploading) handleFiles(event.dataTransfer.files);
              }}
            >
              <span className="upload-icon">↑</span>
              <strong>
                {isDragging ? "Drop files to upload" : "Choose research files"}
              </strong>
              <span>Drag and drop PDF or TXT files here</span>
              <span className="primary-button">Browse files</span>
            </div>

            <input
              ref={fileInput}
              type="file"
              accept=".pdf,.txt,application/pdf,text/plain"
              multiple
              disabled={uploading}
              hidden
              onChange={handleFiles}
            />

            <div className="collection-stats">
              <div className="stat-card">
                <span className="stat-label">Total Documents</span>
                <strong className="stat-value">{documents.length}</strong>
              </div>

              <div className="stat-card">
                <span className="stat-label">Total Storage</span>
                <strong className="stat-value">{totalStorageMB} MB</strong>
              </div>

              <div className="stat-card">
                <span className="stat-label">File Types</span>
                <strong className="stat-value">{fileTypes}</strong>
              </div>
            </div>

            <div className="panel">
              <h3>Selected files ({files.length})</h3>
              {uploading && (
                <p className="muted">Uploading and indexing files...</p>
              )}

              {uploadStatus && (
                <p role="status" aria-live="polite">
                  {uploadStatus}
                </p>
              )}
              {files.length === 0 ? (
                <p className="muted">No files selected yet.</p>
              ) : (
                files.map((file, index) => (
                  <div className="file-row" key={file.id}>
                    <span className="file-icon">▤</span>
                    <div className="file-details">
                      <strong>{file.name}</strong>
                      <small>
                        {(file.size / 1024).toFixed(1)} KB · {file.status}
                      </small>
                    </div>
                    <button
                      className="text-button"
                      onClick={() =>
                        setFiles((current) =>
                          current.filter((_, i) => i !== index),
                        )
                      }
                    >
                      Remove
                    </button>
                  </div>
                ))
              )}
              <p className="muted upload-note">
                Files are uploaded to the backend and indexed when the upload
                succeeds.
              </p>
            </div>
          </section>
        )}
        {activeTab === "Collection" && (
          <section className="page-content">
            <div className="section-heading">
              <div>
                <p className="eyebrow">DOCUMENT LIBRARY</p>
                <h2>Your collection</h2>
                <p>Browse documents indexed in your research workspace.</p>
              </div>

              <button
                className="primary-button"
                onClick={loadDocuments}
                disabled={documentsLoading}
              >
                {documentsLoading ? "Refreshing..." : "↻ Refresh"}
              </button>
            </div>

            <div className="panel">
              <div className="section-heading">
                <div>
                  <h3>Indexed documents</h3>
                  <p className="muted">
                    {filteredDocuments.length} of {documents.length} document(s)
                  </p>
                </div>
              </div>
              <input
                className="collection-search"
                type="search"
                placeholder="Search documents by filename..."
                value={documentSearch}
                onChange={(event) => setDocumentSearch(event.target.value)}
                aria-label="Search documents by filename"
              />

              {documentsLoading && documents.length === 0 && (
                <p className="muted">Loading documents...</p>
              )}

              {documentsError && (
                <p className="error-message" role="alert">
                  {documentsError}
                </p>
              )}

              {!documentsLoading &&
                !documentsError &&
                documents.length === 0 && (
                  <div className="empty-state">
                    <span className="upload-icon">▤</span>
                    <h3>No documents found</h3>
                    <p>Upload a PDF or TXT file to build your collection.</p>
                    <button
                      className="primary-button"
                      onClick={() => setActiveTab("Upload")}
                    >
                      Upload documents
                    </button>
                  </div>
                )}

              {filteredDocuments.map((doc) => (
                <div
                  className="file-row clickable-file-row"
                  key={doc.name}
                  onClick={() => setSelectedDocument(doc)}
                  role="button"
                  tabIndex={0}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" || event.key === " ") {
                      event.preventDefault();
                      setSelectedDocument(doc);
                    }
                  }}
                >
                  <span className="file-icon">▤</span>

                  <div className="file-details">
                    <strong>{doc.name}</strong>
                    <small>
                      {doc.type?.toUpperCase().replace(".", "") || "FILE"}
                      {" · "}
                      {(doc.size_bytes / 1024).toFixed(1)} KB
                    </small>
                  </div>

                  <span className="status-label">Indexed</span>
                </div>
              ))}
              {documents.length > 0 && filteredDocuments.length === 0 && (
                <p className="muted">No documents match "{documentSearch}".</p>
              )}
            </div>
            {selectedDocument && (
              <div className="panel document-details-panel">
                <div className="section-heading">
                  <div>
                    <p className="eyebrow">DOCUMENT DETAILS</p>
                    <h3>{selectedDocument.name}</h3>
                  </div>

                  <button
                    className="primary-button"
                    onClick={() => setSelectedDocument(null)}
                  >
                    Close
                  </button>
                </div>

                <div className="document-detail-row">
                  <span>File type</span>
                  <strong>
                    {selectedDocument.type?.toUpperCase().replace(".", "") ||
                      "Unknown"}
                  </strong>
                </div>

                <div className="document-detail-row">
                  <span>File size</span>
                  <strong>
                    {(selectedDocument.size_bytes / 1024).toFixed(1)} KB
                  </strong>
                </div>
              </div>
            )}
          </section>
        )}
        {activeTab === "Explorer" && (
          <section className="page-content explorer-page">
            <div className="welcome-block">
              <div className="welcome-icon">✳</div>
              <p className="eyebrow">YOUR RESEARCH ASSISTANT</p>
              <h2>Explore your research.</h2>
              <p>
                Ask a question, compare evidence, and get answers grounded in
                your indexed documents.
              </p>
            </div>

            <form className="question-box" onSubmit={askQuestion}>
              <label htmlFor="research-question">Research question</label>
              <textarea
                id="research-question"
                value={question}
                onChange={(event) => setQuestion(event.target.value)}
                placeholder="Ask about vulnerabilities, compare findings, or investigate a topic..."
                rows={4}
              />
              <div className="question-footer">
                <span className="muted">Answers use your research corpus</span>
                <button
                  className="primary-button"
                  type="submit"
                  disabled={loading || !question.trim()}
                >
                  {loading ? "Researching..." : "Run research →"}
                </button>
              </div>
            </form>

            {error && <div className="error-message">{error}</div>}

            {loading && (
              <div className="panel loading-panel">
                <div className="loading-spinner" />
                <div>
                  <strong>Investigating your question</strong>
                  <p className="muted">
                    Retrieving evidence and checking the answer...
                  </p>
                </div>
              </div>
            )}

            {result && (
              <section className="panel answer-panel">
                <div className="answer-heading">
                  <div>
                    <p className="eyebrow">RESEARCH RESULT</p>
                    <h3>Answer</h3>
                  </div>
                  <span className="result-badge">Completed</span>
                </div>

                <div className="answer-text">
                  {result.answer || "No answer was returned."}
                </div>

                <div className="metrics-row">
                  <div>
                    <small>Latency</small>
                    <strong>
                      {result.latency != null
                        ? `${Number(result.latency).toFixed(2)} s`
                        : "—"}
                    </strong>
                  </div>
                  <div>
                    <small>Cost</small>
                    <strong>
                      {result.cost != null
                        ? `$${Number(result.cost).toFixed(4)}`
                        : "—"}
                    </strong>
                  </div>
                  <div>
                    <small>Citations</small>
                    <strong>{result.citations?.length ?? 0}</strong>
                  </div>
                </div>

                <div className="citations-section">
                  <h3>Evidence and citations</h3>

                  {result.evidence?.length ? (
                    result.evidence.map((item, index) => (
                      <div className="citation-card" key={item.id || index}>
                        <span className="citation-number">{index + 1}</span>
                        <div>
                          <strong>
                            {item.source || item.id || `Source ${index + 1}`}
                          </strong>
                          {item.id && <small>{item.id}</small>}
                          {item.text && (
                            <details className="evidence-excerpt">
                              <summary>View evidence excerpt</summary>
                              <p>
                                {item.text.slice(0, 1200)}
                                {item.text.length > 1200 ? "..." : ""}
                              </p>
                            </details>
                          )}
                        </div>
                      </div>
                    ))
                  ) : result.citations?.length ? (
                    result.citations.map((citation, index) => (
                      <div className="citation-card" key={index}>
                        <span className="citation-number">{index + 1}</span>
                        <div>
                          <strong>
                            {typeof citation === "string"
                              ? citation
                              : citation.source ||
                                citation.id ||
                                citation.chunk_id ||
                                JSON.stringify(citation)}
                          </strong>
                        </div>
                      </div>
                    ))
                  ) : (
                    <p className="muted">
                      No citations were returned for this answer.
                    </p>
                  )}
                </div>

                {result.trace_id && (
                  <p className="trace-line">
                    Trace ID: <code>{result.trace_id}</code>
                  </p>
                )}
              </section>
            )}
          </section>
        )}

        {activeTab === "Chat" && (
          <section className="page-content chat-page">
            <p className="eyebrow">ESEARCH WORKSPACE</p>
            <h2>Research chat</h2>
            <p className="chat-subtitle">
              Ask questions about your indexed documents.
            </p>

            <div className="chat-conversation">
              {chatMessages.length === 0 && (
                <div className="chat-welcome">
                  <h3>What would you like to research?</h3>
                  <p>
                    Ask a question and get answers grounded in your documents.
                  </p>
                </div>
              )}

              {chatMessages.map((message, index) => (
                <div
                  key={index}
                  className={`chat-row ${
                    message.role === "user" ? "user-row" : "assistant-row"
                  }`}
                >
                  <div className="chat-message-content">
                    <span className="chat-role">
                      {message.role === "user" ? "You" : "Research Assistant"}
                    </span>

                    <div className="chat-message-text">{message.content}</div>

                    {message.role === "assistant" &&
                      message.citations?.length > 0 && (
                        <div className="chat-sources">
                          <span>Sources</span>
                          {message.citations.map((citation, i) => (
                            <div key={i} className="chat-source">
                              {typeof citation === "string"
                                ? citation
                                : citation.source ||
                                  citation.doc_id ||
                                  citation.chunk_id ||
                                  JSON.stringify(citation)}
                            </div>
                          ))}
                        </div>
                      )}
                  </div>
                </div>
              ))}

              {chatLoading && (
                <div className="chat-row assistant-row">
                  <div className="chat-message-content">
                    <span className="chat-role">Research Assistant</span>
                    <p className="chat-loading">
                      Researching your documents...
                    </p>
                  </div>
                </div>
              )}

              {chatError && (
                <p className="chat-error" role="alert">
                  {chatError}
                </p>
              )}
            </div>

            <form onSubmit={sendChatMessage} className="chat-composer">
              <input
                value={chatQuestion}
                onChange={(event) => setChatQuestion(event.target.value)}
                placeholder="Ask anything about your documents..."
                disabled={chatLoading}
                required
              />
              <button
                type="submit"
                disabled={chatLoading || !chatQuestion.trim()}
                aria-label="Send message"
                title="Send message"
              >
                ↑
              </button>
            </form>

            <p className="chat-disclaimer">
              Answers are generated from your indexed research documents.
            </p>
          </section>
        )}
      </main>

      <aside className="evidence-sidebar">
        <div className="evidence-header">
          <p className="eyebrow">RESEARCH CONTEXT</p>
          <h3>Evidence panel</h3>
          <p>Review the citations returned by your latest query.</p>
        </div>

        {result?.evidence?.length ? (
          <div className="evidence-list">
            {result.evidence.map((item, index) => (
              <div className="evidence-card" key={item.id || index}>
                <span className="evidence-tag">SOURCE {index + 1}</span>

                <strong>{item.source || `Source ${index + 1}`}</strong>

                {item.id && <small>{item.id}</small>}

                {item.text ? (
                  <p>{item.text}</p>
                ) : (
                  <p className="muted">No excerpt available for this source.</p>
                )}
              </div>
            ))}
          </div>
        ) : (
          <div className="evidence-empty">
            <span>⌕</span>
            <strong>No evidence to display</strong>
            <p>Run a research query to see returned evidence here.</p>
          </div>
        )}

        <div className="context-footer">
          <span className="status-dot" />
          <span>Connected to local API when available</span>
        </div>
      </aside>
    </div>
  );
}
