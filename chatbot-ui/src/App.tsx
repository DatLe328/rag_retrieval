import React, { useState } from "react";
import "bootstrap/dist/css/bootstrap.min.css";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

function App() {
  const [messages, setMessages] = useState<{ sender: string; text: string }[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  const sendMessage = async () => {
    if (!input.trim() || loading) return;

    const userMessage = { sender: "user", text: input };
    setMessages([...messages, userMessage]);
    setInput("");
    setLoading(true);

    try {
      const res = await fetch("http://localhost:5000/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: input }),
      });

      const data = await res.json();
      const botMessage = { sender: "bot", text: data.generated_answer };
      setMessages((prev) => [...prev, botMessage]);
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        { sender: "bot", text: "⚠️ Lỗi kết nối tới server." },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      className="d-flex justify-content-center align-items-center vh-100 bg-light"
      style={{ width: "100vw" }}
    >
      <div
        className="card shadow-lg"
        style={{
          width: "80%",
          height: "90%",
          borderRadius: "20px",
          overflow: "hidden",
        }}
      >
        <div className="card-header bg-primary text-white text-center fw-bold fs-4">
          💬 Chatbot Demo
        </div>

        <div
          className="card-body p-3 overflow-auto"
          style={{ background: "#f8f9fa", height: "calc(100% - 110px)" }}
        >
          {messages.map((msg, index) => (
            <div
              key={index}
              className={`d-flex mb-2 ${
                msg.sender === "user" ? "justify-content-end" : "justify-content-start"
              }`}
            >
              <div
                className={`p-2 rounded-3 ${
                  msg.sender === "user"
                    ? "bg-primary text-white"
                    : "bg-white border"
                }`}
                style={{ maxWidth: "75%" }}
              >
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  components={{
                    code({ node, inline, className, children, ...props }) {
                      return !inline ? (
                        <pre
                          style={{
                            background: "#f5f5f5",
                            padding: "8px",
                            borderRadius: "8px",
                            overflowX: "auto",
                          }}
                        >
                          <code {...props}>{children}</code>
                        </pre>
                      ) : (
                        <code
                          style={{
                            background: "#eee",
                            padding: "2px 4px",
                            borderRadius: "4px",
                          }}
                          {...props}
                        >
                          {children}
                        </code>
                      );
                    },
                  }}
                >
                  {msg.text}
                </ReactMarkdown>
              </div>
            </div>
          ))}

          {/* Loading indicator */}
          {loading && (
            <div className="d-flex justify-content-start mb-2">
              <div
                className="p-2 bg-white border rounded-3 d-flex align-items-center"
                style={{ maxWidth: "70%" }}
              >
                <div
                  className="spinner-border spinner-border-sm me-2 text-secondary"
                  role="status"
                ></div>
                <span>Đang trả lời...</span>
              </div>
            </div>
          )}
        </div>

        <div className="card-footer d-flex p-2">
          <input
            className="form-control me-2"
            placeholder="Nhập câu hỏi..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && sendMessage()}
            disabled={loading}
          />
          <button className="btn btn-primary" onClick={sendMessage} disabled={loading}>
            {loading ? (
              <>
                <span
                  className="spinner-border spinner-border-sm me-2"
                  role="status"
                ></span>
                Gửi...
              </>
            ) : (
              "Gửi"
            )}
          </button>
        </div>
      </div>
    </div>
  );
}

export default App;
