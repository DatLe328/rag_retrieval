import React, { useState } from "react";
import "bootstrap/dist/css/bootstrap.min.css";
import { Modal, Button } from "react-bootstrap"; // 1. Import Modal và Button
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

function App() {
  const [messages, setMessages] = useState<{ sender: string; text: string }[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [typingText, setTypingText] = useState("");

  // 2. Cập nhật State để quản lý Modal
  const [showDebugModal, setShowDebugModal] = useState(false); // State để bật/tắt modal
  const [debugData, setDebugData] = useState<object | null>(null); // State để lưu JSON response

  const [multiN, setMultiN] = useState(1);
  const [topK, setTopK] = useState(10);
  const [alpha, setAlpha] = useState(0.5);

  // Hàm đóng modal
  const handleCloseModal = () => setShowDebugModal(false);
  // Hàm mở modal
  const handleShowModal = () => setShowDebugModal(true);

  const sendMessage = async () => {
    if (!input.trim() || loading) return;

    const userMessage = { sender: "user", text: input };
    setMessages([...messages, userMessage]);
    setInput("");
    setLoading(true);
    setTypingText("");
    setDebugData(null);

    try {
      const res = await fetch("http://localhost:5000/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: input,
          multi_n: multiN,
          top_k: topK,
          alpha: alpha,
          user_id: "user123",
        }),
      });

      const data = await res.json();
      setDebugData(data);
      const fullText = data.generated_answer;

      let index = 0;
      const typingInterval = setInterval(() => {
        if (index < fullText.length) {
          setTypingText((prev) => prev + fullText[index]);
          index++;
        } else {
          clearInterval(typingInterval);
          setMessages((prev) => [...prev, { sender: "bot", text: fullText }]);
          setTypingText("");
          setLoading(false);
        }
      }, 5);
    } catch (error) {
      const errorResponse = {
        error: "Connection failed",
        message: (error as Error).message,
      };
      setDebugData(errorResponse);
      setMessages((prev) => [
        ...prev,
        { sender: "bot", text: "⚠️ Lỗi kết nối tới server." },
      ]);
      setLoading(false);
    }
  };

  return (
    // 3. KHÔI PHỤC LAYOUT GỐC ĐỂ KHUNG CHAT LUÔN Ở GIỮA
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
        <div className="card-header bg-primary text-white d-flex justify-content-between align-items-center fw-bold fs-4">
            <span>💬 Chatbot Demo</span>
            {/* Nút này giờ sẽ mở Modal */}
            <button
            className="btn btn-sm btn-outline-light"
            onClick={handleShowModal}
            title="Open Debug Panel"
            >
            {"{...}"}
            </button>
        </div>

        {/* Các phần còn lại của card chat giữ nguyên */}
        <div
          className="card-body p-3 overflow-auto"
          style={{ background: "#f8f9fa", height: "calc(100% - 210px)" }}
        >
          {/* ... mapping messages ... */}
          {messages.map((msg, index) => (
            <div key={index} className={`d-flex mb-2 ${ msg.sender === "user" ? "justify-content-end" : "justify-content-start"}`}>
                <div className={`p-2 rounded-3 ${ msg.sender === "user" ? "bg-primary text-white" : "bg-white border"}`} style={{ maxWidth: "75%" }}>
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.text}</ReactMarkdown>
                </div>
            </div>
          ))}
          {/* ... typing text and loading indicators ... */}
          {typingText && <div className="d-flex justify-content-start mb-2"><div className="p-2 bg-white border rounded-3" style={{ maxWidth: "75%" }}><ReactMarkdown remarkPlugins={[remarkGfm]}>{typingText + "▋"}</ReactMarkdown></div></div>}
          {loading && !typingText && <div className="d-flex justify-content-start mb-2"><div className="p-2 bg-white border rounded-3 d-flex align-items-center" style={{ maxWidth: "70%" }}><div className="spinner-border spinner-border-sm me-2 text-secondary" role="status"></div><span>Đang trả lời...</span></div></div>}
        </div>

        <div className="card-body p-3 border-top bg-white">
          {/* ... sliders ... */}
          <div className="row">
            <div className="col-md-4"><label className="form-label small">Đa truy vấn (multi_n): <strong>{multiN}</strong></label><input type="range" className="form-range" min="1" max="10" step="1" value={multiN} onChange={(e) => setMultiN(parseInt(e.target.value))} disabled={loading} /></div>
            <div className="col-md-4"><label className="form-label small">Số kết quả (top_k): <strong>{topK}</strong></label><input type="range" className="form-range" min="1" max="20" step="1" value={topK} onChange={(e) => setTopK(parseInt(e.target.value))} disabled={loading} /></div>
            <div className="col-md-4"><label className="form-label small">Trọng số Hybrid (alpha): <strong>{alpha}</strong></label><input type="range" className="form-range" min="0" max="1" step="0.05" value={alpha} onChange={(e) => setAlpha(parseFloat(e.target.value))} disabled={loading} /></div>
          </div>
        </div>

        <div className="card-footer d-flex p-2">
          {/* ... input and send button ... */}
          <input className="form-control me-2" placeholder="Nhập câu hỏi..." value={input} onChange={(e) => setInput(e.target.value)} onKeyDown={(e) => e.key === "Enter" && sendMessage()} disabled={loading} />
          <button className="btn btn-primary" onClick={sendMessage} disabled={loading}>{loading ? (<><span className="spinner-border spinner-border-sm me-2" role="status"></span>Gửi...</>) : ("Gửi")}</button>
        </div>
      </div>

      {/* 4. THÊM COMPONENT MODAL CỦA REACT BOOTSTRAP */}
      <Modal show={showDebugModal} onHide={handleCloseModal} size="lg" centered>
        <Modal.Header closeButton>
          <Modal.Title>🐞 Debug Panel</Modal.Title>
        </Modal.Header>
        <Modal.Body style={{ maxHeight: "70vh", overflowY: "auto" }}>
          <pre className="bg-dark text-white p-3 rounded">
            <code>
              {debugData
                ? JSON.stringify(debugData, null, 2)
                : "Gửi một câu hỏi để xem response..."}
            </code>
          </pre>
        </Modal.Body>
        <Modal.Footer>
          <Button variant="secondary" onClick={handleCloseModal}>
            Đóng
          </Button>
        </Modal.Footer>
      </Modal>
    </div>
  );
}

export default App;