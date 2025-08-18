## 🌌 Cosmo Agent — Hệ thống Multi‑Agent tìm kiếm ảnh + VLM

Cosmo là một workflow multi‑agent xây dựng trên Google ADK (Agents Development Kit) để:
- tách một câu hỏi (ưu tiên tiếng Nhật) thành các truy vấn con,
- tìm ảnh liên quan qua Milvus Search API,
- phân tích ảnh song song bằng nhiều VLM agents,
- và tổng hợp câu trả lời cuối cùng.

### 🎯 Tính năng chính
- **🤖 Orchestration bằng Google ADK**: `CosmoFlowAgent` điều phối Main → Search (song song) → VLM (song song) → Aggregator
- **🔑 Xoay vòng API key**: tự gán key theo `agent_id` để tránh quota limit
- **⚡ Song song hóa**: phân lô ảnh và chạy nhiều VLM agents cùng lúc
- **🧩 Tối ưu prompt**: Main/VLM dùng prompt chuyên biệt (tiếng Nhật)
- **🪵 Logging rõ ràng**: theo dõi sự kiện/phiên làm việc

### 🏗️ Quy trình
```
User Query (JA) → Main Agent (chia tối đa 3 sub‑queries) → SearchAgents (gọi Milvus API, tải ảnh) → VLM Agents (phân tích ảnh) → Aggregator (tổng hợp câu trả lời JA)
```

## 📦 Cài đặt

### Yêu cầu
- Python 3.10+
- API key Gemini (Google Generative AI)
- Kết nối tới Milvus Search API (HTTP)

### Cài dependencies
```bash
pip install -r requirements.txt
```

### Thiết lập môi trường
```bash
cp config.env.example config.env
# Mở file config.env và điền GEMINI_API_KEY_* hoặc GEMINI_API_KEY
```

### Cấu hình ứng dụng (`config/config.yaml`)
Các mục chính:
```yaml
app:
  name: "cosmo_app"
  user_id: "cosmo_user"
  session_id: "cosmo_session"

milvus:
  search_url: "http://70.29.215.74:36053/search_default_base64"
  default_top_k: 5

agents:
  main:
    name: "MainCoordinator"
    model: "gemini-2.5-pro"
    temperature: 0.2
    agent_id: 0
    output_key: "task_results"
  search:
    count: 3
    model: "gemini-2.0-flash"
  vlm:
    count: 5
    model: "gemini-2.5-pro"
  aggregator:
    name: "AggregatorAgent"
    model: "gemini-2.5-pro"
    agent_id: 99

paths:
  tools_results: "tools_results"
```

Tùy chọn (chưa bật mặc định) cho Local VLM nằm dưới `local_vlm` trong YAML và đoạn code mẫu comment trong `agent/agent.py`.

## 🚀 Chạy thử
```bash
python main.py
```
- Chọn 1 câu hỏi mẫu hoặc nhập câu hỏi của bạn (nên dùng tiếng Nhật để khớp prompt).
- Kết quả in ra console; ảnh tải về lưu ở `tools_results/` với tên dạng `doc_<id>.png`.

Bạn cũng có thể gọi trực tiếp trong mã:
```python
from main import run_cosmo_workflow
import asyncio
asyncio.run(run_cosmo_workflow("温度差荷重の記号について教えてください"))
```

## 🔧 Thành phần chính

- **`workflow/cosmo_workflow.py`**: định nghĩa `CosmoFlowAgent` (kế thừa `BaseAgent`) và luồng orchestration. Sử dụng `ParallelAgent` cho SearchAgents.
- **`agent/agent.py`**: tạo `main_agent`, danh sách `search_agents` (3), `vlm_agents` (5), và `aggregator_agent`. Dùng xoay vòng API key theo `agent_id`.
- **`agent/load_agent.py`**: `ApiKeyManager` và hàm `create_agent_with_api_key_rotation`.
- **`tools/tools.py`**: lớp `Api` với tool `image_search(query, k)` gọi Milvus API, giải mã base64 và lưu ảnh vào `tools_results/`.
- **`workflow/vlm_runner.py`**: phân lô ảnh cho các VLM agent và chạy song song qua ADK `Runner`.
- **`prompt.py`**: prompt cho Main/Search/VLM/Aggregator (Main & VLM bắt buộc xuất/nhập tiếng Nhật).
- **`utils/helper_workflow.py`**: tiện ích đọc cấu hình, chia ảnh, chuẩn bị input kèm ảnh cho VLM.

## 📁 Cấu trúc dự án
```
Cosmo/
├─ main.py
├─ config/
│  └─ config.yaml
├─ config.env.example
├─ requirements.txt
├─ prompt.py
├─ agent/
│  ├─ agent.py
│  └─ load_agent.py
├─ tools/
│  └─ tools.py
├─ utils/
│  └─ helper_workflow.py
└─ workflow/
   ├─ cosmo_workflow.py
   └─ vlm_runner.py
```

## 🧪 Ghi chú vận hành
- SearchAgents nhận sub‑query từ MainAgent qua `ctx.session.state["search_query_i"]` và gọi `image_search`.
- VLMAgents nhận câu hỏi gốc + ảnh (nếu file tồn tại ở `tools_results/`) và trả lời bằng tiếng Nhật theo luật chặt chẽ (không suy diễn).
- Aggregator gộp các câu trả lời của VLM thành câu trả lời cuối cùng (tiếng Nhật, không giải thích).

## 🩺 Troubleshooting
- **API keys**: đảm bảo `config.env` có `GEMINI_API_KEY` hoặc `GEMINI_API_KEY_1..n`. Biến môi trường được nạp qua `python-dotenv`.
- **Milvus API**: kiểm tra `config/config.yaml` mục `milvus.search_url` và kết nối mạng. API trả JSON có `image_base64`.
- **File ảnh**: nếu ảnh không tải được, kiểm tra quyền ghi thư mục `tools_results/` và dung lượng đĩa.
- **Phiên/ADK**: lỗi khởi tạo `Runner`/`SessionService` thường do cấu hình `app` hoặc môi trường python.

## 🤝 Đóng góp
1) Fork repo  2) Tạo branch  3) Commit  4) Mở Pull Request

## 📄 License
Chưa chỉ định rõ trong repository. Vui lòng liên hệ tác giả nếu cần thông tin license.

## 📞 Hỗ trợ
- Issues: mở ticket trên GitHub repo này
- Email/Liên hệ: xem trang cá nhân của tác giả

---

🌌 Cosmo Agent — Orchestrating image search and VLM reasoning.
