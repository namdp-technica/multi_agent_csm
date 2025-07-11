# Multi-Agent System với Vector Database

Hệ thống multi-agent sử dụng Google ADK với 10 sub-agents và ChromaDB vector database với BGE-M3 embedding model.

## 🚀 Tính năng

- **10 Sub-Agents**: Mỗi agent có chuyên môn riêng và khả năng truy vấn vector database
- **Vector Database**: ChromaDB với BGE-M3 embedding model cho semantic search
- **Main Coordinator**: Agent chính điều phối và phân phối tasks
- **Parallel Processing**: Xử lý song song các tasks độc lập
- **Semantic Search**: Tìm kiếm thông tin dựa trên ngữ nghĩa

## 📋 Yêu cầu

- Python 3.8+
- Google Gemini API Key
- Các dependencies trong `requirements.txt`

## 🛠️ Cài đặt

1. **Clone repository và cài đặt dependencies:**
```bash
pip install -r requirements.txt
```

2. **Thiết lập API Key:**
```bash
export GEMINI_API_KEY="your_gemini_api_key_here"
```

3. **Chạy test vector database:**
```bash
python tool.py
```

4. **Chạy hệ thống multi-agent:**
```bash
python agent.py
```

## 📁 Cấu trúc Project

```
Cosmo/
├── agent.py                 # Main multi-agent system
├── tool.py                  # Vector database tool với ChromaDB + BGE-M3
├── vector_tool_wrapper.py   # Wrapper cho Google ADK integration
├── requirements.txt         # Dependencies
├── README.md               # Hướng dẫn sử dụng
├── Data/                   # Thư mục chứa dữ liệu
└── chroma_db/              # ChromaDB database (tự động tạo)
```

## 🔧 Cách sử dụng

### 1. Khởi tạo Vector Database

```python
from tool import VectorDatabaseTool

# Khởi tạo database
db_tool = VectorDatabaseTool()

# Thêm documents
documents = [
    {
        "id": "doc_1",
        "text": "Nội dung document 1",
        "metadata": {"category": "tech", "topic": "AI"}
    }
]
db_tool.add_documents(documents)

# Truy vấn
results = db_tool.semantic_search("AI applications", n_results=5)
```

### 2. Sử dụng Multi-Agent System

```python
from agent import MultiAgentSystem

# Khởi tạo hệ thống
system = MultiAgentSystem()

# Xử lý request
result = await system.process_request("Tạo ứng dụng web quản lý công việc")
```

### 3. Tích hợp với Google ADK

```python
from vector_tool_wrapper import VectorDatabaseAgentTool

# Tạo tool cho agent
vector_tool = VectorDatabaseAgentTool()

# Sử dụng trong agent
result = await vector_tool.execute("AI in healthcare", {"n_results": 3})
```

## 🤖 Các Agents

### Main Coordinator Agent
- **Chức năng**: Điều phối và phân phối tasks
- **Model**: Gemini 2.5 Pro Preview
- **Tools**: 10 sub-agents + vector database

### Sub-Agents (Agent1-Agent10)
- **Chức năng**: Xử lý tasks chuyên biệt
- **Model**: Gemini 2.0 Flash
- **Tools**: Vector database query
- **Quy trình**: Phân tích → Tìm kiếm → Xử lý → Tổng hợp

## 🔍 Vector Database Features

### ChromaDB + BGE-M3
- **Embedding Model**: BAAI/bge-m3
- **Database**: ChromaDB persistent storage
- **Search**: Semantic similarity search
- **Filtering**: Metadata-based filtering
- **Categories**: healthcare, finance, education, automotive, AI, blockchain, IoT, cloud, security, data science

### API Methods
- `query_database()`: Truy vấn cơ bản
- `semantic_search()`: Tìm kiếm ngữ nghĩa với filter
- `add_documents()`: Thêm documents mới
- `update_document()`: Cập nhật document
- `delete_documents()`: Xóa documents
- `get_collection_stats()`: Thống kê database

## 📊 Sample Data

Hệ thống tự động load 10 sample documents về các chủ đề:
- AI trong healthcare
- Machine learning trong finance
- NLP trong education
- Computer vision trong automotive
- Deep learning và transformers
- Blockchain technology
- IoT applications
- Cloud computing
- Cybersecurity
- Data science

## 🎯 Example Queries

```python
# Ví dụ requests cho hệ thống
example_requests = [
    "Tạo một ứng dụng web đơn giản để quản lý danh sách công việc với giao diện đẹp",
    "Phân tích xu hướng thị trường công nghệ năm 2024 và đưa ra dự đoán cho 2025",
    "Dịch và tối ưu hóa một bài viết về AI từ tiếng Anh sang tiếng Việt"
]
```

## 🔧 Configuration

### Environment Variables
```bash
GEMINI_API_KEY=your_api_key_here
```

### Database Configuration
```python
# Trong tool.py
db_path = "./chroma_db"           # Đường dẫn database
collection_name = "knowledge_base" # Tên collection
```

## 🚀 Performance

- **Parallel Processing**: Các tasks độc lập được xử lý song song
- **Semantic Search**: Tìm kiếm nhanh với BGE-M3 embeddings
- **Caching**: ChromaDB persistent storage
- **Scalability**: Dễ dàng thêm agents và documents mới

## 🐛 Troubleshooting

### Lỗi thường gặp:
1. **API Key không hợp lệ**: Kiểm tra GEMINI_API_KEY
2. **Dependencies thiếu**: Chạy `pip install -r requirements.txt`
3. **Database lỗi**: Xóa thư mục `chroma_db/` và chạy lại
4. **Memory issues**: Giảm số lượng documents hoặc n_results

### Debug Mode:
```python
# Thêm logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📈 Monitoring

### Database Stats
```python
stats = db_tool.get_collection_stats()
print(f"Total documents: {stats['total_documents']}")
```

### Agent Performance
```python
# Xem history của agent
history = coordinator.get_history()
```

## 🔄 Updates

- **Thêm documents mới**: Sử dụng `add_documents()`
- **Cập nhật agents**: Chỉnh sửa instructions trong `agent.py`
- **Thay đổi model**: Cập nhật model name trong configuration

## 📝 License

MIT License - Xem file LICENSE để biết thêm chi tiết. # multi_agent_cosmo
# multi_agent_cosmo
