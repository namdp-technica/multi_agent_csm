# 🌌 Cosmo Agent - Multi-Agent Research System

Cosmo Agent là một hệ thống nghiên cứu thông minh sử dụng kiến trúc multi-agent để phân tích và trả lời các câu hỏi nghiên cứu phức tạp. Hệ thống sử dụng Google's Agent Development Kit (ADK) và ChromaDB để tạo ra một workflow hiệu quả với 1 main agent và 10 sub-agents chuyên biệt.

## 🎯 Tính năng chính

- **🤖 Multi-Agent Architecture**: 1 main agent điều phối + 10 sub-agents thực thi song song
- **🔍 Vector Database Search**: Tích hợp ChromaDB với BGE-M3 embeddings
- **⚡ Parallel Processing**: Thực thi các tác vụ con song song để tối ưu hóa thời gian
- **🔄 Retry Mechanism**: Tự động thử lại với các truy vấn được cải thiện
- **📊 Detailed Logging**: Theo dõi chi tiết timing và workflow
- **💬 Interactive Mode**: Giao diện command-line thân thiện

## 🏗️ Kiến trúc hệ thống

### Workflow Process
```
User Query → Main Agent (Task Decomposition) 
           ↓
           Sub-Agents (Parallel Execution)
           ↓
           Main Agent (Evaluation & Retry if needed)
           ↓
           Final Response + Follow-up Question
```

### Core Components

1. **Main Agent (Coordinator)**
   - Model: `gemini-2.5-pro`
   - Nhiệm vụ: Phân tách task, đánh giá kết quả, tổng hợp
   - Output: `evaluation_result`

2. **Sub-Agents (10 Specialists)**
   - Model: `gemini-2.0-flash`
   - Nhiệm vụ: Vector database search
   - Tools: Semantic search với ChromaDB
   - Output: `search_result`

3. **Vector Database**
   - ChromaDB với BGE-M3 embeddings
   - Persistent storage trong `/chroma_db`

## 📦 Cài đặt

### Yêu cầu hệ thống
- Python 3.8+
- Google AI API key
- Đủ RAM để chạy BGE-M3 model

### Cài đặt dependencies

```bash
pip install google-genai google-adk
pip install chromadb sentence-transformers
pip install asyncio logging
```

### Cấu hình

1. **Google AI API Key**: Thiết lập API key trong environment
```bash
export GOOGLE_API_KEY="your_api_key_here"
```

2. **ChromaDB**: Database sẽ được tạo tự động trong thư mục `./chroma_db`

## 🚀 Cách sử dụng

### Chế độ Fixed Query (Default)
```bash
python main.py
```
Hệ thống sẽ chạy với câu hỏi mặc định về quantum computing và cryptography.

### Chế độ Interactive
Chỉnh sửa `main.py` để sử dụng interactive mode:
```python
# Trong hàm main(), thay thế:
# result = await run_cosmo_workflow(fixed_query)
# Bằng:
await interactive_mode()
```

### Batch Processing
```python
queries = [
    "AI trong giáo dục",
    "Blockchain và an ninh mạng",
    "Machine learning trong y tế"
]
results = await batch_process_queries(queries)
```

## 📁 Cấu trúc dự án

```
/home/namdp/Cosmo/
├── main.py                    # Entry point chính
├── prompt.py                  # Prompts cho agents
├── agent/
│   ├── agent.py              # Định nghĩa main agent & sub-agents
│   └── __pycache__/
├── workflow/
│   ├── cosmo_workflow.py     # Logic workflow chính
│   └── __pycache__/
├── tools/
│   ├── __init__.py
│   ├── tools.py              # VectorDatabaseTool
│   └── __pycache__/
├── chroma_db/                 # ChromaDB persistent storage
│   ├── chroma.sqlite3
│   └── vector_data/
└── __pycache__/
```

## 🔧 Các thành phần chi tiết

### 1. Main Agent (Coordinator)
- **Model**: Gemini 2.5 Pro
- **Chức năng**:
  - **Mode 1**: Phân tách user query thành các sub-tasks
  - **Mode 2**: Đánh giá kết quả từ sub-agents và quyết định retry
- **Output**: JSON format với task list hoặc evaluation result

### 2. Sub-Agents (Specialists)
- **Số lượng**: 10 agents (`agent_search_1` đến `agent_search_10`)
- **Model**: Gemini 2.0 Flash
- **Tools**: Vector database semantic search
- **Chức năng**: Thực hiện tìm kiếm chuyên biệt theo assigned tasks

### 3. Vector Database Tool
- **Engine**: ChromaDB
- **Embeddings**: BGE-M3 (BAAI/bge-m3)
- **Features**:
  - Semantic search
  - Persistent storage
  - Sample data auto-loading

### 4. Workflow Engine (CosmoWorkflow)
- **Base Class**: BaseAgent từ Google ADK
- **Steps**:
  1. Task Decomposition (Main Agent)
  2. Parallel Sub-Agent Execution
  3. Result Evaluation (Main Agent)
  4. Retry if needed
  5. Final Summary + Follow-up Question

## 📊 Monitoring & Logging

### Timing Breakdown
Hệ thống track chi tiết thời gian thực thi:
- **Step 1**: Task Decomposition
- **Step 2**: Sub-agents Parallel Execution  
- **Step 3**: Main Agent Evaluation
- **Step 4**: Retry (nếu cần)
- **Total**: Tổng thời gian workflow

### Session State
```json
{
  "user_query": "Original user question",
  "task_list": [...],
  "search_results": {...},
  "evaluation_result": {...},
  "workflow_status": "completed",
  "execution_time": 15.67,
  "workflow_timing": {
    "step1_duration": 3.2,
    "step2_duration": 8.5,
    "step3_duration": 2.1,
    "step4_duration": 1.87
  }
}
```

## 🎛️ Cấu hình tùy chỉnh

### Thay đổi số lượng Sub-Agents
Trong `agent/agent.py`:
```python
# Thay đổi range để có nhiều hơn/ít hơn 10 agents
for i in range(1, 15):  # Tạo 14 agents
    agent_name = f"agent_search_{i}"
    # ...
```

### Custom Prompts
Chỉnh sửa `prompt.py` để thay đổi behavior:
- `MAIN_AGENT_PROMPT`: Logic coordinator
- `SEARCH_AGENT_PROMPT`: Sub-agent instructions

### Database Configuration
Trong `tools/tools.py`:
```python
# Thay đổi đường dẫn database
db_path = "./custom_chroma_db"
collection_name = "my_knowledge_base"
```

## 🔬 Testing & Development

### Test cơ bản
```bash
python main.py
```

### Debug Mode
Bật detailed logging:
```python
logging.basicConfig(level=logging.DEBUG)
```

### Performance Testing
```python
# Test với multiple queries
queries = ["Query 1", "Query 2", "Query 3"]
start_time = time.time()
results = await batch_process_queries(queries)
total_time = time.time() - start_time
print(f"Total time for {len(queries)} queries: {total_time:.2f}s")
```

## 🚧 Troubleshooting

### Common Issues

1. **API Key Issues**
```bash
# Kiểm tra API key
echo $GOOGLE_API_KEY
```

2. **Memory Issues với BGE-M3**
```python
# Giảm batch size hoặc sử dụng model nhỏ hơn
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
```

3. **ChromaDB Permission Errors**
```bash
# Đảm bảo quyền write
chmod -R 755 ./chroma_db/
```

4. **Workflow Timeout**
```python
# Tăng timeout trong async operations
await asyncio.wait_for(workflow_task, timeout=300)
```

## 📈 Performance Notes

- **Parallel Efficiency**: 10 sub-agents có thể chạy song song
- **Model Selection**: 
  - Gemini 2.5 Pro cho reasoning phức tạp
  - Gemini 2.0 Flash cho speed
- **Memory Usage**: BGE-M3 cần ~2GB RAM
- **Typical Execution**: 10-30 giây cho một query phức tạp

## 🔮 Future Enhancements

- [ ] Web interface với FastAPI
- [ ] Support multiple vector databases
- [ ] Dynamic agent scaling
- [ ] Real-time streaming responses
- [ ] Custom embedding fine-tuning
- [ ] Integration với external APIs
- [ ] Multi-language support

## 🤝 Contributing

1. Fork repository
2. Tạo feature branch
3. Commit changes
4. Push và tạo Pull Request

## 📄 License

Dự án này thuộc về namdp-technica. Chi tiết license xem trong file LICENSE.

## 📞 Support

- **Repository**: namdp-technica/multi_agent_cosmo  
- **Issues**: Tạo issue trên GitHub
- **Email**: Contact qua GitHub profile

---

*🌌 Cosmo Agent - Connecting knowledge across the universe of information*
