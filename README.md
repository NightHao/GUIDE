# GUIDE FastAPI 專案

本專案提供一套完整的圖譜建置與問題回答流程，所有邏輯皆封裝於 `app/` 目錄，並透過 FastAPI 對外提供 REST API 或可直接在 Python 程式中呼叫。若需要端點與資料依賴的更完整說明，請參考 [`GUIDE_API_REFERENCE.md`](GUIDE_API_REFERENCE.md)。

## 📂 目錄結構
```
GUIDE/
├── app/                 # FastAPI 應用程式（API、services、核心邏輯）
├── config/              # 相依套件等設定
├── data/                # 輸入、中間結果與輸出（詳見下方）
├── deployment/          # docker 與部署腳本
│   ├── docker/          # Dockerfile、Compose 設定
│   └── scripts/         # 部署腳本（deploy.sh）
├── .env.example         # 環境變數範例
├── test.py              # 呼叫 `QueryService` 的範例腳本
└── README.md            # 本文件
```

## ⚙️ 環境準備
1. 安裝套件
   ```bash
   pip install -r config/requirements.txt
   ```
2. 建立 `.env`
   ```bash
   cp .env.example .env
   ```
   至少需設定 `OPENAI_API_KEY=你的_API_Key`，其他變數可視需求調整。
3. 確認必要資料檔案（可使用既有範例或自行產生）：

   | 用途 | 預設路徑 | 說明 |
   | --- | --- | --- |
   | 圖譜建置輸入 | `data/input/entities_result.json` | LLM 擷取到的實體 / 關係原始輸出 |
   | 問題對應 Chunk | `data/input/retrieved_chunks_15.json` | 問題與資料片段的對應表 |
   | 問答快取 | `data/intermediate/entities_chunks.json` | 系統查詢後會自動更新，初始可為 `{}` |
   | 別名字典 | `data/intermediate/alias_dict.json` | `POST /api/v1/graph/build-aliases` 會產生 |
   | 最終圖譜 | `data/output/optimized_entity_graph.json` | `POST /api/v1/graph/optimize` 會產生 |

## 🚀 啟動 API
```bash
uvicorn app.main:app --reload --port 8000
```
- Swagger UI：<http://localhost:8000/docs>
- ReDoc：<http://localhost:8000/redoc>

## 🧱 圖譜處理 API 範例
若你已擁有中間檔案，可直接指定路徑並跳過相對應步驟。

```bash
# 1. 建置圖譜
curl -X POST http://localhost:8000/api/v1/graph/build \
  -H "Content-Type: application/json" \
  -d '{
        "entities_file": "data/input/entities_result.json",
        "output_path": "data/intermediate/entity_graph.json"
      }'

# 2. 圖譜處理（縮寫合併、規則擷取）
curl -X POST http://localhost:8000/api/v1/graph/process \
  -H "Content-Type: application/json" \
  -d '{
        "input_graph": "data/intermediate/entity_graph.json",
        "output_graph": "data/intermediate/processed_entity_graph.json",
        "output_log": "data/intermediate/log.json"
      }'

# 3. 描述最佳化
curl -X POST http://localhost:8000/api/v1/graph/optimize \
  -H "Content-Type: application/json" \
  -d '{
        "input_graph": "data/intermediate/processed_entity_graph.json",
        "output_graph": "data/output/optimized_entity_graph.json",
        "similarity_threshold": 0.9
      }'

# 4. 建立別名字典（若 alias_dict.json 尚未存在）
curl -X POST http://localhost:8000/api/v1/graph/build-aliases \
  -H "Content-Type: application/json" \
  -d '{
        "log_file": "data/intermediate/log.json",
        "output_file": "data/intermediate/alias_dict.json"
      }'
```

## ❓ 問答 API 範例
```bash
# 單題提問
temp='{
  "question": "What is V2G?",
  "graph_path": "data/output/optimized_entity_graph.json",
  "subgraph_distance": 2,
  "use_agentic_flow": true
}'

curl -X POST http://localhost:8000/api/v1/query/ask \
  -H "Content-Type: application/json" \
  -d "$temp"
```
範例回應結構：
```json
{
  "question": "What is V2G?",
  "answer": "... LLM 生成的最終回答 ...",
  "intent": {
    "category": "General Information Query",
    "explanation": "系統判斷為一般資訊提問",
    "confidence": null
  },
  "processing_time": 6.3,
  "entities_used": ["VEHICLE-TO-GRID", "EV"],
  "token_usage": null
}
```

> ⚠️ 若縮寫對應到多個節點（例如 `PA`），第一次呼叫會收到 **409** 並在 `detail.ambiguous_aliases` 列出候選。請重新送出 request 並於 `alias_overrides` 指定想要的全名，例如：

```bash
curl -X POST http://localhost:8000/api/v1/query/ask \
  -H "Content-Type: application/json" \
  -d '{
        "question": "What is PA?"
      }'

# 回傳 409，body 會出現 "ambiguous_aliases": [{"alias": "PA", "candidates": ["Platform Adapter", ...]}]

# 帶上 alias_overrides 後重新送出
curl -X POST http://localhost:8000/api/v1/query/ask \
  -H "Content-Type: application/json" \
  -d '{
        "question": "What is PA?",
        "alias_overrides": {"PA": "Platform Adapter"}
      }'
```
其他常用端點：
- `POST /api/v1/query/batch`：批次詢問並可輸出結果（`data/output/batch_results.json`）。
- `GET /api/v1/query/entities/{name}`：查詢個別節點與鄰居資訊。
- `GET /api/v1/query/history`：查看已儲存的提問紀錄與常見實體。

## 🐍 直接在 Python 中呼叫
`test.py` 展示了如何透過 `QueryService` 直接取得答案：
```python
from app.services.query_service import QueryService
from app.models.query import QueryRequest
import asyncio

async def demo():
    service = QueryService()
    req = QueryRequest(
        question="What is V2G?",
        graph_path="data/output/optimized_entity_graph.json",
        subgraph_distance=2,
    )
    resp = await service.ask_question(req)
    print(resp.model_dump())

if __name__ == "__main__":
    asyncio.run(demo())
```

## 🐳 Docker
- 所有 Docker 相關檔案集中在 `deployment/docker/`
  - `Dockerfile`
  - `docker-compose.yml`（開發環境，會讀取根目錄 `.env`）
  - `docker-compose.prod.yml`（精簡版 production 設定）
- 部署腳本 `deployment/scripts/deploy.sh` 會引用 `docker/docker-compose.prod.yml`
- 使用方式範例：
  ```bash
  # 開發 compose
  docker compose -f deployment/docker/docker-compose.yml up --build

  # 生產腳本
  bash deployment/scripts/deploy.sh deploy
  ```

## 📌 其他注意事項
- `FlowOperations` 會預設讀取 `data/intermediate/alias_dict.json`，可視需求覆寫路徑。
- 問答結果會更新 `data/intermediate/entities_chunks.json`，如需清除快取直接移除該檔案即可。
- `.env.example` 提供常用環境變數範例，交接時可直接複製給新成員。

如果你對流程進行擴充（例如：新增服務、增加資料來源），請記得同步更新此 README。
