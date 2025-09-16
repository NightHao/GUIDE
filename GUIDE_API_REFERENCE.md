# GUIDE API 參考指南

本文件整理 FastAPI 服務所有可用端點、資料需求與常見操作，建議搭配主專案的 `README.md` 一起閱讀。除 `admin` 類端點外，所有流程都依賴 `data/` 目錄下的輸入檔案。

## 1. 核心流程速覽

```
entities_result.json ──► POST /graph/build ──► entity_graph.json
              │
              └──► POST /graph/process ──► processed_entity_graph.json, log.json
                              │
                              ├──► POST /graph/optimize ──► optimized_entity_graph.json
                              └──► POST /graph/build-aliases ──► alias_dict.json

retrieved_chunks_15.json + optimized_entity_graph.json + alias_dict.json
  └──► POST /query/ask   (或 /query/batch)

cache: data/intermediate/entities_chunks.json 會在問答時自動更新
```

## 2. Graph API

| Endpoint | 說明 | 主要欄位 / 依賴檔案 |
| --- | --- | --- |
| `POST /api/v1/graph/build` | 解析 `entities_result.json`，建出初始圖譜 | `entities_file`、`output_path` (預設寫入 `data/intermediate/`) |
| `POST /api/v1/graph/process` | 合併重複節點、抽取縮寫與規則 | `input_graph`、`output_graph`、`output_log` |
| `POST /api/v1/graph/optimize` | 聚類合併節點描述，產生精簡圖譜 | `input_graph`、`output_graph`、`similarity_threshold` |
| `POST /api/v1/graph/build-aliases` | 建立雙向別名字典 | `log_file`、`output_file` (`alias_dict.json`) |
| `GET /api/v1/graph/status` | 檢查圖檔存在與統計值 | `graph_path` |
| `GET /api/v1/graph/pipeline` | 一次執行 build → process → optimize → build-aliases | `entities_file` |

### 範例：建置圖譜

```bash
curl -X POST http://127.0.0.1:8000/api/v1/graph/build   -H "Content-Type: application/json"   -d '{
        "entities_file": "data/input/entities_result.json",
        "output_path": "data/intermediate/entity_graph.json"
      }'
```

## 3. Query API

| Endpoint | 說明 | 主要欄位 |
| --- | --- | --- |
| `POST /api/v1/query/ask` | 使用 agentic flow 回答單一問題 | `question`、`graph_path`、`subgraph_distance`、`use_agentic_flow`、`alias_overrides` (選填) |
| `POST /api/v1/query/batch` | 批次提問，可選擇將結果存檔 | `questions` (list)、`save_results` 等 |
| `GET /api/v1/query/entities/{entity_name}` | 查詢節點與鄰居資訊 | `include_neighbors`、`max_distance` |
| `GET /api/v1/query/history` | 讀取快取 (`entities_chunks.json`) | `limit` |

### 範例：問答

```bash
curl -X POST http://127.0.0.1:8000/api/v1/query/ask   -H "Content-Type: application/json"   -d '{
        "question": "What is V2G?",
        "graph_path": "data/output/optimized_entity_graph.json",
        "subgraph_distance": 2,
        "use_agentic_flow": true
      }'
```

若縮寫對應到多個候選（例如 `PA`），API 會先回傳 **409** 與以下結構：

```json
{
  "detail": {
    "needs_alias_confirmation": true,
    "ambiguous_aliases": [
      {"alias": "PA", "candidates": ["Platform Adapter", "PLATFORM ADAPTER"]}
    ]
  }
}
```

請重新送出 request 並在 `alias_overrides` 指定要採用的全名：

```bash
curl -X POST http://127.0.0.1:8000/api/v1/query/ask   -H "Content-Type: application/json"   -d '{
        "question": "What is PA?",
        "alias_overrides": {"PA": "Platform Adapter"}
      }'
```

### 範例：批次問答

```bash
curl -X POST http://127.0.0.1:8000/api/v1/query/batch   -H "Content-Type: application/json"   -d '{
        "questions": ["What is V2G?", "What is the difference between EIM and MTC?"],
        "graph_path": "data/output/optimized_entity_graph.json",
        "subgraph_distance": 2,
        "save_results": true
      }'
```

## 4. Admin API

| Endpoint | 說明 |
| --- | --- |
| `GET /api/v1/admin/health` | 檢查關鍵資料夾、API key 是否存在 |
| `GET /api/v1/admin/metrics` | 讀取 `psutil` 的系統指標 |
| `GET /api/v1/admin/config` | 顯示部分設定 (排除敏感資訊) |
| `GET /api/v1/admin/status` | 綜合服務資訊、系統狀態與配置 |
| `POST /api/v1/admin/reload` | 目前僅回傳成功訊息（預留動態 reload） |
| `GET /api/v1/admin/logs` | 回傳範例 log 訊息 |

## 5. 其他常用端點

| Endpoint | 說明 |
| --- | --- |
| `GET /` | 根路徑資訊（瀏覽器顯示的 JSON） |
| `GET /docs` | Swagger UI，可互動測試所有 API |
| `GET /redoc` | ReDoc 文件視圖，適合瀏覽規格 |
| `GET /health` | 對外健康探針 |

## 6. 檔案與工具提示

- `.env` 需提供 `OPENAI_API_KEY`，其餘多餘環境變數會被忽略。
- `test.py`：一次完成 QueryService 驗證 + 暫時啟動 Uvicorn 檢查 `/health`、`/docs`。
- `deployment/docker/`：
  - `Dockerfile`：多階段建置 FastAPI。
  - `docker-compose.yml`：開發環境樣板（讀取根目錄 `.env`）。
  - `docker-compose.prod.yml`：精簡 production 範例。
- `deployment/scripts/deploy.sh`：使用 `deployment/docker/docker-compose.prod.yml` 進行部署。

## 7. 常用資料檔案一覽

| 檔案 | 描述 |
| --- | --- |
| `data/input/entities_result.json` | 建圖輸入 |
| `data/intermediate/entity_graph.json` | `build` 產物 |
| `data/intermediate/processed_entity_graph.json` | `process` 產物 |
| `data/intermediate/log.json` | `process` 的記錄與 pattern |
| `data/intermediate/alias_dict.json` | `build-aliases` 產物，問答時必須存在 |
| `data/intermediate/entities_chunks.json` | 問答快取（程式會自動寫入） |
| `data/output/optimized_entity_graph.json` | 最終問答使用的圖譜 |

---
