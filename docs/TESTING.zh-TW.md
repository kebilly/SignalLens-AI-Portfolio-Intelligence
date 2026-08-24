# SignalLens 測試工程說明

SignalLens 使用分層測試策略，將可重現的領域邏輯、外部服務邊界及 Streamlit 使用流程分開驗證。測試不會呼叫真實付費 API，也不需要讀取正式 API Key。

## 測試層級

| 層級 | 工具 | 驗證重點 |
|---|---|---|
| 單元測試 | pytest、unittest | 風險公式、問卷計分、情緒解析、ETF 配對及輸入驗證 |
| 服務邊界測試 | 模擬 HTTP Session | OpenAI、Perplexity、FMP、Alpha Vantage 的正常與錯誤回應 |
| 整合測試 | pytest | 風險與情緒量尺、固定報告、PDF 及跨模組資料流 |
| UI 冒煙與流程測試 | Streamlit AppTest | 首頁載入、功能導覽、四階段問卷、結果產生及套用 |
| 靜態分析 | Ruff、mypy | 程式風格、常見錯誤及型別不一致 |
| 持續整合 | GitHub Actions | 每次 push 與 Pull Request 自動執行全部品質門檻 |

## 自動化測試範圍

- 投資人風險問卷的題數、維度、極端答案、分類邊界及矛盾選項。
- 投資組合風險公式、現金部位及 Beta 邏輯。
- HHI、有效持倉數、VaR、CVaR 與風險貢獻。
- 新聞情緒邊界、時間解析、缺失欄位及空資料。
- ETF 的 ISIN 對應、代碼差異及重疊權重。
- PDF 是否可產生、讀取並包含必要章節。
- Prompt 中不可信新聞資料的隔離。
- AI 回應達到輸出上限時的自動續寫與合併。
- 401／403、429、500、逾時、無效 JSON、資料不足及畸形供應商資料。
- Streamlit 首頁及完整四階段問卷的 UI 流程。

## 外部 API 測試原則

測試以 Fake Session 和 Fake Response 模擬供應商，不會傳送網路請求。這能避免：

- 消耗 API 額度。
- 在 CI 中保存正式憑證。
- 因網路或供應商狀態造成不穩定測試。
- 錯誤訊息意外洩漏 API Key。

測試也會確認公開錯誤訊息不包含模擬的秘密值。

## 覆蓋率門檻

CI 對服務與領域邏輯設定最低 70% 覆蓋率。Streamlit UI 檔案不納入數字門檻，因為宣告式版面行數會降低指標的解釋力；UI 改由 Streamlit AppTest 的首頁與完整問卷流程保護。

目前本機基準為 51 項測試全部通過，核心邏輯行覆蓋率為 80.32%。數量與實際覆蓋率會隨功能調整，以 CI 結果為準。

## 本機執行

先安裝開發依賴：

```powershell
py -m pip install -r requirements-dev.txt
```

執行與 CI 相同的檢查：

```powershell
py -m compileall -q app.py portfolio_app tests
py -m ruff check .
py -m mypy portfolio_app
py -m pytest --cov=portfolio_app --cov-report=term-missing --cov-fail-under=70
```

## GitHub Actions 品質門檻

工作流程位於 `.github/workflows/ci.yml`，會在每次 push 及 Pull Request 執行：

1. Python 編譯檢查。
2. Ruff lint。
3. mypy 型別檢查。
4. 單元、整合及 Streamlit UI 測試。
5. 核心邏輯覆蓋率門檻。
6. 上傳 JUnit XML 與 coverage XML 作為工作流程產物。

任何必要檢查失敗時，CI 工作即失敗，讓問題在合併前可見。

