# ANE 訓練 — 在 Apple Neural Engine 上執行反向傳播

> **繁體中文文件** | [English](README.md)

---

## 目錄

1. [專案簡介](#1-專案簡介)
2. [這個專案能做什麼？訓練 vs 推論](#2-這個專案能做什麼訓練-vs-推論)
3. [系統需求](#3-系統需求)
4. [快速開始（普通開發者友善版）](#4-快速開始普通開發者友善版)
5. [三種訓練管線說明](#5-三種訓練管線說明)
6. [專案架構與檔案說明](#6-專案架構與檔案說明)
7. [Bridge API（從 Python 呼叫 ANE）](#7-bridge-api從-python-呼叫-ane)
8. [效能數據](#8-效能數據)
9. [常見問題（FAQ）](#9-常見問題faq)
10. [注意事項與限制](#10-注意事項與限制)
11. [授權](#11-授權)

---

## 1. 專案簡介

### 什麼是 ANE？

**Apple Neural Engine（ANE）** 是蘋果 Silicon 晶片（M1、M2、M3、M4 系列）內建的 AI 加速器。以 M4 為例，ANE 的理論算力高達 **15.8 TFLOPS**。

然而，蘋果官方只允許透過 CoreML 使用 ANE，且 **只開放推論（inference）功能**，完全不支援訓練（training）。

### 這個專案做了什麼？

本專案透過**逆向工程蘋果的私有 API**（`_ANEClient`、`_ANECompiler`、`_ANEInMemoryModelDescriptor`），繞過官方限制，直接在 ANE 硬體上執行：

- **前向傳播（Forward Pass）**
- **反向傳播（Backward Pass）**
- **完整的 Transformer 訓練迴圈**

這是一個**研究性質的概念驗證（Proof of Concept）專案**，目標是展示「ANE 做訓練在硬體上是可行的」這件事。

> ⚠️ **重要聲明：** 這不是生產級框架，也不是 CoreML / MLX / llama.cpp 的替代品。這是研究用的探索性程式碼。

---

## 2. 這個專案能做什麼？訓練 vs 推論

| 功能 | 支援情況 | 說明 |
|------|----------|------|
| **訓練（Training）** | ✅ 支援 | 本專案的核心目的，在 ANE 上跑反向傳播 |
| **推論（Inference）** | ✅ 也支援 | ANE 推論本來就可以，CoreML 也能做；本專案的推論是用私有 API 直接執行 |
| **多層模型訓練** | ⚠️ 部分 | 已實作 12 層 Transformer（Stories110M）；更大的模型需要額外工程 |
| **自訂資料訓練** | ✅ 支援 | 可用自己的 tokenized 資料集訓練 |
| **生產部署** | ❌ 不建議 | 使用私有 API，Apple 隨時可能改動；利用率約 11%，還未達到實用水準 |

### 關鍵事實

- 訓練可以運作，但 **ANE 利用率約 11%**（距離理論峰值還很遠）
- 許多逐元素運算（element-wise ops）仍然退回 CPU 執行
- 目前不適合替代 GPU 訓練（但它**能跑**，這本身就是突破）
- `dW`（權重梯度）在 CPU 上用 `cblas_sgemm` 計算；`dx`（輸入梯度）在 ANE 上計算

---

## 3. 系統需求

| 項目 | 需求 |
|------|------|
| **作業系統** | macOS 15（Sequoia）或以上 |
| **硬體** | Apple Silicon（M1 / M2 / M3 / M4 系列）|
| **測試環境** | 主要在 M4 上測試；其他晶片理論上相容 |
| **Xcode Command Line Tools** | 必須安裝（提供 `xcrun clang`）|
| **Python**（可選） | Python 3.8+（用於 dashboard 監控與資料下載）|
| **pip 套件**（可選） | `blessed`, `psutil`, `numpy`（dashboard 用）|

### 安裝 Xcode Command Line Tools

```bash
xcode-select --install
```

---

## 4. 快速開始（普通開發者友善版）

這裡以最主流的訓練 **Stories110M 模型**（一個 109M 參數的 Llama2 架構 Transformer）為例，帶你一步步操作。

### 步驟一：確認環境

```bash
# 確認是否有 xcrun clang
xcrun clang --version
# 應輸出類似：Apple clang version 16.x.x ...

# 確認 macOS 版本
sw_vers
# ProductVersion 應 >= 15.0

# 確認是 Apple Silicon
uname -m
# 應輸出：arm64
```

### 步驟二：克隆專案

```bash
git clone https://github.com/sam33339999/ANE.git
cd ANE/training
```

### 步驟三：下載訓練資料

```bash
bash download_data.sh
```

這個指令會從 HuggingFace 下載預先 tokenized 的 TinyStories 資料集（Llama 2 BPE，32K 詞彙表），產生 `tinystories_data00.bin`（約 41 MB，包含約 2000 萬個 token）。

> 📌 如果下載失敗，請確認網路可以連上 HuggingFace（`huggingface.co`）。

### 步驟四：（可選）下載預訓練權重

如果你想從已有權重開始微調（而非從頭訓練），可以下載 `stories110M.bin`。  
請參考 [llama2.c](https://github.com/karpathy/llama2.c) 的說明取得模型檔案。

### 步驟五：編譯

```bash
# 靜態基準版（最穩定，適合初次嘗試）
make train_large

# 或：動態權重版（啟動更快，不需要重新編譯）
cd training_dynamic && make train
```

### 步驟六：執行訓練

```bash
# 從零開始訓練（不需要預訓練權重）
cd training_dynamic
./train --scratch

# 或使用靜態版，從預訓練權重微調
cd ..
./train_large --model stories110M.bin --steps 100 --lr 1e-4
```

### 步驟七：（可選）啟動監控 Dashboard

開另一個終端機視窗：

```bash
pip install blessed psutil numpy
sudo python3 dashboard.py           # 靜態管線
# 或
sudo python3 dashboard.py --dynamic  # 動態管線
```

Dashboard 會即時顯示：
- Loss 曲線
- ANE 功耗
- CPU / 記憶體使用量

---

## 5. 三種訓練管線說明

本專案有三種訓練實作，各有優劣，適合不同情境：

### 管線 1：靜態基準（`train_large`）

```
特性：最穩定、最好理解的版本
執行檔：./train_large
```

- 每次訓練步驟，將權重「烘焙」進 MIL kernel 作為常數
- 每訓練 10 步就重新編譯一次（繞過 ANE ~119 次編譯上限）
- **106.7 ms/step**，每次重啟編譯需要 7.6 秒

**適合：** 第一次嘗試、想了解基本流程的人

```bash
make train_large

# 用法
./train_large stories110M.bin 256 100 1e-4
# 等價寫法
./train_large --model stories110M.bin --steps 100 --lr 1e-4
```

### 管線 2：靜態 + ANE 增強（`train_large_ane`）

```
特性：把更多操作移到 ANE 上執行
執行檔：./train_large_ane
```

- 在管線 1 的基礎上，把分類器前向（32K conv）、softmax、RMSNorm 反向也移到 ANE
- **91.8 ms/step**（比管線 1 快 14%），但編譯時間更長（9.6 秒）
- 提供 `--no-ane-extras` 開關，可退回 CPU 執行（方便 debug）

**適合：** 想要更好每步效能的人

```bash
make train_large_ane

./train_large_ane --model stories110M.bin --steps 100 --lr 1e-4
./train_large_ane --no-ane-extras --steps 100  # 停用 ANE 額外功能
```

### 管線 3：動態權重（`training_dynamic/`）

```
特性：啟動最快、最適合長時間訓練
執行檔：training_dynamic/train
```

- 權重透過 IOSurface 空間維度傳遞，**只在啟動時編譯一次**（9 個 kernel）
- 完全不需要 `exec()` 重啟，沒有編譯上限問題
- **111 ms/step**，但啟動只需 **0.4 秒**（vs 靜態版 7.6 秒）
- 對任何訓練步驟數，**整體花費時間最短**（20 步就快 3.9 倍）

**適合：** 大多數實際訓練場景、長時間訓練

```bash
cd training_dynamic
make train

./train --scratch              # 從隨機初始化開始
./train                        # 從 checkpoint 繼續
./train --steps 200 --lr 1e-4  # 指定步數和學習率
```

### 三種管線效能比較

| | 靜態基準 | ANE 增強版 | 動態版 |
|---|---|---|---|
| **總花費（20步）** | 10.1 秒 | 11.7 秒 | **~2.6 秒** |
| 編譯時間 | 7.6 秒（76%）| 9.6 秒（82%）| 0.4 秒（15%）|
| 訓練時間 | 2.1 秒（21%）| 1.8 秒（16%）| 2.2 秒（85%）|
| **每步時間** | 106.7 ms | **91.8 ms** | 111 ms |
| 每批編譯 kernel 數 | 72 個 | 86 個 | 9 個（只一次）|

> 💡 **建議：** 一般情況下使用動態版（管線 3）。如果你想研究程式碼細節，從靜態基準（管線 1）開始讀起。

---

## 6. 專案架構與檔案說明

```
ANE/
├── README.md                  # 英文說明文件
├── README.zh-TW.md            # 本文件（繁體中文）
├── api_exploration.m          # ANE 私有 API 探索（入門讀物）
├── inmem_basic.m              # 記憶體內 MIL 編譯的概念驗證
├── inmem_bench.m              # ANE dispatch 延遲基準測試
├── inmem_peak.m               # 峰值 TFLOPS 測量（2048x2048 矩陣乘法）
├── sram_bench.m               # ANE SRAM 頻寬探測
├── sram_probe.m               # SRAM 大小/佈局探索
├── bridge/                    # C 語言可呼叫的 ANE 橋接層（供 Python 使用）
│   ├── ane_bridge.h           # Bridge API 標頭檔
│   ├── ane_bridge.m           # Bridge API 實作
│   └── libane_bridge.dylib    # 預編譯動態庫
└── training/                  # 主要訓練程式碼
    ├── Makefile               # 建置腳本
    ├── ane_runtime.h          # ANE 私有 API 包裝（compile、eval、IOSurface）
    ├── ane_mil_gen.h          # MIL 程式生成輔助函式
    ├── model.h                # 模型權重初始化與 blob 建構
    ├── forward.h              # 前向傳播 MIL 生成器
    ├── backward.h             # 反向傳播 MIL 生成器
    ├── stories_config.h       # 模型設定（DIM=768, HIDDEN=2048 等）
    ├── stories_io.h           # IOSurface I/O、NEON fp16 轉換、kernel 編譯/執行
    ├── stories_mil.h          # 靜態管線的 MIL 生成器（6 種 kernel）
    ├── stories_cpu_ops.h      # vDSP 向量化 RMSNorm、交叉熵、Adam
    ├── ane_classifier.h       # ANE 分類器前向（32K conv）與 softmax kernel
    ├── ane_rmsnorm_bwd.h      # ANE RMSNorm 反向 kernel
    ├── train.m                # 最小化訓練迴圈（早期原型）
    ├── tiny_train.m           # 2 層小型模型訓練
    ├── train_large.m          # 主程式：靜態基準管線
    ├── train_large_ane.m      # 主程式：ANE 增強管線
    ├── dashboard.py           # TUI 監控 dashboard
    ├── download_data.sh       # 資料下載腳本
    ├── tokenize.py            # tokenization 腳本
    ├── test_*.m               # 各個 kernel 的單元測試
    └── training_dynamic/      # 動態權重管線
        ├── Makefile
        ├── train.m            # 動態管線主程式
        ├── mil_dynamic.h      # 動態權重 kernel 的 MIL 生成器
        ├── config.h           # 模型設定
        ├── io.h               # IOSurface I/O 與 MIL 編譯輔助
        └── cpu_ops.h          # CPU 操作（SiLU 反向、交叉熵、Adam）
```

### 核心概念快速說明

| 術語 | 中文說明 |
|------|---------|
| **MIL** | Model Intermediate Language，ANE 的中間表示語言，類似一種描述神經網路圖的 DSL |
| **IOSurface** | 蘋果的共享記憶體機制，用來在 CPU 和 ANE 之間傳遞張量（tensor）資料 |
| **fp16** | 半精度浮點數，ANE 的原生格式；CPU 用 fp32，需要轉換 |
| **kernel** | ANE 上的一個計算單元，對應一段 MIL 程式 |
| **BLOBFILE** | ANE 程式中的權重儲存格式（128 位元組 header + fp16 資料）|
| **exec() restart** | 用來繞過 ANE ~119 次編譯上限的技巧：儲存 checkpoint 後重新執行程式 |

---

## 7. Bridge API（從 Python 呼叫 ANE）

`bridge/` 目錄提供一個 C 語言介面，讓你可以用 Python（透過 `ctypes`）呼叫 ANE 功能，不需要寫 Objective-C。

### 使用情境

- 想從 Python 腳本直接在 ANE 上執行計算
- 想把 ANE 整合進現有的 Python ML 工作流程
- 想實驗自定義的 MIL 計算圖

### 主要 API

```c
// 初始化 ANE runtime
int ane_bridge_init(void);

// 編譯 MIL 程式 + 權重為 ANE kernel
ANEKernelHandle *ane_bridge_compile(
    const char *mil_text, size_t mil_len,
    const uint8_t *weight_data, size_t weight_len,
    int n_inputs, const size_t *input_sizes,
    int n_outputs, const size_t *output_sizes
);

// 執行 kernel
bool ane_bridge_eval(ANEKernelHandle *kernel);

// 寫入輸入張量
void ane_bridge_write_input(ANEKernelHandle *kernel, int idx,
                             const void *data, size_t bytes);

// 讀出輸出張量
void ane_bridge_read_output(ANEKernelHandle *kernel, int idx,
                              void *data, size_t bytes);

// 釋放 kernel
void ane_bridge_free(ANEKernelHandle *kernel);
```

### 建置 Bridge 動態庫

```bash
cd bridge
make
# 產生 libane_bridge.dylib
```

---

## 8. 效能數據

### 當前效能（M4，單層 Transformer，dim=768，seq=512）

- **9.3 ms/step**，**11.2% ANE 利用率**（1.78 TFLOPS 持續）
- 每個訓練步驟 6 次 ANE kernel dispatch
- 所有前向和反向 dx passes 在 ANE 上執行
- dW（權重梯度）在 CPU（Accelerate cblas）上計算
- 支援 Adam 優化器、梯度累積、checkpoint/resume

### 優化歷程

| 優化項目 | ms/step | ANE 利用率 |
|----------|---------|-----------|
| 基準線（vDSP transpose） | 33.5 | 3.1% |
| Channel-first 佈局 | 20.3 | 5.2% |
| vDSP 向量化 RMSNorm | 14.2 | 7.4% |
| GCD 非同步 cblas 重疊 | 11.4 | 9.2% |
| ANE RMSNorm 融合 | 11.4 | 9.2% |
| Wo^T 融合（7→6 kernels）| 11.4 | 9.2% |
| 延遲 cblas wait | **9.3** | **11.2%** |

---

## 9. 常見問題（FAQ）

### Q1：我沒有 AI 背景，能用這個專案嗎？

可以！這個專案的訓練腳本已經封裝好了，你只需要：
1. 下載資料（`bash download_data.sh`）
2. 編譯（`make train_large` 或 `cd training_dynamic && make train`）
3. 執行（`./train --scratch`）

你不需要理解反向傳播或 MIL 的細節就能跑起來。

---

### Q2：我可以訓練自己的資料嗎？

可以，但需要先把資料 tokenize 成二進位格式。  
可以參考 `training/tokenize.py` 和 [llama2.c](https://github.com/karpathy/llama2.c) 的資料準備流程。  
資料格式：每個 token 是一個 `uint16_t`（2 位元組）的整數，連續排列成 `.bin` 檔案。

---

### Q3：這個專案會不會在 macOS 更新後壞掉？

**有可能。** 本專案使用蘋果的私有 API，這些 API 沒有任何穩定性保證。每次 macOS 更新都可能改動這些介面。  
這是研究用途，請不要用在生產環境。

---

### Q4：可以在 Intel Mac 上執行嗎？

**不行。** ANE 是 Apple Silicon 專屬硬體，Intel Mac 沒有 ANE。

---

### Q5：可以訓練更大的模型嗎？

目前主要測試過 Stories110M（109M 參數，12 層）。更大的模型理論上可行，但需要：
- 更多記憶體管理工程
- 可能需要修改 pipeline 排程邏輯
- ANE 利用率問題仍需解決

---

### Q6：如何停止訓練並繼續？

每個管線都支援 checkpoint：

```bash
# 正常訓練（會自動儲存 checkpoint）
./train --steps 1000

# 下次繼續（不加 --scratch）
./train
# 或明確指定 checkpoint 檔案
./train_large --ckpt my_checkpoint.bin --resume
```

---

### Q7：ANE 利用率為什麼這麼低？

主要原因：
1. **元素操作回退 CPU**：許多逐元素運算（如部分 activation functions）ANE 不原生支援，被迫在 CPU 上執行
2. **記憶體頻寬瓶頸**：fp32↔fp16 轉換有開銷
3. **Kernel 分散**：每步 6 個 kernel dispatch，切換有開銷
4. **研究限制**：這是概念驗證，還未做完整的硬體層級優化

---

### Q8：Efficiency Report 輸出是什麼意思？

```
=== Efficiency Report ===
Total steps:     20          ← 訓練了多少步
Wall time:       11738 ms    ← 總花費時間
Compile time:    9583 ms     ← 編譯 ANE kernel 花的時間
Train time:      1835 ms     ← 實際訓練花的時間
Avg train:       91.8 ms/step ← 平均每步訓練時間
ANE TFLOPS:      1.15 sustained ← ANE 持續算力
```

---

## 10. 注意事項與限制

### 私有 API 警告

本專案使用蘋果未公開的私有 API：
- `_ANEClient`
- `_ANECompiler`  
- `_ANEInMemoryModelDescriptor`

這些 API 沒有公開文件，可能在任何 macOS 更新後失效。

### 已知技術限制

| 限制 | 說明 |
|------|------|
| **SDPA 因果遮罩** | ANE 硬體會忽略 `attn_mask`；因果注意力需要拆解成 Q@K^T（ANE）→ mask+softmax（ANE）→ scores@V（ANE）|
| **~119 編譯上限** | ANE compiler 有資源洩漏問題，每個程序最多編譯約 119 次；以 `exec()` 重啟 + checkpoint 繞過 |
| **單層訓練** | 主要用單層 Transformer 做效能優化；多層需要額外的 pipeline 排程 |
| **合成資料** | 部分測試使用隨機資料；真實 tokenized 資料支援在持續開發中 |

### 法律聲明

本專案基於以下理由進行逆向工程研究：
- 互通性研究（參考 *Sega v. Accolade*, 1992）
- DMCA §1201(f) 互通性條款
- 不包含任何蘋果的程式碼或二進位檔案
- 本專案與蘋果公司無任何關聯

---

## 11. 授權

MIT License — 請參閱 [LICENSE](LICENSE)

---

## 延伸閱讀

- [Part 1: Reverse Engineering M4 ANE（英文）](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine)
- [Part 2: Benchmarks（英文）](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine-615)
- [llama2.c（模型格式參考）](https://github.com/karpathy/llama2.c)
- [CoreML 官方文件](https://developer.apple.com/documentation/coreml)

---

*本文件由人類 + AI 協作撰寫。*
