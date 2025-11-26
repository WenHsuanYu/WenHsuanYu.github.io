---
title: AI Compiler 概覽
tags:
  - null
  - null
  - AI_Compiler
  - frontend
  - backend
  - Optimization
  - MLIR
  - IR
categories:
  - - AI_Compiler_Notes
  - - Work_In_Progress
date: 2025-11-24 11:36:56
---

## 前言

為什麼要寫這篇文章？第一，想要學習的新主題(重新學習編譯器，過去所學的集中在編譯器前端，而且也忘得差不多); 第二，因為 AI Compiler 相對其他技術來說仍然是一個較為新穎的(子)領域，雖然有一些論文，但是現有的 AI Compiler 所使用到的 MLIR 框架這方面的書籍非常少且品質不佳。因此，這篇文章將會先從兩篇參考資料[1][2]中整理並著重在 AI Compiler 基本架構和列出常見使用到的最佳化技術，並且隨著日後的學習過程，持續更新並充實這篇文章的內容。

閱讀文章前，請優先查看後面備註。

## 為什麼需要 AI Compiler？

就我淺薄的認知，深度學習從研究到落地部署的過程中，框架逐漸百花齊放，從早期的 Theano、Caffe，到後來的 TensorFlow、PyTorch，再到 JAX 等等。這些框架各自有不同的設計理念和最佳化策略，但都面臨著一個共同的挑戰：如何將高階的深度學習模型轉換為底層硬體可高效執行的程式碼，這裡面需要考慮到各種硬體像是不同的 CPU、GPU、TPU、FPGA、各種NPU、手機端的 SoC。 儘管有所謂的主流框架像是 PyTorch，但是硬體碎片化導致了轉換為底層硬體可高效能執行的複雜性，因為由最佳化工程師手動設計的規則來設計執行計算圖的最佳方式的所需工程成本高過可接受程度，而且不同的硬體、模型也都需要進行額外的處理，並沒有辦法有所謂的自適應 (non-adaptive)，再來是行業中無法有足夠多的人力供給，這也就是為什麼需要 AI Compiler 的原因。 

## 深度學習硬體

從五年前預印本 [1] 中可以根據深度學習硬體的通用性，分為以下三大類：

1. 通用硬體：透過軟體-硬體協同設計來支援 DL 工作負載。
- 中央處理器 (CPU) 與圖形處理器 (GPU)：這類硬體（例如 CPU 和 GPU）通常會增加特殊的硬體元件來加速 DL 模型。
- 硬體組件：例如，CPU 和 GPU 分別加入 AVX512 向量指令集和張量核心 (tensor core) 等特殊硬體組件來加速 DL 模型
2. 專用硬體 (Dedicated Hardware)：專用硬體是為 DL 計算而完全客製化的，旨在將性能和能源效率提升到極致。這些硬體通常屬於特定應用積體電路 (ASIC)
- Google TPU (Tensor Processing Unit)：這是最著名的專用 DL 硬體之一。
  - 核心架構：TPU 包含 矩陣乘法單元 (MXU)、統一緩衝區 (UB) 和啟動單元 (AU，　現在為 vector unit 的一部分)。MXU 主要由脈動陣列 (systolic array) 組成，該陣列經過最佳化以提高執行矩陣乘法時的功耗和面積效率。
  - 程式設計：與 CPU 和 GPU 相比，TPU 仍可程式設計，但它使用矩陣作為基本操作元，而不是向量或純量。

3. 神經形態硬體 (Neuromorphic Hardware):神經形態晶片利用電子技術來模擬生物大腦的功能
   核心概念： 這些晶片（例如 TrueNorth）在人工神經元之間具有非常高的連接性，它們複製了類似大腦組織的結構：神經元可以同時儲存和處理數據。與傳統晶片不同，神經形態晶片通常擁有許多微處理器，每個微處理器都有少量的本地記憶體。

但是從五年後的今天來看，神經形態硬體的路線在當前大語言模型持續走紅的趨勢下，似乎沒有太多的激盪，反而是專用硬體 (Dedicated Hardware) 的部分，像是各種 NPU、AI 加速器等，成為除了通用硬體外目前的重要發展方向。

然而，我最近也聽了一個 [Podcast](https://youtu.be/jvqsvbntEFQ?t=4919) 訪問 Apache TVM、XGBoost 的創始人陳天奇其中有一個問題是：「 GPU、TUP、NPU 的 high level 來說，它們區別是什麼？」陳天奇的回答是：「其實它顯然來說越來越模糊了，就是 GPU 原來源自於這個圖形編程，但是從 Cuda 再往後走和圖形編程基本上打不著關係， 它是一個純粹的 single instruction multiple thread，就是你可以用很多線程來編程，一般來說 TPU、NPU 可以看成類似同一個 category，就是它會有一個比較複雜的指令集，使得一條指令可以同時讀一塊內存或者做一個比較大的矩乘，一般來說，在 NPU 不一定會有線程的概念，因為一些指令本身就非常大，但是會發現現在的話，這個 GPU、NPU都會相互走向對方，比如 Nvidia 裡面會引入類似於 Tensor Core， 這個就是原來 NPU 裡面才會有的東西，而且很多時候可能很多 NPU 可能也會引入一些線程的概念，為了更接近 Cuda 的生態，所以就相對來說，這個界線就更模糊了。」

或許，某方面來說嚴格名詞定義來區分什麼是 GPU、TPU、NPU 已經沒有那麼重要了 (並不代表三者應視為同一種硬體），因為硬體的設計規格隨著需求而變，反而是如何讓 AI Compiler (更明確的是說如何利用 MLIR 編譯器基礎設施)能夠更好地支援這些多樣化的硬體，才是未來發展的重點。

## AI Compiler 的通用設計架構

AI Compiler 的通用設計架構採用類似傳統編譯器的分層設計，但其採用多級中間表示（IR）是針對深度學習的神經網路專門最佳化，這種通用架構的主要目的是解決將各種深度學習模型部署到多樣化 DL 硬體上的難題，最終會生成最佳化後的程式碼以提高效率。

AI 編譯器的通用設計主要包含三個核心部分：前端 (Frontend)、多級中間表示 (Multi-level IRs) 和中後端（Middle end, Backend）。

1. 編譯器前端

前端接收來自現有深度學習框架（如 TensorFlow、PyTorch、MXNet 等）的模型定義作為輸入與轉換，它將模型轉換為計算圖表示，即高階 IR(High-level IR)，前端需要實現各種格式轉換，以支援不同框架的格式。

高階 IR/圖 IR (Graph IR): 高階 IR 位於前端，且無關於硬體（Hardware-independent，它代表了計算和控制(if-else, Loop)流程，其設計挑戰在於抽象化計算和控制流程的能力，以捕獲和表示多樣化的 DL 模型。

圖 IR 通常用於建立運算子和資料之間的控制流程和依賴關係，並提供圖級別最佳化 (graph-level optimizations) 的介面和豐富的編譯層面的語意資訊以及客製化運算子的延伸

圖 IR 表達方式：

1. 高階圖表示的融合：SSA-based Graph IR

2020 年 survey paper 所提到關於 DAG-based（如 TF GraphDef）與 Let-binding-based（如 TVM Relay）的路線已結束。現代主流 IR（如 StableHLO、TVM Relax、MLIR TOSA）普遍採用了 SSA (Static Single Assignment) 形式。這種形式融合了兩者的優點：既透過 Def-Use chains 保留了資料流圖的最佳化便利性，又引入 Region 與 Block 機制來解決傳統 DAG 難以表達控制流（Control Flow）與作用域（Scope）的缺陷。

- StableHLO: Google/OpenXLA 推進的業界標準。它不僅繼承了 XLA HLO 的運算語義，更引入了 Region 概念來容納子圖（Subgraph），解決了傳統 DAG 表達力不足的問題，成為連接 PyTorch/JAX/TF 前端與硬體後端的通用協議。

- Torch FX: 這是 PyTorch 2.0 的核心，定位為 Python 層級的符號追蹤 IR (Symbolic Trace IR)。雖然它在 Python 側表現為線性的 Node 序列（類 DAG），但其主要職責是作為「捕獲層 (Capture Layer)」，隨後會迅速 Lowering 到 SSA 形式的後端編譯器（如 Inductor 或 XLA）進行最佳化。


2. 張量計算表示的分層與特化
2020 年的三大流派（函式、Lambda、愛因斯坦記號）在 2025 年並非消失，而是演化為編譯堆疊中不同層級的標準。

- 從「基於函式」到「標準化算子集 (Standardized Op Sets)」

前身： XLA HLO (Function-based)
2025 現狀： StableHLO
演進分析： HLO 沒有消失，而是標準化了。它不再僅被視為一個編譯器的內部 IR，而是演變為一種可攜式協議 (Lingua Franca)。它嚴格定義了張量運算的「語義」（做什麼），而完全剝離了「如何執行」的細節，確保了不同 ML 框架與硬體廠商之間的互通性。

- 從「Lambda 表達式」到「可排程迴圈 IR (Scheduleable Loop IR)」

前身： TVM Tensor Expression (Lambda-based)

2025 現狀： TVM TensorIR (TIR)

演進分析： 

  純粹的 Lambda 表達式（如 te.compute）地位下降，退化為一種前端語法糖。真正的核心下沈到了 TIR。

關鍵區別： 

2020 年的模式是「寫 Lambda 交給黑盒生成代碼」；2025 年的模式轉向「白盒控制」。TIR 允許開發者（或自動搜索算法）直接看到並操作 Loop Nest，進行顯式的 Tiling、Vectorization 和 Tensorization。Lambda 只是生成 TIR 的手段，而非最佳化的載體。

- 從「愛因斯坦記號」到「結構化操作 (Structured Operations)」

前身： Tensor Comprehensions (TC)
2025 現狀： MLIR Linalg Dialect

演進分析： 

這是變化最具顛覆性的部分。TC 專案雖已終止，但其數學精神被 MLIR Linalg 繼承並改造。我們不再稱之為愛因斯坦記號，而稱為結構化操作。

核心差異：

TC (舊時代)： 依賴黑盒搜索 (Polyhedral Auto-tuning)，試圖自動從數學公式猜出最佳解，但往往不可控。
Linalg (新標準)： 採用聲明式轉換。linalg.generic 使用類愛因斯坦記號的映射圖 (Affine Maps) 來描述運算，但其目的是將 Loop 結構顯式暴露給編譯器，讓工程師能精確控制 Tiling 和 Fusion 策略。這是一種「受控的數學表達」，而非「自動化的黑盒」。

前端最佳化手段(硬體無關）可以分成如下：

**Node-level optimizations** (屬於粗粒度的最佳化) 最佳化手段如下：

- **Node elimination** (節點消除)：完全移除一個節點，通常是因為該節點的運算結果是冗餘的或無效的。

  例如：對於 $Sum$ 節點而言，只有輸入一個張量導致無效，所以可以直接傳遞該張量。

- **Node Replacement** (節點替換)：用一個成本更低或已經存在的節點來代替當前的節點。
  
  例如： $\mathbf{A}$ 是一個零維度張量（通常是 $\mathbf{0}$），$\mathbf{B}$ 是一個常數，進行運算 $Sum(\mathbf{A}, \mathbf{B})$，由於 $\mathbf{A}$ 對於加法沒有任何貢獻，所以整個 $Sum$ 運算節點被常數節點 $\mathbf{B}$ 所取代

**Block-level optimizations** 最佳化手段如下：

- **Algebraic simplification** (代數簡化) 是由 (1) algebraic identification （代數識別），(2) strength reduction (強度減弱)，使用成本更低的運算取代代價較高的運算， (3) constant folding （常數折疊），用常數表達式的值替換表達式。

  情況一計算順序最佳化：以做矩陣乘法（GEMM)來說，如果我們有兩個 $\mathbf{A}$ 和 $\mathbf{B}$，但是想要相乘 $(\mathbf{A}^T \mathbf{B}^T)$，更有效率的實作方法是交換 $\mathbf{A}$ 和 $\mathbf{B}$ 的順序，先相乘 $(\mathbf{B} \mathbf{A})$，然後將 $\text{GEMM}$ 的輸出轉置 $(\mathbf{B} \mathbf{A})^T$
  
  情況二節點組合最佳化:將多個連續的 $\text{Transpose}$ 節點合併為一個單一節點，消除恆等 $\text{Transpose}$ 節點，並將實際上沒有移動數據的 $\text{Transpose}$ 節點最佳化為 $\text{Reshape}$ 節點。
  
  情況三 ReduceMean 節點最佳化：如果歸約運算符 (reduce operator)的輸入是 4D，並且最後兩個維度需要歸約 (reduce)，則用 $\text{AvgPool}$ 節點替換 $\text{ReduceMean}$（例如在 $\text{Glow}$ 中）。

> 給定一個 $4\text{D}$ 張量 $T(N, C, H, W)$，如果你沿著 $H$ 和 $W$ 維度進行 $\text{ReduceMean}$，結果是一個 $2\text{D}$ 張量 $T'(N, C)$。對於 $T'(i, j)$ 中的每個元素，它會計算 $T(i, j, :, :)$ 中所有 $H \times W$ 個元素的總和，然後除以 $H \times W$ 個數，相當於使用 $\text{AvgPool}$（平均池化，卷積神經網絡 (CNN) 的操作），它的核心功能是沿著空間維度（即 $H$ 和 $W$）計算非重疊區域的平均值。

這類最佳化考慮一個節點序列，然後利用不同類型節點的交換律、結合律和分配律來簡化計算。除了常見的運算子（+、× 等）之外，代數簡化也可以應用於深度學習特有的運算子（例如，reshape、transpose 和 pooling）

- **Operator fusion** (運算子融合): 它將計算圖中多個邏輯上獨立的深度學習運算子（如卷積、ReLU、Batch Normalization 等）合併成一個更大的、單一的「融合運算子」（fused operator）或核心（kernel）


雖然融合的核心目標未變（最大化算術強度 Arithmetic Intensity，減少 HBM 存取與 Kernel Launch 開銷），但其手段已從 2020 年的「基於規則」演進為 2025 年的「基於結構」。

> 過去在 TVM 中，運算子被分為四類：單射 (injective)、歸約 (reduction)、複雜可輸出融合 (complex-out-fusible) 和不透明 (opaque)。當運算子被定義時，其對應的類別就被確定，針對上述類別，TVM 設計了跨運算子的融合規則。

Operator Fusion (算子融合) 的演進：從規則到結構

1. 傳統方法的局限 (The Graph-Level Era)：
在 2020 年（如 TVM Relay 時代），融合主要依賴預定義的分類標籤（如單射、歸約等）進行貪婪的模式匹配。這種方法難以應對現代 LLM 中複雜的記憶體階層需求，且往往止步於簡單的 Element-wise 操作。
2. 現代融合範式 (The Tile-and-Fuse Era)：
2025 年的主流技術（如 MLIR Linalg, PyTorch Inductor）採用 Tile-and-Fuse 策略。編譯器不再將算子視為黑盒，而是深入分析其 Loop Nest 和 Access Maps。
透過將計算分解為適應硬體 SRAM 大小的 Tiles (塊)，編譯器可以將生產者 (Producer) 的計算結果直接在暫存器或共享記憶體中傳遞給消費者 (Consumer)，實現跨越 MatMul/Conv 等複雜算子的深度融合。
3. 演算法級融合 (Algorithmic Fusion)：
受 FlashAttention 啟發，現代融合超越了編譯器的自動最佳化範疇。為了打破 Memory Wall，業界開始採用「重計算 (Recomputation)」策略——在融合的 Kernel 內部重新計算部分數據以避免高昂的記憶體讀寫。這種高度客製化的融合通常由 OpenAI Triton 等語言顯式構建，而非僅依賴編譯器的自動推導。

**Operator sinking**: 將 transpose 等運算子下沉到 batch normalization, ReLU, sigmoid, and channel shuffle 等運算之後。透過這種最佳化，許多相似的操作被移到彼此更近的位置，從而創造了更多代數簡化的機會


定義與演變：

早期（2020 年前）算子下沉被視為硬體無關的代數簡化。
現今（2025 年），它已被整合進佈局傳播 (Layout Propagation) 流程。編譯器依賴目標硬體特性 (Target-Dependent features)（如 Tensor Cores 對 Memory Layout 的約束）來精確控制算子的位置。

執行位置：它不再只是純前端工作，而是位於 Lowering (降級) 或 Codegen (代碼生成) 準備階段。這是為了確保最佳化決策能直接映射到硬體的指令集優勢上。

核心目的：
賦能融合 (Enabling Fusion)： 透過移動資料變換算子，創造出最大的可融合子圖 (Fusible Subgraph)，減少 HBM 存取。
硬體適配 (Hardware Adaptation)： 確保數據佈局滿足加速器的硬性要求（Legalization），最大化硬體利用率。

**Dataflow-level optimizations** (資料流級別最佳化)

- **Common Sub-expression Elimination, CSE** (公共子表達式消除)
  - 一個表達式 E 若其值在先前已被計算且自上次計算以來未改變，則被視為公共子表達式。
  - 在這種情況下，只需計算 E 的值一次，後續使用已計算的值即可避免重新計算。
  - DL編譯器會搜尋整個計算圖以尋找公共子表達式，並將後續的重複子表達式替換為先前計算的結果

- **Dead Code Elimination, DCE** (死碼消除)
  - 如果某段程式碼的計算結果或副作用未被使用，則該程式碼被稱為死碼。
  - DCE 最佳化會移除這些死碼。
  - 死碼通常不是由程式設計師造成，而是由其他圖形最佳化（例如 CSE）所產生。因此，DCE 和 CSE 通常在其他圖形最佳化之後應用。
  - 死儲存消除 (Dead Store Elimination, DSE) 也屬於 DCE，它會移除對從未使用的張量進行的儲存操作

- **Static Memory Elimination** (靜態記憶體規劃)
  - 此最佳化的目的是盡可能重複使用記憶體緩衝區。由於靜態記憶體規劃是在離線（offline）完成的，因此可以應用更複雜的規劃演算法。
  - 通常有兩種方法：
    - 原地記憶體共享 (In-place memory sharing)： 允許操作的輸入和輸出使用相同的記憶體，計算前只分配一個記憶體副本。
    - 標準記憶體共享 (Standard memory sharing)： 重複使用先前操作的記憶體，但沒有重疊。

現在使用緩衝區分配 (Buffer Assignment)：(取代舊的 Static Memory Planning)
在編譯後期，根據目標設備的記憶體階層（SRAM/HBM），執行活躍度分析 (Liveness Analysis) 來最小化峰值記憶體佔用。

- **Layout Transfomation** (佈局轉換)
  - 此最佳化旨在找到儲存張量在計算圖中的最佳資料佈局（data layouts），然後在圖中插入佈局轉換節點。
  - 實際的轉換並非在此階段執行，而是在編譯器後端評估計算圖時執行。
  - 佈局的重要性： 相同操作在不同資料佈局下的性能會有所不同，且最佳佈局也因硬體而異。例如，在 GPU 上，NCHW 格式的操作通常運行得更快。
  - 性能考量： 雖然資料佈局對最終性能有顯著影響，但轉換操作本身會產生可觀的開銷，因為它們也消耗記憶體和計算資源

> 現代 GPU (NVIDIA Tensor Cores) 極度偏好 NHWC (Channels-last) 或者是分塊格式 (Blocked Layouts, 如 NCHW32c)。
> NCHW 是舊式 CUDA Core (Pascal 架構以前) 的偏好。如果你在 2025 年的 H100 GPU 上跑 NCHW 的卷積，Tensor Core 需要隱式轉置，效能會非常差。

> 佈局傳播 (Layout Propagation)：(取代舊的 Layout Transformation)
> 不再是前端猜測，而是由後端根據硬體（如 Tensor Core 需要 NHWC）反向傳播約束，自動插入或消除 Layout 轉換節點。

2. 編譯器後端

編譯器後端的主要功能是：

- 硬體特定的最佳化 (Hardware-specific optimizations)：後端負責處理與目標硬體相關的最佳化和轉換。
- 程式碼生成與編譯 (Code generation and compilation)：最終目的是生成並編譯出針對不同 DL 硬體（如 GPU、TPU、CPU 等）的高度最佳化程式碼。

後端的操作是基於**低層次 IR (Low-level IR)** 進行的，low-level IR 是專為硬體特定最佳化和程式碼生成而設計的且必須足夠細粒度，以反映硬體的特性並表示硬體特定的最佳化，另外也允許在編譯器後端使用成熟的第三方工具鏈:

  - 基於 Halide 的 IR：Halide 的核心理念是計算和調度分離。TVM 採用的 Halide IR 被改進為獨立的符號 IR，追求更好的組織結構和可重用性

    2020 觀點： TVM 使用改進的 Halide IR，強調計算與排程分離。

    2025 現狀： 演化為 TensorIR (TIR)。

    - TVM 已經從舊的 Halide 風格演進為 TensorIR (TIR)。TIR 不僅保留了符號化優勢，還引入了 Block 語義，使其能更好地映射到現代硬體的 Tensor Core 指令。它不再僅是「分離」，而是允許透過 Schedule 原語對 Loop Nest 進行更直接的交互式變換。


  
  - 基於多面體模型的 IR (Polyhedral-based IR)：使用線性規劃和仿射變換等數學方法來最佳化具有靜態控制流的循環代碼。這種模型的靈活性使其廣泛應用於通用編譯器。TC 和 PlaidML 採用此模型作為其低層次 IR

    2020 觀點： TC 和 PlaidML 採用此模型，使用線性規劃進行優化。

    2025 現狀： 純多面體編譯器已死，但精神在 MLIR Linalg 中永存。

    - 現狀： Facebook 的 Tensor Comprehensions (TC) 專案已終止，PlaidML 也已式微。業界發現，完整的多面體模型（如使用整數線性規劃求解器）在編譯時間上太過昂貴，且難以處理非靜態形狀 (Dynamic Shapes)。
    - 繼承者： 多面體的核心概念（仿射映射、依賴分析）被 MLIR Linalg 和 Affine Dialect 吸收。Linalg 採用了「結構化操作」而非全自動的多面體求解，這是一種實用主義的妥協——保留了數學上的優雅，但放棄了不可控的黑盒搜索。

  - 其他獨特 IR：例如 Glow 的低層次 IR 是基於指令的表達式，用於操作張量，並使用限定符（如 @in、@out）幫助確定記憶體最佳化時機。MLIR 是一個元編譯器基礎設施，它引入方言 (dialects) 來表示多個抽象級別，並支援方言之間的靈活轉換。XLA 的 HLO IR 也足夠細粒度，可視為高層次和低層次 IR
    
    2020 觀點： Glow 是基於指令的；MLIR 是基礎設施；XLA HLO 是細粒度的。

    2025 現狀： MLIR 一統天下，Glow 邊緣化，HLO 標準化。
    
    - MLIR: 它不再只是「另一個 IR」，而是構建編譯器的作業系統。幾乎所有新一代編譯器（IREE, Modular MAX, Triton 的內部 IR）都建立在 MLIR 之上。
    - XLA HLO: 演化為 StableHLO，成為跨框架的標準交換格式。
    - Glow: 在 PyTorch 轉向 TorchInductor 和邊緣與行動端轉向 ExecuTorch 後，Glow 的重要性已大幅降低，已於 2025.07.01 公開封存。

> 技術路線的轉變：[Next Steps for PyTorch Compilers](https://dev-discuss.pytorch.org/t/next-steps-for-pytorch-compilers/309)

> Intel/AMD/Qualcomm 的轉向： 在 2020 年左右，這些廠商還會為 Glow 貢獻後端代碼。但到了 2025 年回頭看，Intel 已轉向 OpenVINO/OneAPI，AMD 轉向 Torch-MLIR/IREE，Qualcomm 轉向 QNN/ExecuTorch Delegate。



後端最佳化技術涵蓋了幾個關鍵領域，以確保在目標硬體上實現高效能：

A. 硬體特定最佳化 (Hardware-specific Optimizations)

這些最佳化旨在針對特定硬體（如 GPU、ASIC）生成高效能程式碼。
- 硬體固有映射 (Intrinsic Mapping， 也就是翻譯成由硬體支援的內建指令)：將指令轉換為硬體上已高度最佳化的核心（kernels）。
- 記憶體分配與擷取 (Memory Allocation & Fetching)。
- 記憶體延遲隱藏 (Memory Latency Hiding)。
- 迴圈導向最佳化 (Loop Oriented Optimization Techniques)：包括迴圈融合（Loop fusion）、分塊（Tiling）、重排序（Reordering）和展開（Unrolling）等。
- 平行化 (Parallelization)：利用平行性來最大化硬體利用率。


新增趨勢： 從 Loop 優化轉向 Block/Tile 優化。

2020 年主要談論 Loop Fusion/Reordering。

2025 年，隨著 OpenAI Triton 的普及，後端優化的思維轉向了 Block-based Programming。編譯器（如 Inductor）生成的代碼不再是複雜的 CUDA C++ Loop 模板，而是直接生成管理 Tile 移動和計算的 Triton 程式碼。

Intrinsic Mapping: 現在更強調 MMA (Matrix Multiply-Accumulate) 指令的自動生成，以及對稀疏計算 (Sparse Tensor Core) 的支援。


B. 自動調優 (Auto-tuning)

由於硬體特定最佳化中的參數調整空間龐大，自動調優是必要的步驟，用於確定最佳的參數配置。

- 自動調優的組成部分包括參數化（Parameterization）、成本模型（Cost model）和搜索技術（Parameter searching)。

2020 觀點： 

參數搜索是必要的（如 AutoTVM）。

2025 現狀： 

從「黑盒搜索」轉向「解析模型」與「啟發式生成」, 雖然搜索（如 TVM MetaSchedule）仍然存在，但因為編譯時間過長，現代編譯器更傾向於使用 Deterministic Heuristics (確定性啟發法) 或 Analytical Cost Models (解析成本模型) 來快速生成「足夠好」的 Kernel，或者像 Triton 一樣讓開發者手寫關鍵參數（Block Size），僅對少量參數進行 Auto-tune。

C. 最佳化核心函式庫 (Optimized Kernel Libraries)

後端最佳化還包括利用最佳化後的核心函式庫，例如 $\text{NVIDIA}$ cuDNN, $\text{Intel}$ oneAPI Deep Neural Network Library, $\text{AMD}$ MIOpen

2020 觀點： 

編譯器利用 cuDNN/MIOpen。

2025 現狀： 

編譯器與函式庫的混合 (Hybrid) 模式，編譯器並沒有完全取代函式庫。目前的最佳實踐（如 PyTorch 2.0）是：MatMul 和 Attention 使用高度優化的函式庫 (cuDNN, FlashInfer, Cutlass)，而 Pointwise/Reduction/Normalization 操作則由編譯器動態生成並融合 (Fusion)。編譯器負責生成「膠水代碼」將這些運算黏在一起。

D. 佈局轉換執行 (Layout Transformation Execution)

在編譯器前端（Frontend）插入的佈局轉換節點，其實際轉換操作是在編譯器後端評估計算圖時執行的。

2025 修正： 已演變為 Layout Propagation (佈局傳播)。這不再只是單純的「執行轉換」，而是一個複雜的 Pass，負責在全圖範圍內消除不必要的 Copy，並為 Tensor Core 選擇最佳的 Padding 和 Blocking 策略。

- 程式碼生成目標

後端將最佳化後的低層次 IR 轉換為目標硬體的程式碼，目標平台包括：

- CPU (例如：X86, ARM, RISC-V)。
- GPU (例如：NVIDIA, AMD)。
- ASIC (例如：TPU, Inferentia, NNP, 等)。
- DSP (數位信號處理器)。
- 程式碼生成方案還可能包括 LLVM、CUDA、OpenCL、OpenGL

2025年現狀：

- GPU: 除了 CUDA，Triton 已成為事實上的中間層。許多後端現在選擇生成 Triton IR，再由 Triton 編譯器轉為 PTX，而非直接生成 CUDA C++。
- CPU: RISC-V (RVV) 成為新興的重要目標，LLVM 對其支援已大幅成熟。
- Web/Edge: WebGPU (WGSL) 和 Vulkan (SPIR-V) 成為邊緣運算和瀏覽器端推論的主流目標，IREE 等編譯器對此有極佳的支援。


**2025 年視角下的缺失拼圖 (What's Missing from 2020?)**

如果以 2025 年的標準來看，這份 2020 年的後端整理缺少了以下關鍵概念：

1. 動態形狀 (Dynamic Shapes) 的一等公民支援： 2020 年的後端大多假設靜態 Input Shape。2025 年的後端（尤其是基於 MLIR 的）必須處理 Transformer 模型中變動的 Sequence Length，這對記憶體分配和 Loop 生成提出了全新挑戰。
2. AI 輔助編譯 (AI for Compiler)： 使用機器學習模型來預測最佳的 Tiling size 或 Fusion 策略，已經從學術研究進入了部分生產環境。
3. 稀疏性編譯 (Sparse Compilation)： 隨著 Mixture-of-Experts (MoE) 模型的流行，後端必須能處理稀疏張量，MLIR 的 SparseTensor Dialect 便是為此而生。

在 2025 年，雖然「後端做硬體優化」的大方向沒變，但執行手段變了：

- IR: 從各自為政變成了 MLIR 統治。
- Codegen: 從生成 C++/CUDA 變成了生成 Triton 或 LLVM Dialect。
- Polyhedral: 從全自動黑盒變成了 Linalg 結構化白盒。

## 備註

閱讀完參考資料\[1][2]並著手整理文章內容的同時，我好奇 2020 年的 survey paper 所提的技術路線或發展預測也好，在 2025 年還有多少準確性與適用性，於是我下 prompt 與 AI 問答 (兩到三個平台的 AI) 驗證今昔比較，所以文章內容有 AI 所做的回顧，但是我現在無法驗證它是不是 100% 正確和可靠，但是相信在之後學習的路上，把這些回顧當成待驗證的問題也是一件好事，所以暫時擱置在整理內容的下方， 並且將文章分類為 WIP。


## 參考資料


1. [The Deep Learning Compiler: A Comprehensive Survey](https://arxiv.org/abs/2002.03794)
2. [A friendly introduction to machine learning compilers and optimizers](https://huyenchip.com/2021/09/07/a-friendly-introduction-to-machine-learning-compilers-and-optimizers.html#hand_designed_vs_ml_based_compilers)








