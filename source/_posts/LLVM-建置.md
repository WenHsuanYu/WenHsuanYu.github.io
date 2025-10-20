---
title: LLVM 建置
tag:
  - LLVM
  - 建置
  - CMake
  - Ninja
category:
  - LLVM_Notes
date: 2025-10-20 00:00:00
---


# LLVM 建構與配置

本文將介紹如何從原始碼建構 LLVM 並進行基本配置。LLVM 是一個模組化且可重用的編譯器基礎設施，廣泛應用於各種編譯器和工具鏈中。

## 編譯 LLVM 所需的軟體包

[LLVM 官方文檔](https://llvm.org/docs/GettingStarted.html#software) 建議安裝以下軟體包：

- CMake (版本 3.20 或更高)
- python (版本 3.8 或更高)
- zlib (版本 1.2.3.4 或更高)
- GNU Make (版本 3.79, 3.79.1)，在這邊我們使用 Ninja 作為替代
- PyYAML (版本 5.1 或更高)

部份軟體包是可選的。

## 建置 LLVM

1. 從 Github 上 clone LLVM 原始碼：

   ```bash
   git clone https://github.com/llvm/llvm-project.git
   ```

2. 安裝 Ninja :

    ```bash
    sudo apt install ninja-build
    ```
3. 新增建置資料夾並進入：

   ```bash
   mkdir llvm-build
   cd llvm-build
   ```
4. 使用 CMake 產生 LLVM 建構藍圖：

   ```bash
   cmake -G Ninja -DLLVM_TARGETS_TO_BUILD="X86" \
                -DLLVM_USE_LINKER=lld \
                -DCMAKE_BUILD_TYPE=RelWithDebInfo \
                -DBUILD_SHARED_LIBS=ON \
                ../llvm-project/llvm
   ```
使用 Ninja 產生 assembly-like 建置腳本，然後這邊我們對於 cmake 使用一些客製化選項：

- `-DLLVM_USE_LINKER=lld`：指定使用 lld 作為連結器。
  - `lld` 是 LLVM 的連結器，提供更快的連結速度。
- `-DCMAKE_BUILD_TYPE=RelWithDebInfo`：設定建置類型
  - RelWithDebInfo： `-O2` 最佳化層級，但同時包含除錯資訊，但是無斷言。
  - Debug：無最佳化（`-O0`），包含完整除錯資訊和斷言，會佔用大量硬碟空間。
  - Release：最高最佳化 (`-O3`)，無除錯和斷言資訊。
  - MinSizeRel：針對尺寸進行最佳化。
- `-DLLVM_TARGETS_TO_BUILD="X86"`：指定要建置的目標架構
  - 這邊我們只建置 x86 架構，可以根據需求添加其他架構 (以分號區隔），如 ARM、AArch64、MIPS 等。
- `-DBUILD_SHARED_LIBS=ON`：設定建置為動態共享庫
  - 連結靜態庫通常比動態庫花費更多時間，多個可執行檔連結到同一組靜態庫時，可執行檔的總大小會比動態庫大得多，這樣可以減少最終二進位檔的大小。
  - 使用 Debugger 除錯 LLVM 時， 通常一開始會花費大量時間載入靜態連結的可執行檔，降低除錯效率。

5. 開始建置 LLVM：
   
   ```bash
   ninja all
   ```

    ninja 會根據其平行程度做出決定，以最大平行化來加速建置過程。你也可以使用 `-j` 參數來手動設定平行度，例如：

   ```bash
   ninja -j8 all
   ```

6. (可選）建構最佳化版的 llvm-tblgen

    加入最佳化版本的參數

    ```bash
    cmake -DLLVM_OPTIMIZED_TABLEGEN=ON ..
    ```

    `TableGen` 是 `Domain-Specific Language` (`DSL`) 的一種，用於描述結構化資料，並作為 LLVM 建構過程的一部分產生 C/C++ 程式碼。轉換工具為 `llvm-tblgen`。


 
