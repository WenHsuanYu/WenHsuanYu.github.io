const searchFunc = function (path, search_id, content_id) {
  'use strict';

  // 防呆機制：檢查必要的 DOM 元素是否存在
  const inputElement = document.getElementById(search_id);
  const resultContainer = document.getElementById(content_id);
  if (!inputElement || !resultContainer) {
    console.error('搜尋表單或結果容器未找到，請檢查 search_id 和 content_id 的配置。');
    return;
  }
  // 點擊其他地方時隱藏搜尋結果
  document.addEventListener('click', function (event) {
    const isClickInsideInput = inputElement.contains(event.target);
    const isClickInsideResult = resultContainer.contains(event.target);

    if (!isClickInsideInput && !isClickInsideResult) {
      resultContainer.style.display = 'none';
    }
  });
  
  const closeButtonHTML = "<i id='local-search-close'>×</i>";
  resultContainer.innerHTML = closeButtonHTML + "<ul><span class='local-search-empty'>搜尋中....<span></ul>";

  // 加載 XML 文件並處理數據
  const xhr = new XMLHttpRequest();
  xhr.open('GET', path, true);
  xhr.responseType = 'document';
  xhr.overrideMimeType('text/xml');
  xhr.onload = function () {
    if (xhr.status === 200) {
      const xmlResponse = xhr.responseXML;

      // 防呆機制：檢查 XML 文件是否正確加載
      if (!xmlResponse) {
        console.error('索引文件加載失敗，請檢查文件路徑。');
        resultContainer.innerHTML = closeButtonHTML + "<ul><span class='local-search-empty'>索引文件加載失敗，請稍後再試。<span></ul>";
        return;
      }

      // 解析 XML 文件
      const entries = xmlResponse.getElementsByTagName('entry');
      const datas = Array.from(entries).map(function (entry) {
        return {
          title: entry.getElementsByTagName('title')[0]?.textContent || "Untitled",
          content: entry.getElementsByTagName('content')[0]?.textContent || "",
          url: entry.getElementsByTagName('url')[0]?.textContent || "#"
        };
      });

      // 防呆機制：檢查是否有有效的數據
      if (!datas.length) {
        console.warn('索引文件中沒有有效的數據。');
        resultContainer.innerHTML = closeButtonHTML + "<ul><span class='local-search-empty'>索引文件中沒有有效的數據。<span></ul>";
        return;
      }

      // 提取內部函數到外部作用域
      function matchData(data, keywords) {
        if (!data.content) return { isMatch: false, firstOccur: -1 };
        let isMatch = true;
        let firstOccur = -1;
        for (let index = 0; index < keywords.length; index++) {
          const keyword = keywords[index];
          const indexTitle = data.title.toLowerCase().indexOf(keyword);
          const indexContent = data.content.toLowerCase().indexOf(keyword);

          if (indexTitle < 0 && indexContent < 0) {
            isMatch = false;
            break;
          }
          if (index === 0) {
            firstOccur = indexContent >= 0 ? indexContent : indexTitle;
          }
        }
        return { isMatch, firstOccur };
      }

      function generateResultHTML(data, firstOccur, keywords) {
        let html = "<li><a href='" + data.url + "' class='search-result-title' target='_blank'>" + data.title + "</a>";
        if (firstOccur >= 0) {
          const start = Math.max(firstOccur - 20, 0);
          const end = Math.min(firstOccur + 80, data.content.length);
          let matchContent = data.content.substring(start, end);

          // 高亮關鍵字
          keywords.forEach(function (keyword) {
            const regExp = new RegExp(keyword, "gi");
            matchContent = matchContent.replace(regExp, "<em class='search-keyword'>" + keyword + "</em>");
          });

          html += "<p class='search-result'>" + matchContent + "...</p>";
        }
        html += "</li>";
        return html;
      }

      function handleInputEvent() {
        const query = inputElement.value.trim().toLowerCase();
        
        // 防呆機制：確保 resultContainer 存在
        if (!resultContainer) {
          console.error('結果容器不存在');
          return;
        }
        
        if (query.length === 0) {
          resultContainer.innerHTML = closeButtonHTML + "<ul><span class='local-search-empty'>請輸入搜尋關鍵字<span></ul>";
          return;
        }

        const keywords = query.split(/[\s-]+/).filter(keyword => keyword.length > 0);
        
        // 防呆機制：檢查關鍵字是否有效
        if (keywords.length === 0) {
          resultContainer.innerHTML = closeButtonHTML + "<ul><span class='local-search-empty'>請輸入有效的搜尋關鍵字<span></ul>";
          return;
        }
        
        let resultHTML = '<ul class="search-result-list">';
        let hasResults = false;

        datas.forEach(function (data) {
          const { isMatch, firstOccur } = matchData(data, keywords);
          if (isMatch) {
            hasResults = true;
            resultHTML += generateResultHTML(data, firstOccur, keywords);
          }
        });

        resultHTML += "</ul>";
        
        // 修復：確保關閉按鈕始終存在
        if (hasResults) {
          resultContainer.innerHTML = closeButtonHTML + resultHTML;
        } else {
          resultContainer.innerHTML = closeButtonHTML + "<ul><span class='local-search-empty'>沒有找到內容，請嘗試更換關鍵字。<span></ul>";
        }
      }

      // 添加輸入事件監聽器
      inputElement.addEventListener('input', handleInputEvent);
      
      // 初始化：清空搜尋結果，只顯示關閉按鈕
      resultContainer.innerHTML = closeButtonHTML + "<ul><span class='local-search-empty'>請輸入搜尋關鍵字<span></ul>";
      
    } else {
      console.error('索引文件加載失敗，請檢查文件路徑。');
      resultContainer.innerHTML = closeButtonHTML + "<ul><span class='local-search-empty'>索引文件加載失敗，請稍後再試。<span></ul>";
    }
  };

  xhr.onerror = function () {
    console.error('索引文件加載失敗，請檢查文件路徑。');
    resultContainer.innerHTML = closeButtonHTML + "<ul><span class='local-search-empty'>索引文件加載失敗，請稍後再試。<span></ul>";
  };

  xhr.send();

  // 關閉搜尋結果
  document.addEventListener('click', function (event) {
    if (event.target.id === 'local-search-close') {
      inputElement.value = '';
      resultContainer.innerHTML = closeButtonHTML + "<ul><span class='local-search-empty'>請輸入搜尋關鍵字<span></ul>";
    }
  });
};

const getSearchFile = function () {
  const path = "/search.xml";
  searchFunc(path, 'local-search-input', 'local-search-result');
};