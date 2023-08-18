document.addEventListener('DOMContentLoaded', function () {
    const analyzeButton = document.getElementById('analyze-button');
    const textArea = document.getElementById('text');
    const resultDiv = document.getElementById('result');

    analyzeButton.addEventListener('click', function () {
        const text = textArea.value.trim();
        if (text !== '') {
            chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
                const activeTab = tabs[0];
                chrome.scripting.executeScript({
                    target: { tabId: activeTab.id },
                    function: analyzeText,
                    args: [text],
                });
            });
        }
    });

    function analyzeText(text) {
        chrome.runtime.sendMessage({ action: "analyzeText", text });
    }
    
    chrome.runtime.onMessage.addListener(function (request, sender, sendResponse) {
        if (request.action === "displayResult") {
            const result = request.result;
            resultDiv.innerText = result;
        }
    });
});
