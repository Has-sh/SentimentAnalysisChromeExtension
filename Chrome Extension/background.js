chrome.runtime.onMessage.addListener(function (request, sender, sendResponse) {
    if (request.action === "analyzeText") {
        const text = request.text;
        fetch('http://localhost:8000/analyze_sentiment/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: `text=${encodeURIComponent(text)}`,
        })
        .then(response => response.json())
        .then(data => {
            const result = data.result;
            chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
                chrome.scripting.executeScript({
                    target: { tabId: tabs[0].id },
                    function: displayResult,
                    args: [result],
                });
            });
        });
    }
});

function displayResult(result) {
    chrome.runtime.sendMessage({ action: "displayResult", result });
}
