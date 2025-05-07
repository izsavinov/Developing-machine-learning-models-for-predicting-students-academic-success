const CHAT_ORIGIN = 'https://chatgpt.com';

chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (changeInfo.status === 'complete' && tab.url && tab.url.startsWith(CHAT_ORIGIN)) {
    chrome.tabs.sendMessage(tabId, { action: "checkContentScriptInjected" }, response => {
      if (chrome.runtime.lastError) {
        // Content script is not injected, so inject it
        chrome.scripting.executeScript({
          target: { tabId: tabId },
          files: ['content.js']
        });
      }
    });

    // Set side panel options
    if (chrome.sidePanel) {
      chrome.sidePanel.setOptions({
        tabId,
        path: 'sidepanel.html',
        enabled: true
      }).catch(error => console.error('Error setting side panel options:', error));
    }
  } else {
    // Disable the side panel for other pages
    if (chrome.sidePanel) {
      chrome.sidePanel.setOptions({
        tabId,
        enabled: false
      }).catch(error => console.error('Error disabling side panel:', error));
    }
  }
});

// Add this to ensure the side panel is opened when the extension icon is clicked
chrome.action.onClicked.addListener((tab) => {
  if (tab.url && tab.url.startsWith(CHAT_ORIGIN)) {
    chrome.sidePanel.open({tabId: tab.id}).catch(error => console.error('Error opening side panel:', error));
  } else {
    console.log('Not on the ChatGPT page');
  }
});