// Gemini Sidebar — background script.
// The sidebar itself is a native panel hosted by the container app; this script
// collects page context and forwards requests to it via native messaging.

const MAX_PAGE_CHARS = 30000;
const MAX_SELECTION_CHARS = 10000;

const MENU = {
  toggle: "gs-toggle",
  summarize: "gs-summarize",
  askPage: "gs-ask-page",
  explainSel: "gs-explain-selection",
  askSel: "gs-ask-selection",
};

function sendNative(message) {
  return browser.runtime.sendNativeMessage("application.id", message).catch((err) => {
    console.error("Gemini Sidebar: native message failed", err);
  });
}

const toggleSidebar = () => sendNative({ action: "toggle" });
const askGemini = (prompt, send) => sendNative({ action: "ask", prompt, send });

// Runs inside the page: grabs title, URL, selection and readable text.
function extractPage(maxChars) {
  const pick = () => {
    const candidates = [
      document.querySelector("article"),
      document.querySelector("main"),
      document.querySelector("[role=main]"),
    ].filter(Boolean);
    for (const el of candidates) {
      const t = el.innerText.trim();
      if (t.length > 500) return t;
    }
    return (document.body ? document.body.innerText : "").trim();
  };
  const text = pick().replace(/\n{3,}/g, "\n\n");
  return {
    title: document.title,
    url: location.href,
    selection: String(window.getSelection() || "").trim(),
    text: text.length > maxChars ? text.slice(0, maxChars) + "\n…[truncated]" : text,
  };
}

async function getPage(tab) {
  try {
    const [result] = await browser.scripting.executeScript({
      target: { tabId: tab.id },
      func: extractPage,
      args: [MAX_PAGE_CHARS],
    });
    if (result && result.result) return result.result;
  } catch (err) {
    console.warn("Gemini Sidebar: could not read page content", err);
  }
  return { title: tab.title || "", url: tab.url || "", selection: "", text: "" };
}

function pageBlock(page) {
  const header = `Title: ${page.title}\nURL: ${page.url}`;
  return page.text ? `${header}\n\nPage content:\n"""\n${page.text}\n"""` : header;
}

function clip(text, max) {
  return text.length > max ? text.slice(0, max) + "…" : text;
}

async function summarizePage(tab) {
  const page = await getPage(tab);
  await askGemini(
    `Summarize this web page. Start with a one-sentence TL;DR, then the key points as bullets.\n\n${pageBlock(page)}`,
    true
  );
}

async function askAboutPage(tab) {
  const page = await getPage(tab);
  await askGemini(`Answer my question using this web page as context.\n\n${pageBlock(page)}\n\nMy question: `, false);
}

async function selectionPrompt(tab, selectionText, verb) {
  // Context-menu selectionText can be shortened; prefer the real selection when we can read it.
  const page = await getPage(tab);
  const selection = page.selection || selectionText || "";
  return `${verb}\n\n"""\n${clip(selection, MAX_SELECTION_CHARS)}\n"""\n\n(Selected from "${page.title}" — ${page.url})`;
}

function createMenus() {
  browser.contextMenus.removeAll().then(() => {
    browser.contextMenus.create({ id: MENU.summarize, title: "Summarize Page with Gemini", contexts: ["page"] });
    browser.contextMenus.create({ id: MENU.askPage, title: "Ask Gemini About This Page…", contexts: ["page"] });
    browser.contextMenus.create({ id: MENU.explainSel, title: "Explain “%s” with Gemini", contexts: ["selection"] });
    browser.contextMenus.create({ id: MENU.askSel, title: "Ask Gemini About Selection…", contexts: ["selection"] });
    browser.contextMenus.create({ id: MENU.toggle, title: "Toggle Gemini Sidebar", contexts: ["page", "selection"] });
  });
}

browser.runtime.onInstalled.addListener(createMenus);
browser.runtime.onStartup?.addListener(createMenus);

browser.contextMenus.onClicked.addListener(async (info, tab) => {
  switch (info.menuItemId) {
    case MENU.toggle:
      return toggleSidebar();
    case MENU.summarize:
      return summarizePage(tab);
    case MENU.askPage:
      return askAboutPage(tab);
    case MENU.explainSel:
      return askGemini(await selectionPrompt(tab, info.selectionText, "Explain this in simple terms:"), true);
    case MENU.askSel:
      return askGemini(
        (await selectionPrompt(tab, info.selectionText, "Use this text as context:")) + "\n\nMy question: ",
        false
      );
  }
});

browser.action.onClicked.addListener(() => toggleSidebar());

browser.commands?.onCommand.addListener(async (command) => {
  const [tab] = await browser.tabs.query({ active: true, currentWindow: true });
  if (!tab) return;
  if (command === "summarize-page") return summarizePage(tab);
  if (command === "ask-page") return askAboutPage(tab);
});
