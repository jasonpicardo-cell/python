# Gemini Sidebar for Safari

A Chrome-style Gemini side panel for Safari. Safari has no side-panel API, and gemini.google.com
can't be iframed into pages (it blocks framing and Safari blocks its third-party cookies), so the sidebar is a
native panel (WKWebView) hosted by the container app. The Safari Web Extension drives it via native messaging.

## Features
- ✦ toolbar button or **⌃⌥G** (global) toggles a full-height sidebar docked to the screen edge
- Safari's window is resized to sit beside the sidebar and restored when it closes (toggle in the menu bar ✦)
- Right-click: **Summarize Page with Gemini**, **Ask Gemini About This Page…**, **Explain “selection”**, **Ask Gemini About Selection…**
- Shortcuts in Safari: **⌃⇧S** summarize page, **⌃⇧A** ask about page
- Google sign-in persists; uploads, downloads (→ ~/Downloads), mic input, zoom (⌘+/⌘−/⌘0), ⌘N new chat, ⌘R reload
- Menu-bar ✦: dock left/right, keep on top, make room in Safari, sign out

## Build & install
```sh
cd "Gemini Sidebar"
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcodebuild -project "Gemini Sidebar.xcodeproj" \
  -scheme "Gemini Sidebar" -configuration Release -derivedDataPath build ARCHS=arm64 CODE_SIGN_IDENTITY=- DEVELOPMENT_TEAM= build
cp -R "build/Build/Products/Release/Gemini Sidebar.app" /Applications/
open "/Applications/Gemini Sidebar.app"
```
Then in Safari: Settings ▸ Advanced ▸ *Show features for web developers*, Develop ▸ *Allow Unsigned Extensions*,
and Settings ▸ Extensions ▸ enable **Gemini Sidebar** (allow it on all websites so it can read page text).
"Allow Unsigned Extensions" resets when Safari quits; sign with your Apple Developer team in Xcode to avoid that.

## Layout
- `Gemini Sidebar/` – macOS app: `SidebarController` (panel + WKWebView + prompt injection), `SafariWindowFitter`
  (AppleScript resize), `HotKey`, `OnboardingWindowController`
- `Gemini Sidebar Extension/` – `SafariWebExtensionHandler` forwards messages to the app via `geminisidebar://`;
  `Resources/` holds the web extension (manifest, background.js, icons)

Prompt injection targets Gemini's current DOM (`rich-textarea .ql-editor`, `button.send-button`). If Google changes it,
the prompt is copied to the clipboard instead; update the selectors in `SidebarController.injectScript`.
