import Cocoa
import WebKit

/// Panel that can take keyboard focus (needed to type into Gemini).
final class SidebarPanel: NSPanel {
    override var canBecomeKey: Bool { true }
    override var canBecomeMain: Bool { true }
}

/// Hosts gemini.google.com in a panel docked to the edge of the screen, like Chrome's Gemini side panel.
final class SidebarController: NSObject {
    enum Side: String { case right, left }

    static let home = URL(string: "https://gemini.google.com/app")!
    private static let headerHeight: CGFloat = 40
    private static let minWidth: CGFloat = 340

    private var panel: SidebarPanel!
    private var webView: WKWebView!
    private var toast: NSTextField!
    private var popups: [NSWindow] = []
    private var pendingPrompts: [(text: String, send: Bool)] = []
    private let safariFitter = SafariWindowFitter()
    private let defaults = UserDefaults.standard

    var isVisible: Bool { panel.isVisible }

    var side: Side {
        get { Side(rawValue: defaults.string(forKey: "side") ?? "") ?? .right }
        set {
            defaults.set(newValue.rawValue, forKey: "side")
            if isVisible { safariFitter.restore(); show() }
        }
    }

    var keepOnTop: Bool {
        get { defaults.object(forKey: "keepOnTop") as? Bool ?? true }
        set { defaults.set(newValue, forKey: "keepOnTop"); applyLevel() }
    }

    var makeRoomInSafari: Bool {
        get { defaults.object(forKey: "makeRoomInSafari") as? Bool ?? true }
        set {
            defaults.set(newValue, forKey: "makeRoomInSafari")
            if !newValue { safariFitter.restore() } else if isVisible { safariFitter.makeRoom(for: panel.frame, side: side) }
        }
    }

    private var width: CGFloat {
        get { max(Self.minWidth, defaults.object(forKey: "width") as? CGFloat ?? 440) }
        set { defaults.set(newValue, forKey: "width") }
    }

    override init() {
        super.init()
        buildPanel()
        webView.load(URLRequest(url: Self.home)) // preload so the first show is instant
    }

    // MARK: Showing and hiding

    func toggle() { isVisible ? hide() : show() }

    func show() {
        dock()
        NSApp.activate()
        panel.makeKeyAndOrderFront(nil)
        panel.makeFirstResponder(webView)
        focusPromptBox()
        // After the panel is up: the first run shows a blocking Automation permission prompt.
        if makeRoomInSafari {
            DispatchQueue.main.async { [self] in safariFitter.makeRoom(for: panel.frame, side: side) }
        }
    }

    func hide() {
        panel.orderOut(nil)
        safariFitter.restore()
        // Hand focus back to whatever app (normally Safari) was in front before.
        NSApp.hide(nil)
    }

    func restoreSafariWindow() { safariFitter.restore() }

    private func dock() {
        let mouse = NSEvent.mouseLocation
        guard let screen = NSScreen.screens.first(where: { NSMouseInRect(mouse, $0.frame, false) }) ?? NSScreen.main else { return }
        let area = screen.visibleFrame
        let w = min(width, area.width * 0.7)
        let x = side == .right ? area.maxX - w : area.minX
        panel.setFrame(NSRect(x: x, y: area.minY, width: w, height: area.height), display: true)
    }

    private func applyLevel() {
        panel.level = keepOnTop ? .floating : .normal
    }

    // MARK: Gemini actions

    func newChat() {
        show()
        webView.load(URLRequest(url: Self.home))
    }

    func reload() { webView.reload() }

    func zoom(by delta: CGFloat?) {
        webView.pageZoom = delta.map { min(2, max(0.5, webView.pageZoom + $0)) } ?? 1
    }

    func clearWebsiteData() {
        let store = webView.configuration.websiteDataStore
        store.removeData(ofTypes: WKWebsiteDataStore.allWebsiteDataTypes(), modifiedSince: .distantPast) { [weak self] in
            self?.webView.load(URLRequest(url: Self.home))
        }
    }

    /// Puts `text` into Gemini's prompt box and optionally submits it.
    func ask(_ text: String, send: Bool) {
        show()
        guard !text.isEmpty else { return }
        if webView.url?.host == "gemini.google.com" && !webView.isLoading {
            inject(text, send: send)
        } else {
            pendingPrompts.append((text, send)) // delivered once Gemini has loaded (e.g. after sign-in)
        }
    }

    private func flushPendingPrompts() {
        guard webView.url?.host == "gemini.google.com" else { return }
        let prompts = pendingPrompts
        pendingPrompts.removeAll()
        for prompt in prompts { inject(prompt.text, send: prompt.send) }
    }

    private func inject(_ text: String, send: Bool) {
        webView.callAsyncJavaScript(Self.injectScript, arguments: ["text": text, "send": send], in: nil, in: .page) { [weak self] result in
            guard let self else { return }
            let status = (try? result.get()) as? String ?? "error"
            NSLog("Gemini Sidebar: prompt injection → \(status)")
            if status == "no-input" || status == "error" {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(text, forType: .string)
                self.showToast("Couldn't reach Gemini's prompt box — prompt copied, paste with ⌘V")
            }
            self.panel.makeFirstResponder(self.webView)
        }
    }

    private func focusPromptBox() {
        webView.evaluateJavaScript("document.querySelector('rich-textarea .ql-editor, div.ql-editor[contenteditable=\"true\"]')?.focus()")
    }

    /// Runs as the body of an async function with `text` and `send` in scope.
    private static let injectScript = #"""
    const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
    const findInput = () =>
      document.querySelector('rich-textarea .ql-editor[contenteditable="true"]') ||
      document.querySelector('div.ql-editor[contenteditable="true"]') ||
      document.querySelector('[contenteditable="true"][role="textbox"]') ||
      document.querySelector('textarea');

    let input = null;
    for (let i = 0; i < 100 && !(input = findInput()); i++) await sleep(200);
    if (!input) return "no-input";

    input.focus();
    if (input.tagName === "TEXTAREA") {
      input.value = text;
      input.dispatchEvent(new Event("input", { bubbles: true }));
    } else {
      const range = document.createRange();
      range.selectNodeContents(input);
      const sel = window.getSelection();
      sel.removeAllRanges();
      sel.addRange(range);
      document.execCommand("insertText", false, text);
      if (!input.innerText.trim()) {
        // Fallback if the editor ignored insertText.
        input.innerHTML = "";
        for (const line of text.split("\n")) {
          const p = document.createElement("p");
          p.textContent = line;
          if (!line) p.appendChild(document.createElement("br"));
          input.appendChild(p);
        }
        input.dispatchEvent(new InputEvent("input", { bubbles: true }));
      }
    }
    if (!send) return "inserted";

    await sleep(300);
    for (let i = 0; i < 50; i++) {
      const button =
        document.querySelector('button.send-button') ||
        document.querySelector('button[aria-label="Send message"]') ||
        document.querySelector('button[aria-label*="Send" i]');
      if (button && !button.disabled && button.getAttribute("aria-disabled") !== "true") {
        button.click();
        return "sent";
      }
      await sleep(200);
    }
    input.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", code: "Enter", keyCode: 13, which: 13, bubbles: true }));
    return "enter";
    """#

    // MARK: UI

    private func buildPanel() {
        panel = SidebarPanel(
            contentRect: NSRect(x: 0, y: 0, width: 440, height: 800),
            styleMask: [.titled, .closable, .resizable, .fullSizeContentView],
            backing: .buffered, defer: false)
        panel.title = "Gemini"
        panel.titleVisibility = .hidden
        panel.titlebarAppearsTransparent = true
        panel.isFloatingPanel = true
        panel.hidesOnDeactivate = false
        panel.isReleasedWhenClosed = false
        panel.becomesKeyOnlyIfNeeded = false
        panel.collectionBehavior = [.moveToActiveSpace, .fullScreenAuxiliary]
        panel.minSize = NSSize(width: Self.minWidth, height: 400)
        panel.backgroundColor = .windowBackgroundColor
        panel.delegate = self
        [.closeButton, .miniaturizeButton, .zoomButton].forEach { panel.standardWindowButton($0)?.isHidden = true }
        applyLevel()

        let content = NSView()
        panel.contentView = content

        let header = makeHeader()
        webView = makeWebView()
        toast = makeToast()
        for view in [header, webView!, toast!] {
            view.translatesAutoresizingMaskIntoConstraints = false
            content.addSubview(view)
        }
        NSLayoutConstraint.activate([
            header.topAnchor.constraint(equalTo: content.topAnchor),
            header.leadingAnchor.constraint(equalTo: content.leadingAnchor),
            header.trailingAnchor.constraint(equalTo: content.trailingAnchor),
            header.heightAnchor.constraint(equalToConstant: Self.headerHeight),
            webView.topAnchor.constraint(equalTo: header.bottomAnchor),
            webView.leadingAnchor.constraint(equalTo: content.leadingAnchor),
            webView.trailingAnchor.constraint(equalTo: content.trailingAnchor),
            webView.bottomAnchor.constraint(equalTo: content.bottomAnchor),
            toast.centerXAnchor.constraint(equalTo: content.centerXAnchor),
            toast.bottomAnchor.constraint(equalTo: content.bottomAnchor, constant: -24),
            toast.widthAnchor.constraint(lessThanOrEqualTo: content.widthAnchor, constant: -32),
        ])
    }

    private func makeHeader() -> NSView {
        let header = NSVisualEffectView()
        header.material = .titlebar
        header.blendingMode = .withinWindow

        let icon = NSImageView(image: NSImage(systemSymbolName: "sparkle", accessibilityDescription: nil)!)
        icon.contentTintColor = .systemBlue
        let title = NSTextField(labelWithString: "Gemini")
        title.font = .systemFont(ofSize: 14, weight: .semibold)

        let buttons = [
            headerButton("square.and.pencil", "New chat (⌘N)", #selector(newChatClicked)),
            headerButton("arrow.clockwise", "Reload (⌘R)", #selector(reloadClicked)),
            headerButton("safari", "Open this chat in Safari", #selector(openInBrowserClicked)),
            headerButton("xmark", "Close sidebar (⌃⌥G)", #selector(closeClicked)),
        ]
        let left = NSStackView(views: [icon, title])
        left.spacing = 6
        let right = NSStackView(views: buttons)
        right.spacing = 2
        let divider = NSBox()
        divider.boxType = .separator

        for view in [left, right, divider] {
            view.translatesAutoresizingMaskIntoConstraints = false
            header.addSubview(view)
        }
        NSLayoutConstraint.activate([
            left.leadingAnchor.constraint(equalTo: header.leadingAnchor, constant: 14),
            left.centerYAnchor.constraint(equalTo: header.centerYAnchor),
            right.trailingAnchor.constraint(equalTo: header.trailingAnchor, constant: -8),
            right.centerYAnchor.constraint(equalTo: header.centerYAnchor),
            divider.leadingAnchor.constraint(equalTo: header.leadingAnchor),
            divider.trailingAnchor.constraint(equalTo: header.trailingAnchor),
            divider.bottomAnchor.constraint(equalTo: header.bottomAnchor),
        ])
        return header
    }

    private func headerButton(_ symbol: String, _ tip: String, _ action: Selector) -> NSButton {
        let button = NSButton(image: NSImage(systemSymbolName: symbol, accessibilityDescription: tip)!, target: self, action: action)
        button.bezelStyle = .accessoryBarAction
        button.isBordered = false
        button.toolTip = tip
        button.widthAnchor.constraint(equalToConstant: 28).isActive = true
        button.heightAnchor.constraint(equalToConstant: 28).isActive = true
        return button
    }

    private func makeWebView() -> WKWebView {
        let config = WKWebViewConfiguration()
        config.websiteDataStore = .default() // persists Google sign-in between launches
        config.preferences.isElementFullscreenEnabled = true
        config.mediaTypesRequiringUserActionForPlayback = []
        // Google rejects sign-in from "embedded browsers"; identify as Safari.
        config.applicationNameForUserAgent = "Version/\(Self.safariVersion) Safari/605.1.15"

        let webView = WKWebView(frame: .zero, configuration: config)
        webView.navigationDelegate = self
        webView.uiDelegate = self
        webView.allowsBackForwardNavigationGestures = true
        webView.allowsMagnification = true
        webView.isInspectable = true
        return webView
    }

    private static var safariVersion: String {
        let plist = URL(fileURLWithPath: "/Applications/Safari.app/Contents/Info.plist")
        let info = NSDictionary(contentsOf: plist)
        return info?["CFBundleShortVersionString"] as? String ?? "26.0"
    }

    private func makeToast() -> NSTextField {
        let label = NSTextField(wrappingLabelWithString: "")
        label.alignment = .center
        label.textColor = .white
        label.font = .systemFont(ofSize: 12, weight: .medium)
        label.drawsBackground = true
        label.backgroundColor = NSColor.black.withAlphaComponent(0.8)
        label.wantsLayer = true
        label.layer?.cornerRadius = 8
        label.isHidden = true
        return label
    }

    private func showToast(_ message: String) {
        toast.stringValue = "  \(message)  "
        toast.isHidden = false
        DispatchQueue.main.asyncAfter(deadline: .now() + 4) { [weak self] in self?.toast.isHidden = true }
    }

    @objc private func newChatClicked() { newChat() }
    @objc private func reloadClicked() { reload() }
    @objc private func closeClicked() { hide() }
    @objc private func openInBrowserClicked() {
        NSWorkspace.shared.open(webView.url ?? Self.home)
    }

    private static func isGoogleHost(_ url: URL?) -> Bool {
        guard let host = url?.host?.lowercased() else { return false }
        let allowed = ["google.com", "gstatic.com", "googleusercontent.com", "youtube.com", "googleapis.com"]
        return allowed.contains { host == $0 || host.hasSuffix("." + $0) } || host.hasPrefix("www.google.")
    }
}

// MARK: - NSWindowDelegate

extension SidebarController: NSWindowDelegate {
    func windowShouldClose(_ sender: NSWindow) -> Bool {
        if sender == panel { hide(); return false }
        return true
    }

    func windowDidEndLiveResize(_ notification: Notification) {
        guard (notification.object as? NSWindow) == panel else { return }
        width = panel.frame.width
        dock() // snap back to the screen edge
        if makeRoomInSafari { safariFitter.makeRoom(for: panel.frame, side: side) }
    }

    func windowWillClose(_ notification: Notification) {
        if let window = notification.object as? NSWindow { popups.removeAll { $0 == window } }
    }
}

// MARK: - WKNavigationDelegate

extension SidebarController: WKNavigationDelegate {
    func webView(_ webView: WKWebView, decidePolicyFor action: WKNavigationAction,
                 decisionHandler: @escaping (WKNavigationActionPolicy) -> Void) {
        let url = action.request.url
        if action.shouldPerformDownload { return decisionHandler(.download) }
        // Links in Gemini's answers that point elsewhere open in the default browser.
        if action.targetFrame?.isMainFrame == true, action.navigationType == .linkActivated,
           let url, url.scheme?.hasPrefix("http") == true, !Self.isGoogleHost(url) {
            NSWorkspace.shared.open(url)
            return decisionHandler(.cancel)
        }
        decisionHandler(.allow)
    }

    func webView(_ webView: WKWebView, decidePolicyFor response: WKNavigationResponse,
                 decisionHandler: @escaping (WKNavigationResponsePolicy) -> Void) {
        decisionHandler(response.canShowMIMEType ? .allow : .download)
    }

    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
        if webView == self.webView { flushPendingPrompts() }
    }

    func webView(_ webView: WKWebView, navigationAction: WKNavigationAction, didBecome download: WKDownload) {
        download.delegate = self
    }

    func webView(_ webView: WKWebView, navigationResponse: WKNavigationResponse, didBecome download: WKDownload) {
        download.delegate = self
    }

    func webViewWebContentProcessDidTerminate(_ webView: WKWebView) {
        webView.reload()
    }
}

// MARK: - WKUIDelegate (popups, file pickers, dialogs, microphone)

extension SidebarController: WKUIDelegate {
    func webView(_ webView: WKWebView, createWebViewWith configuration: WKWebViewConfiguration,
                 for action: WKNavigationAction, windowFeatures: WKWindowFeatures) -> WKWebView? {
        let url = action.request.url
        if let url, url.scheme?.hasPrefix("http") == true, !Self.isGoogleHost(url) {
            NSWorkspace.shared.open(url)
            return nil
        }
        // Google sign-in / account popups stay inside the app so cookies land in the sidebar.
        let popupView = WKWebView(frame: NSRect(x: 0, y: 0, width: 500, height: 640), configuration: configuration)
        popupView.uiDelegate = self
        popupView.navigationDelegate = self
        let window = NSWindow(contentRect: popupView.frame, styleMask: [.titled, .closable, .resizable],
                              backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        window.level = .floating
        window.contentView = popupView
        window.delegate = self
        window.center()
        window.makeKeyAndOrderFront(nil)
        popups.append(window)
        return popupView
    }

    func webViewDidClose(_ webView: WKWebView) {
        popups.first { $0.contentView == webView }?.close()
    }

    func webView(_ webView: WKWebView, runOpenPanelWith parameters: WKOpenPanelParameters,
                 initiatedByFrame frame: WKFrameInfo, completionHandler: @escaping ([URL]?) -> Void) {
        let open = NSOpenPanel()
        open.allowsMultipleSelection = parameters.allowsMultipleSelection
        open.canChooseDirectories = parameters.allowsDirectories
        open.canChooseFiles = true
        open.beginSheetModal(for: webView.window ?? panel) { response in
            completionHandler(response == .OK ? open.urls : nil)
        }
    }

    func webView(_ webView: WKWebView, runJavaScriptAlertPanelWithMessage message: String,
                 initiatedByFrame frame: WKFrameInfo, completionHandler: @escaping () -> Void) {
        let alert = NSAlert()
        alert.messageText = message
        alert.beginSheetModal(for: webView.window ?? panel) { _ in completionHandler() }
    }

    func webView(_ webView: WKWebView, runJavaScriptConfirmPanelWithMessage message: String,
                 initiatedByFrame frame: WKFrameInfo, completionHandler: @escaping (Bool) -> Void) {
        let alert = NSAlert()
        alert.messageText = message
        alert.addButton(withTitle: "OK")
        alert.addButton(withTitle: "Cancel")
        alert.beginSheetModal(for: webView.window ?? panel) { completionHandler($0 == .alertFirstButtonReturn) }
    }

    func webView(_ webView: WKWebView, requestMediaCapturePermissionFor origin: WKSecurityOrigin,
                 initiatedByFrame frame: WKFrameInfo, type: WKMediaCaptureType,
                 decisionHandler: @escaping (WKPermissionDecision) -> Void) {
        decisionHandler(origin.host.hasSuffix("google.com") ? .grant : .deny)
    }
}

// MARK: - WKDownloadDelegate (images and files Gemini generates)

extension SidebarController: WKDownloadDelegate {
    func download(_ download: WKDownload, decideDestinationUsing response: URLResponse,
                  suggestedFilename: String, completionHandler: @escaping (URL?) -> Void) {
        let folder = FileManager.default.urls(for: .downloadsDirectory, in: .userDomainMask)[0]
        var destination = folder.appendingPathComponent(suggestedFilename)
        let base = destination.deletingPathExtension().lastPathComponent
        let ext = destination.pathExtension
        var n = 2
        while FileManager.default.fileExists(atPath: destination.path) {
            destination = folder.appendingPathComponent(ext.isEmpty ? "\(base) \(n)" : "\(base) \(n).\(ext)")
            n += 1
        }
        completionHandler(destination)
    }

    func downloadDidFinish(_ download: WKDownload) {
        showToast("Saved to Downloads")
    }

    func download(_ download: WKDownload, didFailWithError error: Error, resumeData: Data?) {
        showToast("Download failed: \(error.localizedDescription)")
    }
}
