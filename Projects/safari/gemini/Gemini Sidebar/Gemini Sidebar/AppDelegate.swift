import Carbon.HIToolbox
import Cocoa
import SafariServices

let extensionBundleIdentifier = "dev.local.GeminiSidebar.Extension"

final class AppDelegate: NSObject, NSApplicationDelegate, NSMenuDelegate {
    let sidebar = SidebarController()
    private var statusItem: NSStatusItem!
    private var hotKey: HotKey?
    private var onboarding: OnboardingWindowController?
    private var launchedViaURL = false

    // MARK: Lifecycle

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.accessory)
        NSApp.mainMenu = buildMainMenu()
        setUpStatusItem()

        // ⌃⌥G toggles the sidebar from anywhere.
        hotKey = HotKey(keyCode: UInt32(kVK_ANSI_G), modifiers: UInt32(controlKey | optionKey)) { [weak self] in
            self?.sidebar.toggle()
        }

        // When launched by the Safari extension, the URL event decides what to show.
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { [weak self] in
            guard let self, !self.launchedViaURL else { return }
            self.showOnboardingIfNeeded(otherwise: { self.sidebar.show() })
        }
    }

    func application(_ application: NSApplication, open urls: [URL]) {
        launchedViaURL = true
        urls.forEach(handle)
    }

    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool) -> Bool {
        sidebar.show()
        return false
    }

    func applicationWillTerminate(_ notification: Notification) {
        sidebar.restoreSafariWindow()
    }

    // MARK: URL commands from the Safari extension

    private func handle(_ url: URL) {
        guard url.scheme == "geminisidebar" else { return }
        let items = URLComponents(url: url, resolvingAgainstBaseURL: false)?.queryItems ?? []
        func value(_ name: String) -> String? { items.first { $0.name == name }?.value }

        switch url.host {
        case "toggle": sidebar.toggle()
        case "show": sidebar.show()
        case "hide": sidebar.hide()
        case "new": sidebar.newChat()
        case "ask": sidebar.ask(value("prompt") ?? "", send: value("send") == "1")
        default: NSLog("%@", "Gemini Sidebar: unknown command \(url.absoluteString)")
        }
    }

    // MARK: Onboarding

    private func showOnboardingIfNeeded(otherwise: @escaping () -> Void) {
        SFSafariExtensionManager.getStateOfSafariExtension(withIdentifier: extensionBundleIdentifier) { state, _ in
            let enabled = state?.isEnabled ?? false
            Task { @MainActor in
                if enabled { otherwise() } else { self.showOnboarding() }
            }
        }
    }

    @objc func showOnboarding() {
        if onboarding == nil {
            onboarding = OnboardingWindowController(onShowSidebar: { [weak self] in self?.sidebar.show() })
        }
        onboarding?.refresh()
        NSApp.activate()
        onboarding?.showWindow(nil)
        onboarding?.window?.center()
        onboarding?.window?.makeKeyAndOrderFront(nil)
    }

    // MARK: Menus

    private func setUpStatusItem() {
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.squareLength)
        let image = NSImage(systemSymbolName: "sparkle", accessibilityDescription: "Gemini Sidebar")
        image?.isTemplate = true
        statusItem.button?.image = image
        let menu = NSMenu()
        menu.delegate = self
        statusItem.menu = menu
    }

    /// Rebuilt on open so checkmarks reflect current settings.
    func menuNeedsUpdate(_ menu: NSMenu) {
        guard menu == statusItem.menu else { return }
        menu.removeAllItems()

        let toggle = item(sidebar.isVisible ? "Hide Gemini Sidebar" : "Show Gemini Sidebar", #selector(toggleSidebar))
        toggle.keyEquivalent = "g"
        toggle.keyEquivalentModifierMask = [.control, .option]
        menu.addItem(toggle)
        menu.addItem(item("New Chat", #selector(newChat)))
        menu.addItem(item("Reload", #selector(reload)))
        menu.addItem(.separator())

        let right = item("Dock on Right", #selector(dockRight))
        right.state = sidebar.side == .right ? .on : .off
        let left = item("Dock on Left", #selector(dockLeft))
        left.state = sidebar.side == .left ? .on : .off
        let onTop = item("Keep Above Other Windows", #selector(toggleKeepOnTop))
        onTop.state = sidebar.keepOnTop ? .on : .off
        let room = item("Make Room in Safari Window", #selector(toggleMakeRoom))
        room.state = sidebar.makeRoomInSafari ? .on : .off
        [right, left, onTop, room].forEach(menu.addItem)
        menu.addItem(.separator())

        menu.addItem(item("Setup & Safari Extension Settings…", #selector(showOnboarding)))
        menu.addItem(item("Sign Out of Google…", #selector(signOut)))
        menu.addItem(.separator())
        menu.addItem(NSMenuItem(title: "Quit Gemini Sidebar", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q"))
    }

    private func item(_ title: String, _ action: Selector) -> NSMenuItem {
        let item = NSMenuItem(title: title, action: action, keyEquivalent: "")
        item.target = self
        return item
    }

    /// Never shown (accessory app) but provides ⌘C/⌘V/⌘W etc. while the sidebar is focused.
    private func buildMainMenu() -> NSMenu {
        let main = NSMenu()

        let appMenu = NSMenu()
        appMenu.addItem(withTitle: "Hide Sidebar", action: #selector(toggleSidebar), keyEquivalent: "h").target = self
        appMenu.addItem(withTitle: "Quit Gemini Sidebar", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        main.addItem(submenu: appMenu, title: "Gemini Sidebar")

        let edit = NSMenu(title: "Edit")
        edit.addItem(withTitle: "Undo", action: Selector(("undo:")), keyEquivalent: "z")
        edit.addItem(withTitle: "Redo", action: Selector(("redo:")), keyEquivalent: "Z")
        edit.addItem(.separator())
        edit.addItem(withTitle: "Cut", action: #selector(NSText.cut(_:)), keyEquivalent: "x")
        edit.addItem(withTitle: "Copy", action: #selector(NSText.copy(_:)), keyEquivalent: "c")
        edit.addItem(withTitle: "Paste", action: #selector(NSText.paste(_:)), keyEquivalent: "v")
        edit.addItem(withTitle: "Select All", action: #selector(NSText.selectAll(_:)), keyEquivalent: "a")
        main.addItem(submenu: edit, title: "Edit")

        let view = NSMenu(title: "View")
        view.addItem(withTitle: "New Chat", action: #selector(newChat), keyEquivalent: "n").target = self
        view.addItem(withTitle: "Reload", action: #selector(reload), keyEquivalent: "r").target = self
        view.addItem(withTitle: "Actual Size", action: #selector(zoomReset), keyEquivalent: "0").target = self
        view.addItem(withTitle: "Zoom In", action: #selector(zoomIn), keyEquivalent: "+").target = self
        view.addItem(withTitle: "Zoom Out", action: #selector(zoomOut), keyEquivalent: "-").target = self
        main.addItem(submenu: view, title: "View")

        let window = NSMenu(title: "Window")
        window.addItem(withTitle: "Close", action: #selector(NSWindow.performClose(_:)), keyEquivalent: "w")
        main.addItem(submenu: window, title: "Window")
        return main
    }

    // MARK: Actions

    @objc private func toggleSidebar() { sidebar.toggle() }
    @objc private func newChat() { sidebar.newChat() }
    @objc private func reload() { sidebar.reload() }
    @objc private func dockRight() { sidebar.side = .right }
    @objc private func dockLeft() { sidebar.side = .left }
    @objc private func toggleKeepOnTop() { sidebar.keepOnTop.toggle() }
    @objc private func toggleMakeRoom() { sidebar.makeRoomInSafari.toggle() }
    @objc private func zoomIn() { sidebar.zoom(by: 0.1) }
    @objc private func zoomOut() { sidebar.zoom(by: -0.1) }
    @objc private func zoomReset() { sidebar.zoom(by: nil) }

    @objc private func signOut() {
        let alert = NSAlert()
        alert.messageText = "Sign out of Google in the sidebar?"
        alert.informativeText = "This clears the sidebar's cookies and website data. Safari is not affected."
        alert.addButton(withTitle: "Sign Out")
        alert.addButton(withTitle: "Cancel")
        NSApp.activate()
        if alert.runModal() == .alertFirstButtonReturn { sidebar.clearWebsiteData() }
    }
}

private extension NSMenu {
    func addItem(submenu: NSMenu, title: String) {
        let item = NSMenuItem(title: title, action: nil, keyEquivalent: "")
        item.submenu = submenu
        addItem(item)
    }
}
