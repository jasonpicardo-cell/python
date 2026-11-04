import CoreAudio
import SwiftUI

final class AppDelegate: NSObject, NSApplicationDelegate {
    private var statusItem: NSStatusItem?
    private let model = AppModel.shared

    // The menu-bar icon stays after the window is closed; quit from its menu (or ⌘Q).
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { false }

    func applicationWillFinishLaunching(_ notification: Notification) {
        // Opened by "Open at Login": run silently as a menu-bar item, no window or Dock icon.
        let launchedAtLogin = NSAppleEventManager.shared().currentAppleEvent?
            .paramDescriptor(forKeyword: AEKeyword(keyAEPropData))?.enumCodeValue == OSType(keyAELaunchedAsLogInItem)
        // The login-item Apple event isn't always delivered (newer macOS): also treat a launch
        // within a few minutes of boot as an automatic start.
        var boot = timeval(); var size = MemoryLayout<timeval>.stride
        let justBooted = sysctlbyname("kern.boottime", &boot, &size, nil, 0) == 0
            && Date().timeIntervalSince1970 - TimeInterval(boot.tv_sec) < 300
        if launchedAtLogin || justBooted {
            WindowOpener.startHidden = true
            NSApp.setActivationPolicy(.accessory)
        }
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        let item = NSStatusBar.system.statusItem(withLength: NSStatusItem.squareLength)
        if let button = item.button {
            button.image = NSImage(systemSymbolName: "square.grid.3x3.fill", accessibilityDescription: "Pixel Frame")
            button.image?.isTemplate = true
            button.toolTip = "Click: next image as wallpaper · Right-click: menu"
            button.target = self
            button.action = #selector(statusClicked)
            button.sendAction(on: [.leftMouseUp, .rightMouseUp])
        }
        statusItem = item

        // Closing the window drops back to menu-bar-only: no Dock icon, not in ⌘Tab.
        NotificationCenter.default.addObserver(forName: NSWindow.willCloseNotification, object: nil, queue: .main) { note in
            guard let w = note.object as? NSWindow, w.title == "Pixel Frame" else { return }
            DispatchQueue.main.async {
                let stillOpen = NSApp.windows.contains { $0.title == "Pixel Frame" && $0 !== w && $0.isVisible }
                if !stillOpen {
                    NSApp.setActivationPolicy(.accessory)
                    // The window's SwiftUI tree was just torn down: give its memory back.
                    DispatchQueue.main.asyncAfter(deadline: .now() + 2) { AppModel.returnMemoryToSystem() }
                }
            }
        }
    }

    func applicationWillTerminate(_ notification: Notification) {
        MonitorControl.enableAll()
    }

    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool) -> Bool {
        showMainWindow()
        return true
    }

    @objc private func statusClicked() {
        let event = NSApp.currentEvent
        if event?.type == .rightMouseUp || event?.modifierFlags.contains(.control) == true {
            showMenu()
        } else {
            changeWallpaper(advance: true)
        }
    }

    /// Dock icon right-click menu: same monitor toggles as the menu-bar icon.
    func applicationDockMenu(_ sender: NSApplication) -> NSMenu? {
        model.refreshMonitors()
        let menu = NSMenu()
        addMonitorItems(to: menu)
        return menu
    }

    /// One checkable item per external monitor (ticked = on); choosing it switches that monitor off/on.
    private func addMonitorItems(to menu: NSMenu) {
        let monitors = model.monitors
        guard !monitors.isEmpty else { return }
        let header = NSMenuItem(title: "External Monitors", action: nil, keyEquivalent: "")
        header.isEnabled = false
        menu.addItem(header)
        for m in monitors {
            let item = menu.addItem(withTitle: m.name, action: #selector(menuToggleMonitor(_:)), keyEquivalent: "")
            item.target = self
            item.representedObject = m.id
            item.state = m.isOn ? .on : .off
        }
        if monitors.contains(where: { !$0.isOn }) {
            let all = menu.addItem(withTitle: "Turn All Monitors On", action: #selector(menuAllMonitorsOn), keyEquivalent: "")
            all.target = self
        }
    }

    @objc private func menuToggleMonitor(_ sender: NSMenuItem) {
        guard let id = sender.representedObject as? String,
              let m = model.monitors.first(where: { $0.id == id }) else { return }
        model.setMonitor(m, on: !m.isOn)
    }

    @objc private func menuAllMonitorsOn() {
        model.monitors.filter { !$0.isOn }.forEach { model.setMonitor($0, on: true) }
    }

    /// "Sound" section: one item per output device (ticked = current); choosing one routes sound to it.
    private func addSoundItems(to menu: NSMenu) {
        let devices = AudioOutput.outputDevices()
        guard !devices.isEmpty else { return }
        let header = NSMenuItem(title: "Sound", action: nil, keyEquivalent: "")
        header.isEnabled = false
        menu.addItem(header)
        for d in devices {
            let item = menu.addItem(withTitle: d.name, action: #selector(menuSelectSound(_:)), keyEquivalent: "")
            item.target = self
            item.tag = Int(d.id)
            item.state = d.isDefault ? .on : .off
        }
    }

    @objc private func menuSelectSound(_ sender: NSMenuItem) {
        AudioOutput.setDefault(AudioDeviceID(sender.tag))
    }

    private func showMenu() {
        model.refreshMonitors()
        let menu = NSMenu()
        if !model.monitors.isEmpty {
            addMonitorItems(to: menu)
            menu.addItem(.separator())
        }
        addSoundItems(to: menu)
        menu.addItem(.separator())
        menu.addItem(withTitle: "Next Image as Wallpaper", action: #selector(menuNext), keyEquivalent: "").target = self
        menu.addItem(withTitle: "Set Current Image as Wallpaper", action: #selector(menuCurrent), keyEquivalent: "").target = self
        let auto = menu.addItem(withTitle: "Auto-change Wallpaper", action: #selector(menuAuto), keyEquivalent: "")
        auto.target = self
        auto.state = model.autoWallpaper ? .on : .off
        let restore = menu.addItem(withTitle: "Restore System Wallpaper", action: #selector(menuRestore), keyEquivalent: "")
        restore.target = self
        restore.isEnabled = model.wallpaperIsActive
        menu.addItem(.separator())
        menu.addItem(withTitle: "Open Pixel Frame", action: #selector(menuOpen), keyEquivalent: "").target = self
        menu.addItem(withTitle: "Quit Pixel Frame", action: #selector(menuQuit), keyEquivalent: "q").target = self
        statusItem?.menu = menu
        statusItem?.button?.performClick(nil)
        statusItem?.menu = nil                      // restore click-to-action behaviour
    }

    @objc private func menuNext() { changeWallpaper(advance: true) }
    @objc private func menuCurrent() { changeWallpaper(advance: false) }
    @objc private func menuRestore() { model.removeWallpaper() }
    @objc private func menuAuto() { model.autoWallpaper.toggle() }
    @objc private func menuOpen() { showMainWindow() }
    @objc private func menuQuit() { NSApp.terminate(nil) }

    private func changeWallpaper(advance: Bool) {
        let ok = advance ? model.nextWallpaper() : model.applyWallpaper()
        guard ok else { showMainWindow(); return }   // nothing loaded yet: let the user add images
        flashIcon()
    }

    private func flashIcon() {
        guard let button = statusItem?.button else { return }
        let original = button.image
        button.image = NSImage(systemSymbolName: "checkmark.circle.fill", accessibilityDescription: "Wallpaper changed")
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.8) { button.image = original }
    }

    private func showMainWindow() {
        NSApp.setActivationPolicy(.regular)
        NSApp.activate()
        if let w = NSApp.windows.first(where: { $0.title == "Pixel Frame" }) {
            if w.isMiniaturized { w.deminiaturize(nil) }
            w.makeKeyAndOrderFront(nil)
        } else {
            WindowOpener.open?()                    // SwiftUI discarded the closed window: ask it for a new one
        }
    }
}

@main
struct PixelFrameApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    private let model = AppModel.shared

    var body: some Scene {
        Window("Pixel Frame", id: "main") {
            ContentView()
                .environmentObject(model)
        }
        .defaultSize(width: 1100, height: 700)
        .commands {
            CommandGroup(replacing: .newItem) {
                Button("Add Images or Folders…") { model.presentOpenPanel() }
                    .keyboardShortcut("o")
            }
            CommandMenu("Slideshow") {
                Button("Play / Pause") { model.playing.toggle() }
                    .keyboardShortcut("p", modifiers: [.command, .option])
                Button("Next Image") { model.next() }
                    .keyboardShortcut("]", modifiers: .command)
                Button("Previous Image") { model.previous() }
                    .keyboardShortcut("[", modifiers: .command)
                Divider()
                Button("Next Image as Wallpaper") { model.nextWallpaper() }
                    .keyboardShortcut("w", modifiers: [.command, .shift])
                Button("Set Current Image as Wallpaper") { model.applyWallpaper() }
                Divider()
                Button("Show Full Screen Pixel Display") { model.enterPresentation() }
                    .keyboardShortcut("f", modifiers: [.command, .shift])
            }
        }
    }
}
