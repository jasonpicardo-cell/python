import SwiftUI

struct ContentView: View {
    @EnvironmentObject var model: AppModel
    @Environment(\.openWindow) private var openWindow

    var body: some View {
        HStack(spacing: 0) {
            if !model.isPresenting {
                ControlPanel()
                    .frame(width: 330)
                Divider()
            }
            DisplayArea()
        }
        .background(Color.black)
        .frame(minWidth: 820, minHeight: 560)
        .onDrop(of: [.fileURL], isTargeted: nil) { providers in
            let lock = NSLock()
            var urls: [URL] = []
            let group = DispatchGroup()
            for provider in providers {
                group.enter()
                _ = provider.loadObject(ofClass: URL.self) { url, _ in
                    if let url { lock.lock(); urls.append(url); lock.unlock() }
                    group.leave()
                }
            }
            group.notify(queue: .main) { if !urls.isEmpty { model.addSources(urls) } }
            return true
        }
        .onAppear {
            WindowOpener.open = { openWindow(id: "main") }   // lets the menu-bar item recreate a closed window
            if WindowOpener.startHidden {                    // launched at login: stay in the menu bar
                WindowOpener.startHidden = false
                DispatchQueue.main.async {
                    NSApp.windows.filter { $0.title == "Pixel Frame" }.forEach { $0.close() }
                }
                return
            }
            NSApp.setActivationPolicy(.regular)
            NSApp.activate()
        }
    }
}

enum WindowOpener {
    static var open: (() -> Void)?
    static var startHidden = false
}

struct DisplayArea: View {
    @EnvironmentObject var model: AppModel

    var body: some View {
        ZStack {
            Color.black
            if model.displayFrame != nil {
                PixelDisplay(
                    image: model.displayFrame,
                    itemID: model.currentItemID,
                    gap: model.shownGap,
                    integerScale: model.shownIntegerScale,
                    fade: model.fade && !model.reducedEffects
                )
            } else {
                EmptyState()
            }
            if model.showClock && model.displayFrame != nil {
                ClockOverlay()
            }
            if model.isPresenting && model.hudVisible {
                VStack {
                    Spacer()
                    HUD().padding(.bottom, 40)
                }
            }
        }
        .onContinuousHover { phase in
            if case .active = phase, model.isPresenting { model.pokeHUD() }
        }
        .onTapGesture(count: 2) {
            if model.isPresenting { model.exitPresentation() } else { model.enterPresentation() }
        }
    }
}

/// Time (hh:mm:ss AM/PM) and date (dd-MMM-yyyy). Hops to a new corner/edge with every new image.
/// Rendered by a layer-backed AppKit view (see ClockNSView) so the per-second tick never re-runs SwiftUI layout.
struct ClockOverlay: View {
    @EnvironmentObject var model: AppModel
    /// Wallpaper style: one line "hh:mm:ss dd.MMM.yy EEE", fixed at the bottom left.
    var fixedLine = false

    var body: some View {
        ClockDisplay(slot: fixedLine ? 6 : model.clockSlot,
                     fixedLine: fixedLine,
                     running: model.suspendReasons.isEmpty)
            .allowsHitTesting(false)
    }
}

struct EmptyState: View {
    @EnvironmentObject var model: AppModel

    var body: some View {
        VStack(spacing: 14) {
            Image(systemName: "square.grid.3x3.fill")
                .font(.system(size: 44))
                .foregroundStyle(.secondary)
            if model.isCreatingCache {
                Text("Creating cache…").foregroundStyle(.secondary)
            } else if model.sourcePaths.isEmpty {
                Text("Add or drop images or a folder to begin").foregroundStyle(.secondary)
                Button("Add Images or Folders…") { model.presentOpenPanel() }
            } else if model.playlist.isEmpty {
                Text("Choose your settings, then click Create Cache").foregroundStyle(.secondary)
                Button("Create Cache") { model.createCache() }
            } else {
                Text("Loading…").foregroundStyle(.secondary)
            }
        }
    }
}

struct HUD: View {
    @EnvironmentObject var model: AppModel

    var body: some View {
        HStack(spacing: 20) {
            Button { model.previous() } label: { Image(systemName: "backward.fill") }
            Button { model.playing.toggle() } label: {
                Image(systemName: model.playing ? "pause.fill" : "play.fill")
            }
            Button { model.next() } label: { Image(systemName: "forward.fill") }
            Text("\(model.index + 1) / \(model.playlist.count)")
                .monospacedDigit()
                .font(.callout)
            Button { model.exitPresentation() } label: {
                Image(systemName: "arrow.down.right.and.arrow.up.left")
            }
        }
        .buttonStyle(.plain)
        .font(.title2)
        .foregroundStyle(.white)
        .padding(.horizontal, 24)
        .padding(.vertical, 12)
        .background(.ultraThinMaterial, in: Capsule())
    }
}

/// Slider that takes keyboard focus (Tab) and responds to arrow keys; Shift = 5x steps.
struct KeyboardSlider: View {
    @Binding var value: Double
    let range: ClosedRange<Double>
    let step: Double

    var body: some View {
        Slider(value: $value, in: range, step: step)
            .focusable()
            .onKeyPress(phases: [.down, .repeat]) { press in
                let direction: Double
                switch press.key {
                case .leftArrow, .downArrow: direction = -1
                case .rightArrow, .upArrow: direction = 1
                default: return .ignored
                }
                let amount = step * (press.modifiers.contains(.shift) ? 5 : 1)
                value = min(range.upperBound, max(range.lowerBound, value + direction * amount))
                return .handled
            }
    }
}

struct ControlPanel: View {
    @EnvironmentObject var model: AppModel

    var body: some View {
        Form {
            Section {
                if model.monitors.isEmpty {
                    Text("No external monitors connected.").foregroundStyle(.secondary)
                }
                ForEach(model.monitors) { m in
                    Toggle(m.name, isOn: Binding(
                        get: { m.isOn },
                        set: { model.setMonitor(m, on: $0) }))
                }
                HStack {
                    Button("Refresh") { model.refreshMonitors() }
                    Button("Turn All On") { model.monitors.filter { !$0.isOn }.forEach { model.setMonitor($0, on: true) } }
                        .disabled(model.monitors.allSatisfy(\.isOn))
                }
                if !model.monitorMessage.isEmpty {
                    Text(model.monitorMessage).font(.caption).foregroundStyle(.orange)
                }
            } header: {
                Text("External monitors")
            } footer: {
                Text("Untick to switch off a Thunderbolt / HDMI / DisplayPort monitor. It is disconnected from macOS and goes to standby; windows move to your other screens. Monitors are turned back on when Pixel Frame quits.")
                    .font(.caption).foregroundStyle(.secondary)
            }

            Section("Library") {
                HStack {
                    Button("Add Images/Folders…") { model.presentOpenPanel() }
                    Button("Clear", role: .destructive) { model.clearSources() }
                        .disabled(model.sourcePaths.isEmpty)
                }
                LabeledContent("Sources", value: "\(model.sourcePaths.count)")
                LabeledContent("Cached images", value: "\(model.playlist.count)")
                LabeledContent("Cached in RAM", value: model.ramSummary)
            }

            Section("Pixel display") {
                Picker("Resolution", selection: $model.gridID) {
                    Text("Auto · fits this screen (\(model.autoGrid.width) × \(model.autoGrid.height))")
                        .tag(PixelGrid.autoID)
                    Section("Widescreen 16:10 (fits MacBook Air)") {
                        ForEach(PixelGrid.widescreen) { g in
                            Text(g.title).tag(g.id)
                        }
                    }
                    Section("Square (Divoom style)") {
                        ForEach(PixelGrid.square) { g in
                            Text(g.title).tag(g.id)
                        }
                    }
                }
                if model.isAutoGrid {
                    HStack {
                        Text("Pixel size")
                        KeyboardSlider(value: $model.autoCell, range: 1...32, step: 1)
                        Text("\(Int(model.autoCell)) pt").monospacedDigit().frame(width: 44, alignment: .trailing)
                    }
                }
                Picker("Image fit", selection: $model.fit) {
                    ForEach(FitMode.allCases) { m in
                        Text(m.title).tag(m)
                    }
                }
                HStack {
                    Text("LED gap")
                    KeyboardSlider(value: $model.gap, range: 0...0.4, step: 0.01)
                    Text("\(Int((model.gap * 100).rounded()))%").monospacedDigit().frame(width: 44, alignment: .trailing)
                }
                Toggle("Crisp whole-number scaling", isOn: $model.integerScale)
                Toggle("Cross-fade transition", isOn: $model.fade)
                Toggle("Show time & date (moves each image)", isOn: $model.showClock)
                if !model.imageInfo.isEmpty {
                    Text(model.imageInfo).font(.caption).foregroundStyle(.secondary)
                }
            }

            Section("Slideshow") {
                HStack {
                    Text("Interval")
                    Spacer()
                    TextField("", value: $model.interval, format: .number)
                        .multilineTextAlignment(.trailing)
                        .frame(width: 60)
                    Text("sec")
                    Stepper("", value: $model.interval, in: 1...3600, step: 1)
                        .labelsHidden()
                }
                Toggle("Shuffle", isOn: $model.shuffle)
                HStack {
                    Spacer()
                    Button { model.previous() } label: { Image(systemName: "backward.fill") }
                    Button { model.playing.toggle() } label: {
                        Image(systemName: model.playing ? "pause.fill" : "play.fill")
                    }
                    Button { model.next() } label: { Image(systemName: "forward.fill") }
                    Spacer()
                }
                .buttonStyle(.bordered)
                .disabled(model.playlist.isEmpty)
            }

            Section {
                Text(model.cacheSummary).font(.caption).foregroundStyle(.secondary)
                if model.cacheIsStale {
                    Text("Settings or sources changed since the cache was created. Click Create Cache to apply them.")
                        .font(.caption).foregroundStyle(.orange)
                }
                if model.isCreatingCache {
                    ProgressView(value: model.cacheProgress)
                    Button("Cancel") { model.cancelCache() }
                } else {
                    Button("Create Cache") { model.createCache() }
                        .buttonStyle(.borderedProminent)
                        .disabled(model.sourcePaths.isEmpty)
                }
                if !model.cacheMessage.isEmpty && !model.isCreatingCache {
                    Text(model.cacheMessage).font(.caption).foregroundStyle(.secondary)
                }
            } header: {
                Text("Cache")
            } footer: {
                Text("Finish your resolution, fit, LED gap and scaling choices first, then create the cache. It deletes the old cache and renders every image once. The slideshow and wallpaper then use only the cache, never the original files.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Section {
                Picker("Screen", selection: $model.exportTargetID) {
                    ForEach(ExportTarget.available()) { t in
                        Text(t.title).tag(t.id)
                    }
                }
                if model.isExporting {
                    ProgressView(value: model.exportProgress)
                    Button("Cancel Export") { model.cancelExport() }
                } else {
                    Button("Export for Screen Saver…") { model.presentExportPanel() }
                        .disabled(model.sourcePaths.isEmpty)
                }
                if !model.exportMessage.isEmpty {
                    Text(model.exportMessage).font(.caption).foregroundStyle(.secondary)
                }
            } header: {
                Text("Screen saver export")
            } footer: {
                Text("Saves lossless PNGs at the screen's native pixel size with the current resolution, fit and LED gap baked in. Then in System Settings › Screen Saver choose a Photos/Folder style and pick the exported folder.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Section {
                Toggle("Change automatically every interval", isOn: $model.autoWallpaper)
                    .disabled(model.playlist.count < 2)
                Button("Set Current Image as Wallpaper") { model.applyWallpaper() }
                    .disabled(model.playlist.isEmpty)
                Button("Restore System Wallpaper") { model.removeWallpaper() }
                    .disabled(!model.wallpaperIsActive)
                if !model.wallpaperMessage.isEmpty {
                    Text(model.wallpaperMessage).font(.caption).foregroundStyle(.secondary)
                }
            } header: {
                Text("Wallpaper")
            } footer: {
                Text("Pixel Frame draws the wallpaper itself, from images already in RAM, so changing it never reads the disk. It sits behind your desktop icons on every screen, but the lock screen keeps the system wallpaper, and it disappears when the app quits. Uses the slideshow interval (minimum 5 s) and keeps running with the window closed, but not while the Mac or display sleeps or is locked. Click the menu-bar icon for the next image (right-click for more). Applies to the current Space on each screen.")
                    .font(.caption).foregroundStyle(.secondary)
            }

            Section("Power") {
                Toggle("Keep display awake while full screen", isOn: $model.keepAwake)
                Toggle("Quit app when Mac / display sleeps", isOn: $model.quitOnSleep)
                LabeledContent("Status", value: model.statusText)
                if model.reducedEffects {
                    Text("Low Power / thermal limit: fade off, slower preload.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }

            Section {
                Button {
                    model.enterPresentation()
                } label: {
                    Label("Show Full Screen", systemImage: "arrow.up.left.and.arrow.down.right")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
                .disabled(model.playlist.isEmpty)
            } footer: {
                Text("Esc exit · Space play/pause · ←/→ previous/next · double-click toggles full screen")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
        .onAppear { model.refreshMonitors() }
    }
}
