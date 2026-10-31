import AppKit
import SwiftUI
import UniformTypeIdentifiers

/// Reasons the slideshow must be completely idle (no timers, no decoding).
enum SuspendReason: String, CaseIterable {
    case systemSleep, displaySleep, screenLocked, sessionInactive, windowHidden

    var label: String {
        switch self {
        case .systemSleep: return "Mac asleep"
        case .displaySleep: return "display asleep"
        case .screenLocked: return "screen locked"
        case .sessionInactive: return "session inactive"
        case .windowHidden: return "window hidden"
        }
    }
}

// All mutation happens on the main thread (observers use queue: .main, background work hops back via @MainActor tasks).
final class AppModel: ObservableObject {
    static let shared = AppModel()      // shared with the menu-bar item
    private let defaults = UserDefaults.standard

    // MARK: Persisted settings

    @Published var gridID: String {
        didSet { defaults.set(gridID, forKey: "gridID"); schedulePreview() }
    }
    @Published var fit: FitMode {
        didSet { defaults.set(fit.rawValue, forKey: "fit"); schedulePreview() }
    }
    @Published var interval: Double {
        didSet {
            if interval < 1 { interval = 1 }
            if interval > 3600 { interval = 3600 }
            defaults.set(interval, forKey: "interval")
            rescheduleTimer()
            rescheduleWallpaperTimer()
        }
    }
    @Published var shuffle: Bool {
        didSet { defaults.set(shuffle, forKey: "shuffle"); rebuildPlaylist(keepCurrent: true) }
    }
    @Published var fade: Bool {
        didSet { defaults.set(fade, forKey: "fade") }
    }
    @Published var gap: Double {
        didSet { defaults.set(gap, forKey: "gap"); schedulePreview() }
    }
    @Published var integerScale: Bool {
        didSet { defaults.set(integerScale, forKey: "integerScale"); schedulePreview() }
    }
    @Published var autoCell: Double {
        didSet {
            defaults.set(autoCell, forKey: "autoCell")
            cellTask?.cancel()                           // debounce slider drags: re-render once, not per tick
            cellTask = Task { @MainActor [weak self] in
                try? await Task.sleep(for: .milliseconds(400))
                guard !Task.isCancelled else { return }
                self?.refreshAutoGrid()
            }
        }
    }
    @Published var autoWallpaper: Bool {
        didSet {
            defaults.set(autoWallpaper, forKey: "autoWallpaper")
            updateWallpaperActivity()
            rescheduleWallpaperTimer()
            if autoWallpaper { applyWallpaper() }          // start with the current image right away
            ramNeedsChanged()
        }
    }
    @Published var showClock: Bool {
        didSet { defaults.set(showClock, forKey: "showClock") }
    }
    @Published var keepAwake: Bool {
        didSet { defaults.set(keepAwake, forKey: "keepAwake"); updateAwakeAssertion() }
    }
    @Published var quitOnSleep: Bool {
        didSet { defaults.set(quitOnSleep, forKey: "quitOnSleep") }
    }

    // MARK: Runtime state

    @Published private(set) var sourcePaths: [String]
    @Published private(set) var playlist: [URL] = []
    @Published private(set) var index: Int = 0
    @Published private(set) var currentFrame: CGImage?
    @Published private(set) var currentItemID = 0         // bumps when a different image is shown
    @Published private(set) var clockSlot = 8             // 0...8 cell of a 3x3 layout (4 = centre, unused)
    @Published private(set) var autoGrid = PixelGrid.recommended
    /// Not @Published: an auto-change every few seconds must not re-evaluate the whole UI while nobody looks at it.
    private(set) var wallpaperMessage = "" {
        didSet {
            if wallpaperMessage != oldValue, NSApp?.windows.contains(where: { $0.canBecomeMain && $0.isVisible && $0.occlusionState.contains(.visible) }) == true {
                objectWillChange.send()
            }
        }
    }
    @Published private(set) var cachedCount: Int = 0
    @Published private(set) var manifest: CacheManifest?    // the cache on disk; the only source of images at runtime
    @Published private(set) var isCreatingCache = false
    @Published private(set) var cacheProgress = 0.0
    @Published private(set) var cacheMessage = ""
    @Published private(set) var previewFrame: CGImage?      // current image re-rendered with unsaved settings
    @Published private(set) var sourcesChanged = false      // a source folder changed after the cache was built
    @Published private(set) var suspendReasons: Set<SuspendReason> = []
    @Published private(set) var reducedEffects = false
    @Published private(set) var isPresenting = false
    @Published var hudVisible = false
    @Published private(set) var monitors: [ExternalMonitor] = []
    @Published private(set) var monitorMessage = ""
    @Published var playing = true {
        didSet { rescheduleTimer(); updateAwakeAssertion() }
    }

    private let cache = FrameCache()
    private var timer: Timer?
    private var cacheTask: Task<Void, Never>?
    private var trimTask: Task<Void, Never>?
    private var previewTask: Task<Void, Never>?
    private var cacheGeneration = 0                           // bumps whenever the RAM cache is reloaded from a new cache
    private var preloadTask: Task<Void, Never>?
    private var hudTask: Task<Void, Never>?
    private var lastPoke = Date.distantPast
    private var showToken = 0
    private var observers: [NSObjectProtocol] = []
    private var keyMonitor: Any?
    private var pressureSource: DispatchSourceMemoryPressure?
    private var awakeActivity: NSObjectProtocol?
    private var cellTask: Task<Void, Never>?
    private let desktop = DesktopWallpaper()
    private var wallpaperToken = 0
    private var wallpaperTimer: Timer?
    private var wallpaperIndex = 0
    private var wallpaperOrder: [URL] = []                    // its own order, so slideshow reshuffles don't skip or repeat
    private var wallpaperActivity: NSObjectProtocol?
    private var animTask: Task<Void, Never>?
    private var pressureRecovery: Task<Void, Never>?
    private var watcher: FolderWatcher?
    private var lastShownURL: URL?
    private var currentIsAnimated = false
    private var animLoopedOnce = true

    // MARK: Derived

    var isAutoGrid: Bool { gridID == PixelGrid.autoID }
    var grid: PixelGrid { isAutoGrid ? autoGrid : PixelGrid.find(gridID) }
    var config: RenderConfig { RenderConfig(width: grid.width, height: grid.height, fit: fit) }

    /// What is actually on screen comes from the cache, so its baked settings win until the cache is recreated.
    var shownGap: Double { previewFrame != nil ? gap : (manifest?.gap ?? gap) }
    var shownIntegerScale: Bool { previewFrame != nil ? integerScale : (manifest?.integerScale ?? integerScale) }
    var displayFrame: CGImage? { previewFrame ?? currentFrame }

    var imageInfo: String {
        let g = grid
        let ram = ByteCountFormatter.string(fromByteCount: Int64(g.width * g.height * 4), countStyle: .memory)
        return "Grid \(g.width) × \(g.height) px · \(ram) RAM per image"
    }

    /// Everything that gets baked into the cache. If it differs from the manifest, the cache is out of date.
    private var renderSettings: String {
        ["\(config.width)x\(config.height)", fit.rawValue, "\(gap)", "\(integerScale)"].joined(separator: ";")
    }

    /// Render settings, then the source list. A manifest signature starting with `renderSettings + ";"` was built with them.
    private var currentSignature: String {
        renderSettings + ";" + sourcePaths.joined(separator: "|")
    }

    private func cacheMatchesRenderSettings(_ m: CacheManifest) -> Bool {
        m.signature.hasPrefix(renderSettings + ";")
    }

    /// Auto-change keeps drawing from RAM while the window is hidden, so the RAM cache must stay filled then.
    private var wallpaperNeedsRAM: Bool { autoWallpaper && desktop.isActive }

    private var ramSuspended: Bool {
        !suspendReasons.subtracting(wallpaperNeedsRAM ? [.windowHidden] : []).isEmpty
    }

    var wallpaperIsActive: Bool { desktop.isActive }

    /// Whether the wallpaper still needs the RAM cache changed (auto-change on/off, wallpaper shown/removed): fill or trim it.
    private func ramNeedsChanged() {
        if ramSuspended {
            preloadTask?.cancel()
            scheduleTrim()
        } else {
            trimTask?.cancel()
            startPreload()
        }
    }

    var cacheIsStale: Bool {
        guard let m = manifest else { return false }
        return sourcesChanged || m.signature != currentSignature
    }

    var cacheSummary: String {
        guard let m = manifest else { return "No cache yet" }
        let size = ByteCountFormatter.string(fromByteCount: m.diskBytes, countStyle: .file)
        let when = m.created.formatted(date: .abbreviated, time: .shortened)
        return "\(m.items.count) images · \(size) · \(when)"
    }

    var ramSummary: String {
        let bytes = Int64(cachedCount * (manifest?.gridWidth ?? 0) * (manifest?.gridHeight ?? 0) * 4)
        let size = ByteCountFormatter.string(fromByteCount: bytes, countStyle: .memory)
        return "\(cachedCount) / \(playlist.count)  (\(size))"
    }

    var statusText: String {
        if playlist.isEmpty { return "Idle" }
        if !suspendReasons.isEmpty {
            let names = suspendReasons.map { $0.label }.sorted().joined(separator: ", ")
            return "Paused (\(names))"
        }
        return playing ? "Running" : "Paused"
    }

    private var shouldRun: Bool {
        playing && suspendReasons.isEmpty && playlist.count > 1
    }

    // MARK: Init

    init() {
        let d = UserDefaults.standard
        if !d.bool(forKey: "migratedAutoGrid") {          // one-time: switch existing installs to screen-fit
            d.set(true, forKey: "migratedAutoGrid")
            d.set(PixelGrid.autoID, forKey: "gridID")
        }
        gridID = d.string(forKey: "gridID") ?? PixelGrid.autoID
        fit = FitMode(rawValue: d.string(forKey: "fit") ?? "") ?? .fit
        let savedInterval = d.double(forKey: "interval")
        interval = savedInterval >= 1 ? savedInterval : 5          // default transition: 5 s
        shuffle = d.bool(forKey: "shuffle")
        fade = d.object(forKey: "fade") as? Bool ?? true
        gap = d.object(forKey: "gap") as? Double ?? 0.12
        integerScale = d.object(forKey: "integerScale") as? Bool ?? true
        let savedCell = d.double(forKey: "autoCell")
        autoCell = savedCell >= 1 ? savedCell : 12
        showClock = d.object(forKey: "showClock") as? Bool ?? true
        autoWallpaper = d.bool(forKey: "autoWallpaper")
        keepAwake = d.bool(forKey: "keepAwake")
        quitOnSleep = d.bool(forKey: "quitOnSleep")
        sourcePaths = d.stringArray(forKey: "sources") ?? []

        autoGrid = PixelGrid.auto(width: targetScreenSize().width, height: targetScreenSize().height, cell: autoCell)
        manifest = CacheStore.loadManifest()
        cache.maxCount = computeMaxCount()
        updateReducedEffects()
        installObservers()
        installKeyMonitor()
        installMemoryPressureHandler()
        if !sourcePaths.isEmpty { updateWatcher() }
        updateWallpaperActivity()
        if manifest != nil {
            rebuildPlaylist(keepCurrent: false)                          // load the cache folder into RAM
            if d.bool(forKey: "wallpaperSet") { applyWallpaper() }
        }
    }

    deinit {
        for o in observers { NotificationCenter.default.removeObserver(o) }
        if let m = keyMonitor { NSEvent.removeMonitor(m) }
        pressureSource?.cancel()
        timer?.invalidate()
        wallpaperTimer?.invalidate()
        animTask?.cancel()
    }

    // MARK: Sources

    func presentOpenPanel() {
        let panel = NSOpenPanel()
        panel.title = "Choose images or folders"
        panel.prompt = "Add"
        panel.allowsMultipleSelection = true
        panel.canChooseFiles = true
        panel.canChooseDirectories = true
        panel.allowedContentTypes = [.image]      // JPEG, PNG, HEIC, GIF, TIFF, BMP, WebP, RAW, ...
        if panel.runModal() == .OK {
            addSources(panel.urls)
        }
    }

    func addSources(_ urls: [URL]) {
        var merged = sourcePaths
        for u in urls where !merged.contains(u.path) { merged.append(u.path) }
        sourcePaths = merged
        defaults.set(sourcePaths, forKey: "sources")
        updateWatcher()                                   // takes effect at the next "Create Cache"
        schedulePreview()
    }

    func clearSources() {
        cacheTask?.cancel()
        preloadTask?.cancel()
        sourcePaths = []
        defaults.removeObject(forKey: "sources")
        CacheStore.delete()                               // the cache is derived data: no sources, no cache
        removeWallpaper()
        manifest = nil
        sourcesChanged = false
        cacheMessage = ""
        previewTask?.cancel()
        previewFrame = nil
        resetPlayback()
        updateWatcher()
        if isPresenting { exitPresentation() }
    }

    private func resetPlayback() {
        playlist = []
        wallpaperOrder = []
        wallpaperIndex = 0
        index = 0
        cacheGeneration += 1
        cache.removeAll()
        cachedCount = 0
        animTask?.cancel()
        currentFrame = nil
        lastShownURL = nil
        rescheduleTimer()
        rescheduleWallpaperTimer()
    }

    private func updateWatcher() {
        let dirs = sourcePaths.filter {
            var isDir: ObjCBool = false
            return FileManager.default.fileExists(atPath: $0, isDirectory: &isDir) && isDir.boolValue
        }
        watcher = dirs.isEmpty ? nil : FolderWatcher(paths: dirs) { [weak self] in self?.sourcesChanged = true }
    }

    /// The playlist is the cache folder's contents; the original files are never opened here.
    private func rebuildPlaylist(keepCurrent: Bool) {
        let currentURL: URL? = (keepCurrent && playlist.indices.contains(index)) ? playlist[index] : nil
        var list = (manifest?.items ?? []).map { CacheStore.gridURL(name: $0.name) }
        if shuffle { list.shuffle() }
        playlist = list
        if let cu = currentURL, let i = list.firstIndex(of: cu) {
            index = i
        } else {
            index = 0
        }
        wallpaperOrder = list
        wallpaperIndex = index
        rescheduleTimer()
        rescheduleWallpaperTimer()
        showCurrent()
        startPreload()
    }

    // MARK: Setting preview

    /// Unsaved settings are previewed on the current image only (rendered from its original, not stored anywhere).
    /// The slideshow and wallpaper keep using the cache until "Create Cache".
    private func schedulePreview() {
        previewTask?.cancel()
        previewTask = Task { @MainActor [weak self] in
            try? await Task.sleep(for: .milliseconds(400))
            guard let self = self, !Task.isCancelled, !self.isCreatingCache else { return }
            if let m = self.manifest, self.cacheMatchesRenderSettings(m) {
                self.previewFrame = nil                   // only the source list differs, or nothing: keep the slideshow running
                return
            }
            let cfg = self.config
            let paths = self.sourcePaths
            var original: URL?
            if let m = self.manifest, self.playlist.indices.contains(self.index) {
                let name = self.playlist[self.index].deletingPathExtension().lastPathComponent
                original = m.items.first { $0.name == name }?.path.map { URL(fileURLWithPath: $0) }
            }
            if self.manifest != nil, original == nil {            // cache from an older build: original unknown, don't guess
                self.previewFrame = nil
                return
            }
            let frame = await Task.detached(priority: .userInitiated) { () -> CGImage? in
                guard let url = original ?? ImageScanner.scan(paths: paths).first else { return nil }
                return PixelRenderer.render(url: url, config: cfg)
            }.value
            guard !Task.isCancelled, !self.isCreatingCache else { return }
            self.previewFrame = frame
        }
    }

    // MARK: Cache

    /// Deletes the old cache and bakes a new one from the sources with the current settings:
    /// a tiny pixel frame per image. Slideshow and wallpaper both draw those frames from RAM.
    func createCache() {
        guard !isCreatingCache, !sourcePaths.isEmpty else { return }
        let paths = sourcePaths
        let grid = config
        let gap = self.gap, integer = integerScale
        let signature = currentSignature

        isCreatingCache = true
        cacheProgress = 0
        cacheMessage = "Creating cache…"
        preloadTask?.cancel()
        trimTask?.cancel()
        rescheduleWallpaperTimer()                        // paused while the cache is rebuilt
        cacheTask = Task { @MainActor [weak self] in
            let work = Task.detached(priority: .utility) { () -> CacheManifest? in
                let urls = ImageScanner.scan(paths: paths)
                guard !urls.isEmpty else { return nil }
                return CacheStore.build(urls: urls, grid: grid, gap: gap, integerScale: integer,
                                        signature: signature) { fraction in
                    DispatchQueue.main.async { self?.cacheProgress = fraction }
                }
            }
            // Detached tasks don't inherit cancellation, so forward it explicitly.
            let result = await withTaskCancellationHandler { await work.value } onCancel: { work.cancel() }
            guard let self else { return }
            self.isCreatingCache = false
            if let result {
                self.sourcesChanged = false
                self.install(result)
                self.cacheMessage = "Cache created: \(self.cacheSummary)"
            } else {
                self.cacheMessage = Task.isCancelled
                    ? "Cancelled — the previous cache is unchanged"
                    : "Couldn't create the cache (no readable images?)"
                self.rescheduleWallpaperTimer()
                self.startPreload()
            }
        }
    }

    func cancelCache() { cacheTask?.cancel() }

    /// Swaps in a freshly built cache: RAM is cleared and refilled from the new folder, and the wallpaper is redrawn from it.
    private func install(_ m: CacheManifest) {
        previewTask?.cancel()
        previewFrame = nil                                // from now on the display shows the cache again
        manifest = m
        cacheGeneration += 1
        cache.removeAll()
        cache.maxCount = computeMaxCount()
        cachedCount = 0
        lastShownURL = nil
        rebuildPlaylist(keepCurrent: false)
        if desktop.isActive { applyWallpaper() }
    }

    // MARK: Display + RAM cache

    /// Ceiling for the RAM cache. It only fills as far as the library needs (38 images ≈ 5 MB at 240×150),
    /// but large libraries must fit entirely or least-recently-used eviction sends loads back to the disk.
    private func computeMaxCount() -> Int {
        let budget = min(UInt64(512) * 1024 * 1024, ProcessInfo.processInfo.physicalMemory / 32)
        let perFrame = max(1, (manifest?.gridWidth ?? 96) * (manifest?.gridHeight ?? 60) * 4)
        return max(16, Int(budget) / perFrame)
    }

    private func showCurrent() {
        showToken += 1
        let token = showToken
        guard playlist.indices.contains(index) else {
            animTask?.cancel()
            currentFrame = nil
            lastShownURL = nil
            return
        }
        let url = playlist[index]
        if let set = cache.get(url) {                 // RAM hit: no disk access
            display(set, url: url)
            return
        }
        let gen = cacheGeneration
        Task { @MainActor [weak self] in
            let set = await Task.detached(priority: .userInitiated) { CacheStore.loadFrames(url) }.value
            guard let self = self, token == self.showToken, gen == self.cacheGeneration else { return }
            if let set {
                self.cache.insert(set, for: url)
                self.cachedCount = self.cache.count
                self.display(set, url: url)
            } else if !self.isCreatingCache {             // a cache swap in progress can fail a read: not a bad image
                self.dropBadImage(url)                // cache file missing or unreadable
            }
        }
    }

    private func display(_ set: FrameSet, url: URL) {
        animTask?.cancel()
        animTask = nil
        if url != lastShownURL {                      // new picture: new fade + new clock corner
            lastShownURL = url
            currentItemID += 1
            moveClock()
        }
        currentFrame = set.first
        currentIsAnimated = set.isAnimated
        animLoopedOnce = !set.isAnimated || set.totalDuration > 60
        if set.isAnimated, suspendReasons.isEmpty { startAnimation(set) }
    }

    private func startAnimation(_ set: FrameSet) {
        animTask = Task { @MainActor [weak self] in
            var i = 0
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(set.delays[i]))
                guard let self = self, !Task.isCancelled else { return }
                i = (i + 1) % set.images.count
                if i == 0 { self.animLoopedOnce = true }
                self.currentFrame = set.images[i]
            }
        }
    }

    /// Moves the clock to a different edge/corner cell than the one it was in.
    private func moveClock() {
        let slots = [0, 1, 2, 3, 5, 6, 7, 8].filter { $0 != clockSlot }
        clockSlot = slots.randomElement() ?? 8
    }

    private func dropBadImage(_ url: URL) {
        wallpaperOrder.removeAll { $0 == url }
        if wallpaperIndex >= wallpaperOrder.count { wallpaperIndex = 0 }
        playlist.removeAll { $0 == url }
        if index >= playlist.count { index = 0 }
        showCurrent()
    }

    /// Loads the whole cache folder into RAM once, serially at utility QoS, starting at the current image.
    private func startPreload() {
        preloadTask?.cancel()
        guard !playlist.isEmpty, !ramSuspended else { return }   // nothing to show = nothing to load
        let gen = cacheGeneration
        let start = min(index, playlist.count - 1)
        let ordered = Array(playlist[start...]) + Array(playlist[..<start])

        preloadTask = Task { @MainActor [weak self] in
            var done = 0
            for url in ordered {
                guard let self = self, !Task.isCancelled else { return }
                if self.cache.contains(url) { continue }
                if self.cache.isFull { break }
                let set = await Task.detached(priority: .utility) { CacheStore.loadFrames(url) }.value
                guard !Task.isCancelled, gen == self.cacheGeneration else { return }
                if let set { self.cache.insert(set, for: url) }
                done += 1
                if done % 16 == 0 { self.cachedCount = self.cache.count }
                let pauseMs = self.reducedEffects ? 100 : 2
                try? await Task.sleep(for: .milliseconds(pauseMs))
            }
            self?.cachedCount = self?.cache.count ?? 0
        }
    }

    /// Hidden / asleep / locked for a minute: nothing is on screen, so give the RAM cache back (unless auto-change wallpaper is drawing from it).
    private func scheduleTrim() {
        trimTask?.cancel()
        trimTask = Task { @MainActor [weak self] in
            try? await Task.sleep(for: .seconds(60))
            guard let self = self, !Task.isCancelled, self.ramSuspended else { return }
            let keep: URL? = self.playlist.indices.contains(self.index) ? self.playlist[self.index] : nil
            self.cache.removeAll(keeping: keep)
            self.cachedCount = self.cache.count
            Self.returnMemoryToSystem()
        }
    }

    /// Hands freed malloc pages back to the OS so the resident footprint matches what is really in use.
    static func returnMemoryToSystem() {
        malloc_zone_pressure_relief(nil, 0)
    }

    private func handleMemoryPressure() {
        preloadTask?.cancel()
        let keep: URL? = playlist.indices.contains(index) ? playlist[index] : nil
        cache.removeAll(keeping: keep)
        cache.maxCount = 32
        cachedCount = cache.count
        Self.returnMemoryToSystem()
        pressureRecovery?.cancel()                        // restore the normal budget once things calm down
        pressureRecovery = Task { @MainActor [weak self] in
            try? await Task.sleep(for: .seconds(120))
            guard let self = self, !Task.isCancelled else { return }
            self.cache.maxCount = self.computeMaxCount()
            self.startPreload()
        }
    }

    // MARK: Slideshow

    func next() {
        guard !playlist.isEmpty else { return }
        advanceIndex()
        showCurrent()
        rescheduleTimer()
    }

    func previous() {
        guard !playlist.isEmpty else { return }
        index = (index - 1 + playlist.count) % playlist.count
        showCurrent()
        rescheduleTimer()
    }

    private func advanceFromTimer() {
        guard shouldRun else { return }
        if currentIsAnimated && !animLoopedOnce { return }   // let a GIF finish its loop first
        advanceIndex()
        showCurrent()
    }

    /// Steps forward; with shuffle on, a fresh random order starts every lap (never repeating the last image first).
    private func advanceIndex() {
        let n = playlist.count
        let nextIndex = (index + 1) % n
        if nextIndex == 0, shuffle, n > 2 {
            let last = playlist[index]
            playlist.shuffle()
            if playlist[0] == last { playlist.swapAt(0, n - 1) }
        }
        index = nextIndex
    }

    private func rescheduleTimer() {
        timer?.invalidate()
        timer = nil
        guard shouldRun else { return }                // no timer at all while idle = zero wakeups
        let t = Timer(timeInterval: interval, repeats: true) { [weak self] _ in
            self?.advanceFromTimer()
        }
        t.tolerance = min(interval * 0.25, 2.0)        // lets macOS coalesce wakeups
        RunLoop.main.add(t, forMode: .common)
        timer = t
    }

    // MARK: Sleep / lock / hidden detection

    private func setSuspend(_ reason: SuspendReason, _ on: Bool) {
        let wasIdle = suspendReasons.isEmpty
        let before = suspendReasons
        defer { if suspendReasons != before { rescheduleWallpaperTimer() } }
        if on { suspendReasons.insert(reason) } else { suspendReasons.remove(reason) }
        if wasIdle != suspendReasons.isEmpty {
            rescheduleTimer()
            updateAwakeAssertion()
            if suspendReasons.isEmpty {
                trimTask?.cancel()
                showCurrent()                              // restarts any animation
                startPreload()                             // refill RAM from the cache folder if it was trimmed
            } else {
                animTask?.cancel()
                if ramSuspended { preloadTask?.cancel(); scheduleTrim() }
            }
        }
    }

    private func observe(_ name: Notification.Name,
                         center: NotificationCenter = NotificationCenter.default,
                         _ handler: @escaping (Notification) -> Void) {
        let token = center.addObserver(forName: name, object: nil, queue: .main, using: handler)
        observers.append(token)
    }

    private func installObservers() {
        let ws = NSWorkspace.shared.notificationCenter

        observe(NSWorkspace.willSleepNotification, center: ws) { [weak self] _ in
            guard let self = self else { return }
            if self.quitOnSleep { NSApp.terminate(nil); return }
            self.setSuspend(.systemSleep, true)
        }
        observe(NSWorkspace.didWakeNotification, center: ws) { [weak self] _ in
            self?.setSuspend(.systemSleep, false)
        }
        observe(NSWorkspace.screensDidSleepNotification, center: ws) { [weak self] _ in
            guard let self = self else { return }
            if self.quitOnSleep { NSApp.terminate(nil); return }
            self.setSuspend(.displaySleep, true)
        }
        observe(NSWorkspace.screensDidWakeNotification, center: ws) { [weak self] _ in
            self?.setSuspend(.displaySleep, false)
        }
        observe(NSWorkspace.sessionDidResignActiveNotification, center: ws) { [weak self] _ in
            self?.setSuspend(.sessionInactive, true)
        }
        observe(NSWorkspace.sessionDidBecomeActiveNotification, center: ws) { [weak self] _ in
            self?.setSuspend(.sessionInactive, false)
        }

        let dist: NotificationCenter = DistributedNotificationCenter.default()
        observe(Notification.Name("com.apple.screenIsLocked"), center: dist) { [weak self] _ in
            self?.setSuspend(.screenLocked, true)
        }
        observe(Notification.Name("com.apple.screenIsUnlocked"), center: dist) { [weak self] _ in
            self?.setSuspend(.screenLocked, false)
        }

        // Minimised, fully covered, or on another Space -> nothing to show -> stop working.
        observe(NSWindow.didChangeOcclusionStateNotification) { [weak self] note in
            guard let window = note.object as? NSWindow, window.canBecomeMain else { return }
            self?.setSuspend(.windowHidden, !window.occlusionState.contains(.visible))
        }

        observe(NSApplication.didChangeScreenParametersNotification) { [weak self] _ in
            self?.refreshAutoGrid()
            self?.desktop.refreshScreens()
            self?.refreshMonitors()
        }
        observe(NSWindow.didChangeScreenNotification) { [weak self] _ in
            self?.refreshAutoGrid()
        }

        observe(NSWindow.didEnterFullScreenNotification) { [weak self] _ in
            guard let self = self else { return }
            self.isPresenting = true
            self.refreshAutoGrid()
            self.pokeHUD()
            self.updateAwakeAssertion()
        }
        observe(NSWindow.didExitFullScreenNotification) { [weak self] _ in
            guard let self = self else { return }
            self.isPresenting = false
            self.hudTask?.cancel()
            self.hudVisible = false
            self.updateAwakeAssertion()
        }

        observe(Notification.Name.NSProcessInfoPowerStateDidChange) { [weak self] _ in
            self?.updateReducedEffects()
        }
        observe(ProcessInfo.thermalStateDidChangeNotification) { [weak self] _ in
            self?.updateReducedEffects()
        }
    }

    // MARK: Wallpaper

    /// Wallpaper has its own position in the playlist, so it keeps cycling even when the slideshow window is closed.
    @discardableResult
    func nextWallpaper() -> Bool {
        guard !wallpaperOrder.isEmpty else { return false }
        wallpaperIndex += 1
        if wallpaperIndex >= wallpaperOrder.count {            // new lap: with shuffle, a fresh order that doesn't start with the last image
            wallpaperIndex = 0
            let n = wallpaperOrder.count
            if shuffle, n > 2 {
                let last = wallpaperOrder[n - 1]
                wallpaperOrder.shuffle()
                if wallpaperOrder[0] == last { wallpaperOrder.swapAt(0, n - 1) }
            }
        }
        return setWallpaper(url: wallpaperOrder[wallpaperIndex])
    }

    /// Runs while auto-change is on, even with the window closed; stops for sleep, display sleep, lock and logout.
    private func rescheduleWallpaperTimer() {
        wallpaperTimer?.invalidate()
        wallpaperTimer = nil
        let blockers = suspendReasons.subtracting([.windowHidden])
        guard autoWallpaper, playlist.count > 1, blockers.isEmpty, !isCreatingCache else { return }
        let every = max(5, interval)
        let t = Timer(timeInterval: every, repeats: true) { [weak self] _ in self?.nextWallpaper() }
        t.tolerance = min(every * 0.25, 5)
        RunLoop.main.add(t, forMode: .common)
        wallpaperTimer = t
    }

    /// Keeps macOS App Nap from throttling the timer while the window is hidden (system sleep still allowed).
    private func updateWallpaperActivity() {
        if autoWallpaper, wallpaperActivity == nil {
            wallpaperActivity = ProcessInfo.processInfo.beginActivity(
                options: .userInitiatedAllowingIdleSystemSleep, reason: "Auto-change wallpaper")
        } else if !autoWallpaper, let a = wallpaperActivity {
            ProcessInfo.processInfo.endActivity(a)
            wallpaperActivity = nil
        }
    }

    /// Draws the current image as wallpaper on every screen.
    @discardableResult
    func applyWallpaper() -> Bool {
        guard playlist.indices.contains(index) else {
            wallpaperMessage = "Create the cache first"
            return false
        }
        wallpaperIndex = wallpaperOrder.firstIndex(of: playlist[index]) ?? 0
        return setWallpaper(url: playlist[index])
    }

    /// Shows the frame from the RAM cache on the desktop windows; the disk is only read if it isn't in RAM yet.
    private func setWallpaper(url: URL) -> Bool {
        guard manifest != nil, !isCreatingCache else {
            wallpaperMessage = isCreatingCache ? "Wait for the cache to finish" : "Create the cache first"
            return false
        }
        if let set = cache.get(url) {
            showWallpaper(set.first, url: url)
            return true
        }
        wallpaperToken += 1
        let token = wallpaperToken, gen = cacheGeneration
        Task { @MainActor [weak self] in
            let set = await Task.detached(priority: .userInitiated) { CacheStore.loadFrames(url) }.value
            guard let self = self, token == self.wallpaperToken, gen == self.cacheGeneration, let set else { return }
            self.cache.insert(set, for: url)
            self.showWallpaper(set.first, url: url)
        }
        return true
    }

    private func showWallpaper(_ frame: CGImage, url: URL) {
        let wasActive = desktop.isActive
        defer { if !wasActive { ramNeedsChanged() } }
        desktop.show(frame, gap: manifest?.gap ?? gap, integerScale: manifest?.integerScale ?? integerScale)
        if !defaults.bool(forKey: "wallpaperSet") { defaults.set(true, forKey: "wallpaperSet") }
        let name = url.deletingPathExtension().lastPathComponent
        wallpaperMessage = "Wallpaper set: " + (manifest?.items.first { $0.name == name }?.source ?? name)
    }

    /// Takes the desktop windows away so the system wallpaper shows again.
    func removeWallpaper() {
        wallpaperToken += 1
        desktop.hide()
        defaults.set(false, forKey: "wallpaperSet")
        wallpaperMessage = ""
        ramNeedsChanged()
    }

    // MARK: External monitors

    func refreshMonitors() { monitors = MonitorControl.list() }

    func setMonitor(_ monitor: ExternalMonitor, on: Bool) {
        monitorMessage = MonitorControl.set(monitor, on: on) ?? ""
        // The display list settles a moment after the reconfiguration.
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { [weak self] in self?.refreshMonitors() }
        refreshMonitors()
    }

    // MARK: Screen-fit ("Auto") grid

    /// Size in points of the area the full-screen slideshow will occupy (below the notch on notched Macs).
    private func targetScreenSize() -> CGSize {
        let screen = NSApp?.windows.first(where: { $0.canBecomeMain && $0.isVisible })?.screen
            ?? NSScreen.main ?? NSScreen.screens.first
        guard let s = screen else { return CGSize(width: 1440, height: 900) }
        return CGSize(width: s.frame.width, height: s.frame.height - s.safeAreaInsets.top)
    }

    /// Re-derives the grid from the current screen. No-op unless the result actually changed.
    func refreshAutoGrid() {
        let size = targetScreenSize()
        let g = PixelGrid.auto(width: size.width, height: size.height, cell: autoCell)
        guard g != autoGrid else { return }
        autoGrid = g
        schedulePreview()
    }

    private func updateReducedEffects() {
        let info = ProcessInfo.processInfo
        reducedEffects = info.isLowPowerModeEnabled
            || info.thermalState == .serious
            || info.thermalState == .critical
    }

    private func installMemoryPressureHandler() {
        let src = DispatchSource.makeMemoryPressureSource(eventMask: [.warning, .critical], queue: .main)
        src.setEventHandler { [weak self] in
            self?.handleMemoryPressure()
        }
        src.resume()
        pressureSource = src
    }

    private func updateAwakeAssertion() {
        let want = keepAwake && isPresenting && playing
        if want, awakeActivity == nil {
            awakeActivity = ProcessInfo.processInfo.beginActivity(
                options: [.idleDisplaySleepDisabled],
                reason: "Pixel display slideshow"
            )
        } else if !want, let activity = awakeActivity {
            ProcessInfo.processInfo.endActivity(activity)
            awakeActivity = nil
        }
    }

    // MARK: Screen saver export

    @Published var exportTargetID: String = ExportTarget.available().first?.id ?? ""
    @Published private(set) var isExporting = false
    @Published private(set) var exportProgress = 0.0
    @Published private(set) var exportMessage = ""
    private var exportTask: Task<Void, Never>?

    var exportTarget: ExportTarget? {
        let all = ExportTarget.available()
        return all.first { $0.id == exportTargetID } ?? all.first
    }

    func presentExportPanel() {
        guard !sourcePaths.isEmpty, let target = exportTarget else { return }
        let panel = NSOpenPanel()
        panel.title = "Choose where to save the screen saver images"
        let ecfg = exportConfig(for: target)
        panel.message = "A new folder \"PixelFrame \(ecfg.width)x\(ecfg.height)\" is created inside the folder you pick."
        panel.prompt = "Export Here"
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.canCreateDirectories = true
        guard panel.runModal() == .OK, let parent = panel.url else { return }
        export(into: parent, target: target)
    }

    /// Auto mode sizes the grid for the chosen export screen (in its native pixels), not the current window.
    private func exportConfig(for target: ExportTarget) -> RenderConfig {
        guard isAutoGrid else { return config }
        let g = PixelGrid.auto(width: CGFloat(target.width), height: CGFloat(target.height), cell: autoCell * target.scale)
        return RenderConfig(width: g.width, height: g.height, fit: fit)
    }

    private func export(into parent: URL, target: ExportTarget) {
        let paths = sourcePaths
        let cfg = exportConfig(for: target)
        let gap = self.gap
        let integer = integerScale
        let size = (width: target.width, height: target.height)
        let folder = parent.appendingPathComponent("PixelFrame \(cfg.width)x\(cfg.height)", isDirectory: true)

        exportTask?.cancel()
        isExporting = true
        exportProgress = 0
        exportMessage = "Exporting…"
        exportTask = Task { @MainActor [weak self] in
            let work = Task.detached(priority: .utility) { () -> (done: Int, failed: Int) in
                let urls = ImageScanner.scan(paths: paths)
                try? FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
                var done = 0, failed = 0
                let digits = max(4, String(urls.count).count)
                for (i, url) in urls.enumerated() {
                    if Task.isCancelled { break }
                    let ok: Bool = autoreleasepool {
                        guard let frame = PixelRenderer.render(url: url, config: cfg),
                              let full = PixelExporter.compose(frame: frame, size: size, gap: gap, integerScale: integer)
                        else { return false }
                        let base = url.deletingPathExtension().lastPathComponent
                        let number = String(format: "%0\(digits)d", i + 1)
                        return PixelExporter.writePNG(full, to: folder.appendingPathComponent("\(number) \(base).png"))
                    }
                    if ok { done += 1 } else { failed += 1 }
                    let fraction = Double(i + 1) / Double(urls.count)
                    await MainActor.run { self?.exportProgress = fraction }
                }
                return (done, failed)
            }
            // Detached tasks don't inherit cancellation, so forward it explicitly.
            let result = await withTaskCancellationHandler { await work.value } onCancel: { work.cancel() }
            guard let self = self else { return }
            self.isExporting = false
            let wasCancelled = Task.isCancelled
            self.exportMessage = wasCancelled
                ? "Export cancelled (\(result.done) saved)"
                : "Saved \(result.done) image\(result.done == 1 ? "" : "s")" + (result.failed > 0 ? ", \(result.failed) skipped" : "")
            if result.done > 0 { NSWorkspace.shared.activateFileViewerSelecting([folder]) }
        }
    }

    func cancelExport() { exportTask?.cancel() }

    // MARK: Full-screen presentation

    private var mainWindow: NSWindow? {
        NSApp.windows.first { $0.canBecomeMain && $0.isVisible } ?? NSApp.keyWindow
    }

    func enterPresentation() {
        guard !playlist.isEmpty, let w = mainWindow else { return }
        if !w.styleMask.contains(.fullScreen) { w.toggleFullScreen(nil) }
    }

    func exitPresentation() {
        guard let w = mainWindow else { return }
        if w.styleMask.contains(.fullScreen) { w.toggleFullScreen(nil) }
    }

    func pokeHUD() {
        let now = Date()
        if hudVisible && now.timeIntervalSince(lastPoke) < 0.3 { return }
        lastPoke = now
        if !hudVisible { hudVisible = true }
        hudTask?.cancel()
        hudTask = Task { @MainActor [weak self] in
            try? await Task.sleep(for: .seconds(3))
            guard let self = self, !Task.isCancelled else { return }
            self.hudVisible = false
            NSCursor.setHiddenUntilMouseMoves(true)
        }
    }

    private func installKeyMonitor() {
        keyMonitor = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { [weak self] event in
            guard let self = self else { return event }
            return self.handleKey(event)
        }
    }

    private func handleKey(_ event: NSEvent) -> NSEvent? {
        guard isPresenting else { return event }
        switch event.keyCode {
        case 53: exitPresentation(); return nil        // esc
        case 49: playing.toggle(); return nil          // space
        case 123: previous(); return nil               // left
        case 124: next(); return nil                   // right
        default: return event
        }
    }
}
