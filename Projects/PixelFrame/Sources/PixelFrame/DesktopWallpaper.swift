import AppKit

/// One borderless, click-through window per screen at desktop level (below the icons), sized to the screen.
private final class DesktopWindow: NSWindow {
    let pixelView = PixelNSView(frame: .zero)

    init(screen: NSScreen) {
        super.init(contentRect: screen.frame, styleMask: .borderless, backing: .buffered, defer: false)
        level = NSWindow.Level(rawValue: Int(CGWindowLevelForKey(.desktopWindow)))
        collectionBehavior = [.canJoinAllSpaces, .stationary, .ignoresCycle, .fullScreenNone]
        isOpaque = true
        backgroundColor = .black
        hasShadow = false
        ignoresMouseEvents = true
        isReleasedWhenClosed = false
        pixelView.autoresizingMask = [.width, .height]
        contentView = pixelView
        setFrame(screen.frame, display: false)
    }
}

/// Draws the wallpaper itself from a frame that is already in RAM, so changing it never touches the disk.
/// (macOS's own wallpaper API only accepts files, and re-reads the file on every change.)
final class DesktopWallpaper {
    private var windows: [DesktopWindow] = []
    private var image: CGImage?
    private var gap = 0.0
    private var integerScale = true
    private var itemID = 0

    var isActive: Bool { !windows.isEmpty }

    func show(_ image: CGImage, gap: Double, integerScale: Bool) {
        self.image = image
        self.gap = gap
        self.integerScale = integerScale
        itemID += 1
        if windows.count != NSScreen.screens.count { rebuild() }
        push()
    }

    /// Screens were added, removed or rearranged.
    func refreshScreens() {
        guard isActive else { return }
        rebuild()
        push()
    }

    /// Removes the windows; the system wallpaper shows again.
    func hide() {
        windows.forEach { $0.close() }
        windows = []
        image = nil
    }

    private func rebuild() {
        windows.forEach { $0.close() }
        windows = NSScreen.screens.map { DesktopWindow(screen: $0) }
        windows.forEach { $0.orderFrontRegardless() }
    }

    private func push() {
        for w in windows {
            w.pixelView.update(image: image, itemID: itemID, gap: gap, integerScale: integerScale, fade: false)
        }
    }
}
