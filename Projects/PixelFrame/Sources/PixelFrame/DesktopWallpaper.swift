import AppKit
import SwiftUI

/// One borderless, click-through window per screen at desktop level (below the icons), sized to the screen.
private final class DesktopWindow: NSWindow {
    let pixelView = PixelNSView(frame: .zero)

    init(screen: NSScreen, model: AppModel?) {
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
        if let model {                                   // time & date drawn over the wallpaper, click-through
            let clock = NSHostingView(rootView: WallpaperClock().environmentObject(model))
            clock.frame = pixelView.bounds
            clock.autoresizingMask = [.width, .height]
            pixelView.addSubview(clock)
        }
        setFrame(screen.frame, display: false)
    }
}

/// The same time & date overlay as the slideshow, shown only when "Show time & date" is on.
private struct WallpaperClock: View {
    @EnvironmentObject var model: AppModel
    var body: some View {
        if model.showClock { ClockOverlay(fixedLine: true) }
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
    weak var model: AppModel?

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
        windows = NSScreen.screens.map { DesktopWindow(screen: $0, model: model) }
        windows.forEach { $0.orderFrontRegardless() }
    }

    private func push() {
        for w in windows {
            w.pixelView.update(image: image, itemID: itemID, gap: gap, integerScale: integerScale, fade: false)
        }
    }
}
