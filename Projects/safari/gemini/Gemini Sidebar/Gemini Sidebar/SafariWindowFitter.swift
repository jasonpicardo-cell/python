import AppKit

/// Shrinks Safari's front window so it sits beside the sidebar instead of under it,
/// and puts it back when the sidebar closes. Uses AppleScript (asks for Automation permission once).
final class SafariWindowFitter {
    private struct Saved { let windowID: Int; let original: [Int]; let applied: [Int] }
    private var saved: Saved?
    private static let minSafariWidth = 600

    func makeRoom(for sidebar: NSRect, side: SidebarController.Side) {
        guard isSafariRunning, let primary = NSScreen.screens.first else { return }
        guard let info = run("""
            tell application "Safari"
                if (count of windows) is 0 then return {}
                set w to front window
                return {id of w} & (bounds of w)
            end tell
            """), info.numberOfItems == 5 else { return }
        let values = (1...5).map { Int(info.atIndex($0)?.int32Value ?? 0) }
        let windowID = values[0]
        var bounds = Array(values[1...]) // {left, top, right, bottom}, origin at top-left of the primary screen

        // Sidebar frame in AppleScript coordinates.
        let sbLeft = Int(sidebar.minX), sbRight = Int(sidebar.maxX)
        let sbTop = Int(primary.frame.height - sidebar.maxY), sbBottom = Int(primary.frame.height - sidebar.minY)
        let overlaps = bounds[0] < sbRight && bounds[2] > sbLeft && bounds[1] < sbBottom && bounds[3] > sbTop
        guard overlaps else { return }

        let original = saved?.windowID == windowID ? saved!.original : bounds
        if side == .right {
            bounds[2] = sbLeft
            if bounds[2] - bounds[0] < Self.minSafariWidth { bounds[0] = max(sbLeft - Self.minSafariWidth, 0) }
        } else {
            bounds[0] = sbRight
            if bounds[2] - bounds[0] < Self.minSafariWidth { bounds[2] = sbRight + Self.minSafariWidth }
        }
        if setBounds(windowID: windowID, bounds) {
            saved = Saved(windowID: windowID, original: original, applied: bounds)
        }
    }

    func restore() {
        guard let saved, isSafariRunning else { self.saved = nil; return }
        self.saved = nil
        // Only undo our change if the user hasn't moved or resized the window since.
        guard let info = run("tell application \"Safari\" to return bounds of (first window whose id is \(saved.windowID))"),
              info.numberOfItems == 4 else { return }
        let current = (1...4).map { Int(info.atIndex($0)?.int32Value ?? 0) }
        guard current == saved.applied else { return }
        _ = setBounds(windowID: saved.windowID, saved.original)
    }

    private var isSafariRunning: Bool {
        !NSRunningApplication.runningApplications(withBundleIdentifier: "com.apple.Safari").isEmpty
    }

    private func setBounds(windowID: Int, _ b: [Int]) -> Bool {
        run("tell application \"Safari\" to set bounds of (first window whose id is \(windowID)) to {\(b[0]), \(b[1]), \(b[2]), \(b[3])}") != nil
    }

    @discardableResult
    private func run(_ source: String) -> NSAppleEventDescriptor? {
        var error: NSDictionary?
        let result = NSAppleScript(source: source)?.executeAndReturnError(&error)
        if let error { NSLog("%@", "Gemini Sidebar: AppleScript failed: \(error)"); return nil }
        return result
    }
}
