import AppKit
import CoreGraphics

/// Maps normalized panel coordinates to global screen points on the M14t display,
/// honoring display rotation and re-resolving when the display layout changes.
final class DisplayMapper {
    private let hint: String?
    private let rotationOverride: Int?
    private(set) var displayID: CGDirectDisplayID = 0
    private var bounds = CGRect.zero
    private var rotation = 0
    private var dirty = true

    init(hint: String?, rotationOverride: Int?) {
        self.hint = hint
        self.rotationOverride = rotationOverride
        let ctx = Unmanaged.passUnretained(self).toOpaque()
        CGDisplayRegisterReconfigurationCallback({ _, flags, ctx in
            guard let ctx, !flags.contains(.beginConfigurationFlag) else { return }
            Unmanaged<DisplayMapper>.fromOpaque(ctx).takeUnretainedValue().dirty = true
        }, ctx)
        refresh()
    }

    static func onlineDisplays() -> [CGDirectDisplayID] {
        var ids = [CGDirectDisplayID](repeating: 0, count: 32)
        var n: UInt32 = 0
        CGGetOnlineDisplayList(32, &ids, &n)
        return Array(ids.prefix(Int(n)))
    }

    static func name(of id: CGDirectDisplayID) -> String {
        NSScreen.screens.first {
            ($0.deviceDescription[NSDeviceDescriptionKey("NSScreenNumber")] as? NSNumber)?.uint32Value == id
        }?.localizedName ?? "Display \(id)"
    }

    static func describeAll() -> String {
        onlineDisplays().map { id in
            String(format: "  id=%u  %@  vendor=0x%04x model=0x%04x  bounds=%@  rotation=%d",
                   id, name(of: id), CGDisplayVendorNumber(id), CGDisplayModelNumber(id),
                   NSStringFromRect(CGDisplayBounds(id)), Int(CGDisplayRotation(id)))
        }.joined(separator: "\n")
    }

    private func pick() -> CGDirectDisplayID {
        let ids = Self.onlineDisplays()
        if let hint {
            if let n = UInt32(hint), ids.contains(n) { return n }
            if let id = ids.first(where: { Self.name(of: $0).localizedCaseInsensitiveContains(hint) }) { return id }
            log("display '\(hint)' not found; auto-detecting")
        }
        return ids.first { CGDisplayVendorNumber($0) == M14t.displayVendor && CGDisplayModelNumber($0) == M14t.displayModel }
            ?? ids.first { Self.name(of: $0).localizedCaseInsensitiveContains("M14t") }
            ?? ids.first { CGDisplayVendorNumber($0) == M14t.displayVendor }
            ?? CGMainDisplayID()
    }

    func refresh() {
        let id = pick()
        let b = CGDisplayBounds(id)
        let r = rotationOverride ?? Int(CGDisplayRotation(id))
        if id != displayID || b != bounds || r != rotation {
            log("mapping touch to \(Self.name(of: id)) (id \(id)) bounds=\(NSStringFromRect(b)) rotation=\(r)")
        }
        displayID = id; bounds = b; rotation = r; dirty = false
    }

    func map(u: Double, v: Double) -> CGPoint {
        if dirty { refresh() }
        let (x, y): (Double, Double)
        switch rotation {
        case 90: (x, y) = (v, 1 - u)
        case 180: (x, y) = (1 - u, 1 - v)
        case 270: (x, y) = (1 - v, u)
        default: (x, y) = (u, v)
        }
        return CGPoint(x: bounds.minX + x * (bounds.width - 1),
                       y: bounds.minY + y * (bounds.height - 1))
    }
}
