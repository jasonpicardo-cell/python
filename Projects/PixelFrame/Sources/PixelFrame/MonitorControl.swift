import AppKit
import CoreGraphics

// Private CoreGraphics call (same one System Settings / display utilities use) that disconnects or reconnects a display.
@_silgen_name("CGSConfigureDisplayEnabled")
private func CGSConfigureDisplayEnabled(_ config: CGDisplayConfigRef?, _ display: CGDirectDisplayID, _ enabled: Bool) -> CGError

struct ExternalMonitor: Identifiable, Equatable {
    let id: String              // display UUID: stable across reconnects
    let displayID: CGDirectDisplayID
    let name: String
    let connection: String
    var isOn: Bool
}

/// Lists external (Thunderbolt / HDMI / DisplayPort) monitors and switches them off or on.
/// A switched-off monitor is disconnected from macOS (loses signal and goes to standby) until re-enabled.
enum MonitorControl {
    private static let namesKey = "monitorNames"
    private static let offKey = "monitorsTurnedOff"     // uuid -> display ID, for monitors we switched off (they leave the online list)

    private static var turnedOff: [String: UInt32] {
        get { (UserDefaults.standard.dictionary(forKey: offKey) as? [String: UInt32]) ?? [:] }
        set { UserDefaults.standard.set(newValue, forKey: offKey) }
    }

    static func list() -> [ExternalMonitor] {
        var count: UInt32 = 0
        CGGetOnlineDisplayList(0, nil, &count)
        var ids = [CGDirectDisplayID](repeating: 0, count: Int(count))
        CGGetOnlineDisplayList(count, &ids, &count)

        var names = UserDefaults.standard.dictionary(forKey: namesKey) as? [String: String] ?? [:]
        var result: [ExternalMonitor] = []
        for did in ids.prefix(Int(count)) where CGDisplayIsBuiltin(did) == 0 {
            guard let uuid = CGDisplayCreateUUIDFromDisplayID(did)?.takeRetainedValue(),
                  let key = CFUUIDCreateString(nil, uuid) as String? else { continue }
            let active = CGDisplayIsActive(did) != 0
            if let screen = NSScreen.screens.first(where: { $0.displayID == did }) {
                names[key] = screen.localizedName
            }
            let name = names[key] ?? "External display \(did)"
            result.append(ExternalMonitor(id: key, displayID: did, name: name,
                                          connection: "External", isOn: active))
        }
        UserDefaults.standard.set(names, forKey: namesKey)

        // Monitors we switched off no longer appear in the online list: keep showing them so they can be turned back on.
        var off = turnedOff
        for (key, did) in off {
            if let seen = result.first(where: { $0.id == key }) {
                if seen.isOn { off[key] = nil }          // it came back (re-enabled elsewhere / replugged)
                continue
            }
            result.append(ExternalMonitor(id: key, displayID: did, name: names[key] ?? "External display \(did)",
                                          connection: "External", isOn: false))
        }
        turnedOff = off
        return result
    }

    /// Returns an error message, or nil on success.
    static func set(_ monitor: ExternalMonitor, on: Bool) -> String? {
        if !on {
            let activeCount = list().filter(\.isOn).count + NSScreen.screens.filter { CGDisplayIsBuiltin($0.displayID ?? 0) != 0 }.count
            if activeCount <= 1 { return "Can't turn off the only active display." }
        }
        var config: CGDisplayConfigRef?
        guard CGBeginDisplayConfiguration(&config) == .success else { return "Could not start display configuration." }
        guard CGSConfigureDisplayEnabled(config, monitor.displayID, on) == .success else {
            CGCancelDisplayConfiguration(config)
            return "macOS refused to change \(monitor.name)."
        }
        guard CGCompleteDisplayConfiguration(config, .forSession) == .success else {
            return "Could not apply the change to \(monitor.name)."
        }
        var off = turnedOff
        off[monitor.id] = on ? nil : monitor.displayID
        turnedOff = off
        return nil
    }

    /// Reconnect everything (used on quit so a monitor is never left unreachable).
    static func enableAll() {
        for m in list() where !m.isOn { _ = set(m, on: true) }
    }
}

extension NSScreen {
    var displayID: CGDirectDisplayID? {
        (deviceDescription[NSDeviceDescriptionKey("NSScreenNumber")] as? NSNumber)?.uint32Value
    }
}
