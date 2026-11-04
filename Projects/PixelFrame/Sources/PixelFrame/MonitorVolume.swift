import Foundation
import IOKit

// Private IOKit calls used by display utilities to talk DDC/CI to external monitors on Apple Silicon.
@_silgen_name("IOAVServiceCreateWithService")
private func IOAVServiceCreateWithService(_ allocator: CFAllocator?, _ service: io_service_t) -> Unmanaged<CFTypeRef>?
@_silgen_name("IOAVServiceWriteI2C")
private func IOAVServiceWriteI2C(_ service: CFTypeRef, _ chip: UInt32, _ offset: UInt32,
                                 _ data: UnsafeMutableRawPointer, _ size: UInt32) -> IOReturn
@_silgen_name("IOAVServiceReadI2C")
private func IOAVServiceReadI2C(_ service: CFTypeRef, _ chip: UInt32, _ offset: UInt32,
                                _ data: UnsafeMutableRawPointer, _ size: UInt32) -> IOReturn

/// Hardware volume of external monitors (their speakers / headphone jack) over DDC/CI, VCP code 0x62.
/// macOS volume keys can't do this because the monitor's own volume isn't part of the system mixer.
enum MonitorVolume {
    private static let volumeCode: UInt8 = 0x62
    private static let key = "monitorVolume"

    /// Last known level (0…100). Monitors often can't be read back, so we remember what we set.
    private static var level: Int {
        get { UserDefaults.standard.object(forKey: key) as? Int ?? 50 }
        set { UserDefaults.standard.set(newValue, forKey: key) }
    }

    private static func externalServices() -> [CFTypeRef] {
        var iter: io_iterator_t = 0
        guard IOServiceGetMatchingServices(kIOMainPortDefault, IOServiceMatching("DCPAVServiceProxy"), &iter) == KERN_SUCCESS else { return [] }
        defer { IOObjectRelease(iter) }
        var result: [CFTypeRef] = []
        while case let entry = IOIteratorNext(iter), entry != 0 {
            defer { IOObjectRelease(entry) }
            let loc = IORegistryEntryCreateCFProperty(entry, "Location" as CFString, kCFAllocatorDefault, 0)?
                .takeRetainedValue() as? String
            guard loc == "External", let svc = IOAVServiceCreateWithService(kCFAllocatorDefault, entry) else { continue }
            result.append(svc.takeRetainedValue())
        }
        return result
    }

    private static func checksum(_ bytes: [UInt8]) -> UInt8 { bytes.reduce(0x6E ^ 0x51, ^) }

    private static func write(_ svc: CFTypeRef, value: Int) -> Bool {
        var p: [UInt8] = [0x84, 0x03, volumeCode, UInt8((value >> 8) & 0xFF), UInt8(value & 0xFF)]
        p.append(checksum(p))
        for _ in 0..<2 {                                    // DDC is flaky: send twice
            if IOAVServiceWriteI2C(svc, 0x37, 0x51, &p, UInt32(p.count)) != kIOReturnSuccess { return false }
            usleep(10_000)
        }
        return true
    }

    /// (current, max) as reported by the monitor, if it answers.
    private static func read(_ svc: CFTypeRef) -> (Int, Int)? {
        var req: [UInt8] = [0x82, 0x01, volumeCode]
        req.append(checksum(req))
        guard IOAVServiceWriteI2C(svc, 0x37, 0x51, &req, UInt32(req.count)) == kIOReturnSuccess else { return nil }
        usleep(50_000)
        var buf = [UInt8](repeating: 0, count: 12)
        guard IOAVServiceReadI2C(svc, 0x37, 0x51, &buf, UInt32(buf.count)) == kIOReturnSuccess,
              buf[2] == 0x02, buf[3] == 0, buf[4] == volumeCode else { return nil }
        return (Int(buf[8]) << 8 | Int(buf[9]), Int(buf[6]) << 8 | Int(buf[7]))
    }

    static func currentLevel() -> Int {
        if let (cur, max) = externalServices().lazy.compactMap(read).first, max > 0 { level = cur * 100 / max }
        return level
    }

    /// Sets an absolute level (0…100) on every external monitor. Returns it, or nil if none accepted.
    @discardableResult
    static func set(_ value: Int) -> Int? {
        let new = min(100, Swift.max(0, value))
        var ok = false
        for svc in externalServices() where write(svc, value: new) { ok = true }
        guard ok else { return nil }
        level = new
        return new
    }

    /// Moves the volume by `delta` points on every external monitor. Returns the new level, or nil if no monitor accepted it.
    @discardableResult
    static func change(by delta: Int) -> Int? {
        let services = externalServices()
        guard !services.isEmpty else { return nil }
        if let (cur, max) = services.lazy.compactMap(read).first, max > 0 { level = cur * 100 / max }
        let new = min(100, Swift.max(0, level + delta))
        var ok = false
        for svc in services where write(svc, value: new) { ok = true }
        guard ok else { return nil }
        level = new
        return new
    }
}
