// HID report layouts for the Lenovo ThinkVision M14t Gen 2 touch/pen controller
// (USB VID 0x2D1F, PID 0x53D1, "Pen and multitouch sensor").
// Derived from the device's HID report descriptors (see README).

import Foundation

enum M14t {
    static let vendorID = 0x2D1F
    static let productID = 0x53D1
    static let displayVendor: UInt32 = 0x30AE // "LEN"
    static let displayModel: UInt32 = 0x63FD

    // Report IDs
    static let multitouchReport: UInt8 = 0x0C   // interface 0, 5 contacts per report, 60 bytes
    static let singleTouchReport: UInt8 = 0x24  // interface 1, 1 contact, 11 bytes
    static let penReport: UInt8 = 0x14          // interface 0, 22 bytes
    static let mouseReport: UInt8 = 0x01        // interface 0, absolute mouse fallback

    // Logical ranges
    static let touchMaxX = 30154.0  // 0x75CA (30.154 cm)
    static let touchMaxY = 18846.0  // 0x499E (18.846 cm)
    static let stMaxX = 30155.0     // 0x75CB
    static let stMaxY = 18847.0     // 0x499F
    static let penMaxX = 30155.0
    static let penMaxY = 18847.0
    static let penMaxPressure = 4095.0
}

struct Contact {
    var id: Int
    var tip: Bool
    var confident: Bool
    var u: Double   // 0...1 across the panel's native width
    var v: Double   // 0...1 down the panel's native height
}

struct PenSample {
    var tip, barrel, eraser, invert, barrel2, inRange: Bool
    var u, v: Double
    var pressure: Double // 0...1
    var tiltX, tiltY: Double // degrees
}

@inline(__always) private func le16(_ b: UnsafeBufferPointer<UInt8>, _ o: Int) -> Int {
    Int(b[o]) | Int(b[o + 1]) << 8
}
@inline(__always) private func le32(_ b: UnsafeBufferPointer<UInt8>, _ o: Int) -> Int {
    Int(b[o]) | Int(b[o + 1]) << 8 | Int(b[o + 2]) << 16 | Int(b[o + 3]) << 24
}
@inline(__always) private func clamp01(_ x: Double) -> Double { min(max(x, 0), 1) }

/// Reassembles Windows "hybrid mode" multitouch frames: the first report of a frame
/// carries the total contact count, continuation reports carry 0.
struct MultitouchAssembler {
    private var expected = 0
    private var contacts: [Contact] = []

    /// Returns a complete frame (all contacts, including ones lifting) when available.
    mutating func feed(_ b: UnsafeBufferPointer<UInt8>) -> [Contact]? {
        guard b.count >= 58, b[0] == M14t.multitouchReport else { return nil }
        let count = Int(b[2])
        if count > 0 { expected = count; contacts.removeAll(keepingCapacity: true) }
        else if expected == 0 { return [] } // empty frame: nothing touching
        let remaining = expected - contacts.count
        for i in 0..<min(5, max(remaining, 0)) {
            let o = 3 + i * 11
            let flags = b[o]
            contacts.append(Contact(
                id: le16(b, o + 1),
                tip: flags & 0x01 != 0,
                confident: flags & 0x04 != 0,
                u: clamp01(Double(le32(b, o + 3)) / M14t.touchMaxX),
                v: clamp01(Double(le32(b, o + 7)) / M14t.touchMaxY)))
        }
        if contacts.count >= expected {
            expected = 0
            return contacts
        }
        return nil
    }
}

func parseSingleTouch(_ b: UnsafeBufferPointer<UInt8>) -> [Contact]? {
    guard b.count >= 9, b[0] == M14t.singleTouchReport else { return nil }
    if b[2] == 0 { return [] }
    let flags = b[3]
    return [Contact(id: Int(b[4]),
                    tip: flags & 0x01 != 0,
                    confident: flags & 0x04 != 0,
                    u: clamp01(Double(le16(b, 5)) / M14t.stMaxX),
                    v: clamp01(Double(le16(b, 7)) / M14t.stMaxY))]
}

func parsePen(_ b: UnsafeBufferPointer<UInt8>) -> PenSample? {
    guard b.count >= 20, b[0] == M14t.penReport else { return nil }
    let f = b[1]
    return PenSample(tip: f & 0x01 != 0, barrel: f & 0x02 != 0, eraser: f & 0x04 != 0,
                     invert: f & 0x08 != 0, barrel2: f & 0x10 != 0, inRange: f & 0x20 != 0,
                     u: clamp01(Double(le16(b, 2)) / M14t.penMaxX),
                     v: clamp01(Double(le16(b, 4)) / M14t.penMaxY),
                     pressure: clamp01(Double(le16(b, 6)) / M14t.penMaxPressure),
                     tiltX: Double(Int16(truncatingIfNeeded: le16(b, 16))) / 100,
                     tiltY: Double(Int16(truncatingIfNeeded: le16(b, 18))) / 100)
}

func hex(_ b: UnsafeBufferPointer<UInt8>) -> String {
    b.map { String(format: "%02x", $0) }.joined(separator: " ")
}
