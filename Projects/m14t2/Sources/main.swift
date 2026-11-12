// m14t-touch — user-space touchscreen driver for the Lenovo ThinkVision M14t Gen 2 on macOS.

import AppKit
import ApplicationServices
import IOKit.hid

let logURL = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Library/Logs/m14t-touch.log")
let logFile: FileHandle? = {
    FileManager.default.createFile(atPath: logURL.path, contents: nil)  // truncate on each launch
    return try? FileHandle(forWritingTo: logURL)
}()

/// Logs to ~/Library/Logs/m14t-touch.log, and to the terminal when run from one.
func log(_ s: String) {
    let ts = ISO8601DateFormatter.string(from: Date(), timeZone: .current, formatOptions: [.withTime, .withColonSeparatorInTime])
    let data = "[\(ts)] \(s)\n".data(using: .utf8)!
    logFile?.write(data)
    if isatty(STDERR_FILENO) != 0 { FileHandle.standardError.write(data) }
}

let usage = """
usage: m14t-touch [options]
  --dump                     print raw HID reports, don't generate events
  --list-displays            list displays and exit
  --display <id|name>        map touch to this display (default: auto-detect M14t)
  --rotate <0|90|180|270>    override rotation used for mapping
  --long-press-right-click   hold one finger 0.6 s for a right click
  --invert-scroll            reverse two-finger scroll direction
  --no-pen                   ignore the pen
  --no-gestures              only click/drag/scroll/right-click (no zoom, rotate, swipes)
  --seize                    take exclusive access to the touch controller
  -v, --verbose              log generated events
"""

var opts = Options()
var args = CommandLine.arguments.dropFirst()
while let a = args.popFirst() {
    switch a {
    case "--dump": opts.dump = true
    case "--list-displays": print(DisplayMapper.describeAll()); exit(0)
    case "--display": opts.display = args.popFirst()
    case "--rotate":
        guard let r = args.popFirst().flatMap(Int.init), [0, 90, 180, 270].contains(r) else { print(usage); exit(2) }
        opts.rotation = r
    case "--long-press-right-click": opts.longPressRightClick = true
    case "--invert-scroll": opts.invertScroll = true
    case "--no-pen": opts.pen = false
    case "--no-gestures": opts.gestures = false
    case "--seize": opts.seize = true
    case "-v", "--verbose": opts.verbose = true
    case "-h", "--help": print(usage); exit(0)
    default: print("unknown option \(a)\n\(usage)"); exit(2)
    }
}

final class Driver {
    let opts: Options
    let mapper: DisplayMapper
    let engine: TouchEngine
    var assembler = MultitouchAssembler()
    var sawMultitouch = false
    var reportCount = 0
    let manager = IOHIDManagerCreate(kCFAllocatorDefault, IOOptionBits(kIOHIDOptionsTypeNone))

    init(opts: Options) {
        self.opts = opts
        mapper = DisplayMapper(hint: opts.display, rotationOverride: opts.rotation)
        engine = TouchEngine(opts: opts, mapper: mapper)
    }

    func start() {
        IOHIDManagerSetDeviceMatching(manager, [
            kIOHIDVendorIDKey: M14t.vendorID,
            kIOHIDProductIDKey: M14t.productID,
        ] as CFDictionary)
        let ctx = Unmanaged.passUnretained(self).toOpaque()
        IOHIDManagerRegisterDeviceMatchingCallback(manager, { ctx, _, _, dev in
            let usage = IOHIDDeviceGetProperty(dev, kIOHIDPrimaryUsageKey as CFString) as? Int ?? -1
            log("connected: touch controller interface (primary usage 0x\(String(usage, radix: 16)))")
            _ = ctx
        }, ctx)
        IOHIDManagerRegisterDeviceRemovalCallback(manager, { _, _, _, _ in
            log("disconnected: touch controller interface")
        }, ctx)
        IOHIDManagerRegisterInputReportCallback(manager, { ctx, _, _, _, reportID, report, len in
            guard let ctx else { return }
            Unmanaged<Driver>.fromOpaque(ctx).takeUnretainedValue()
                .onReport(id: UInt8(truncatingIfNeeded: reportID), UnsafeBufferPointer(start: report, count: len))
        }, ctx)
        IOHIDManagerScheduleWithRunLoop(manager, CFRunLoopGetMain(), CFRunLoopMode.defaultMode.rawValue)

        log("m14t-touch started" + (opts.dump ? " (dump mode)" : "") + ". Displays:\n" + DisplayMapper.describeAll())
    }

    private(set) var isOpen = false

    /// Opens the touch controller. Fails with kIOReturnNotPermitted (0xe00002e2) without Input Monitoring.
    @discardableResult
    func open() -> IOReturn {
        if isOpen { return kIOReturnSuccess }
        let openOpts = IOOptionBits(opts.seize ? kIOHIDOptionsTypeSeizeDevice : kIOHIDOptionsTypeNone)
        var r = IOHIDManagerOpen(manager, openOpts)
        if r != kIOReturnSuccess && opts.seize {
            log(String(format: "seize failed (0x%08x); opening shared", r))
            r = IOHIDManagerOpen(manager, IOOptionBits(kIOHIDOptionsTypeNone))
        }
        if r == kIOReturnSuccess { isOpen = true; log("touch controller opened") }
        else { IOHIDManagerClose(manager, IOOptionBits(kIOHIDOptionsTypeNone)) }
        return r
    }

    func onReport(id: UInt8, _ raw: UnsafeBufferPointer<UInt8>) {
        // The buffer normally starts with the report ID; normalize just in case.
        reportCount += 1
        if reportCount == 1 { log("receiving touch data") }
        let bytes = raw.first == id ? Array(raw) : [id] + Array(raw)
        bytes.withUnsafeBufferPointer { b in
            if opts.dump { dump(b); return }
            switch b[0] {
            case M14t.multitouchReport:
                sawMultitouch = true
                if let frame = assembler.feed(b) { engine.handleTouch(frame) }
            case M14t.singleTouchReport where !sawMultitouch:
                if let frame = parseSingleTouch(b) { engine.handleTouch(frame) }
            case M14t.penReport:
                if let s = parsePen(b) { engine.handlePen(s) }
            default: break
            }
        }
    }

    private var dumpAssembler = MultitouchAssembler()
    func dump(_ b: UnsafeBufferPointer<UInt8>) {
        var line = String(format: "id=0x%02x len=%2d  ", b[0], b.count)
        switch b[0] {
        case M14t.multitouchReport:
            if let f = dumpAssembler.feed(b) {
                line += "MT " + f.map { c in
                    let p = mapper.map(u: c.u, v: c.v)
                    return "#\(c.id)\(c.tip ? "↓" : "↑")\(c.confident ? "" : "?")(\(Int(p.x)),\(Int(p.y)))"
                }.joined(separator: " ")
            } else { line += "MT (partial) " + hex(b) }
        case M14t.singleTouchReport:
            line += "ST " + (parseSingleTouch(b)?.map { c in
                let p = mapper.map(u: c.u, v: c.v)
                return "#\(c.id)\(c.tip ? "↓" : "↑")(\(Int(p.x)),\(Int(p.y)))"
            }.joined() ?? hex(b))
        case M14t.penReport:
            if let s = parsePen(b) {
                let p = mapper.map(u: s.u, v: s.v)
                line += String(format: "PEN range=%d tip=%d barrel=%d eraser=%d (%d,%d) p=%.2f tilt=%.0f,%.0f",
                               s.inRange ? 1 : 0, s.tip ? 1 : 0, s.barrel ? 1 : 0, s.eraser ? 1 : 0,
                               Int(p.x), Int(p.y), s.pressure, s.tiltX, s.tiltY)
            }
        default: line += hex(b)
        }
        print(line); fflush(stdout)
    }
}

// MARK: - App (menu-bar agent; never blocks the main thread)

final class AppDelegate: NSObject, NSApplicationDelegate {
    var driver: Driver?
    var statusItem: NSStatusItem?
    let statusLine = NSMenuItem(title: "Starting…", action: nil, keyEquivalent: "")
    var askedHID = false, askedAX = false

    func applicationDidFinishLaunching(_ n: Notification) {
        let item = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        item.button?.image = NSImage(systemSymbolName: "hand.point.up.left", accessibilityDescription: "M14t Touch")
        let menu = NSMenu()
        menu.addItem(withTitle: "M14t Touch", action: nil, keyEquivalent: "").isEnabled = false
        menu.addItem(statusLine)
        menu.addItem(.separator())
        menu.addItem(withTitle: "Open Input Monitoring Settings…", action: #selector(openHID), keyEquivalent: "").target = self
        menu.addItem(withTitle: "Open Accessibility Settings…", action: #selector(openAX), keyEquivalent: "").target = self
        menu.addItem(withTitle: "Show Log", action: #selector(showLog), keyEquivalent: "").target = self
        menu.addItem(.separator())
        menu.addItem(withTitle: "Quit M14t Touch", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        item.menu = menu
        statusItem = item
        checkPermissions()
    }

    private func setStatus(_ s: String) {
        if statusLine.title != s { statusLine.title = s; log(s) }
    }

    var lastOpenError: IOReturn = 0

    /// Polls permissions on a timer. Input Monitoring is verified by actually opening the
    /// device (IOHIDCheckAccess can be stale for ad-hoc signed apps).
    func checkPermissions() {
        if driver == nil { driver = Driver(opts: opts); driver?.start() }
        guard let driver else { return }

        let hidCheck = IOHIDCheckAccess(kIOHIDRequestTypeListenEvent)
        if !driver.isOpen {
            if hidCheck != kIOHIDAccessTypeGranted && !askedHID {
                askedHID = true
                _ = IOHIDRequestAccess(kIOHIDRequestTypeListenEvent)
            }
            let r = driver.open()
            if r != kIOReturnSuccess && r != lastOpenError {
                lastOpenError = r
                log(String(format: "open touch controller: 0x%08x, IOHIDCheckAccess=%u", r, hidCheck.rawValue))
            }
        }

        let axOK = opts.dump || AXIsProcessTrusted()
        if !axOK && !askedAX {
            askedAX = true
            let prompt = kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String
            _ = AXIsProcessTrustedWithOptions([prompt: true] as CFDictionary)
        }

        if !driver.isOpen && !axOK { setStatus("⚠️ Needs Input Monitoring + Accessibility") }
        else if !driver.isOpen { setStatus("⚠️ Needs Input Monitoring permission") }
        else if !axOK { setStatus("⚠️ Needs Accessibility permission") }
        else { setStatus(opts.dump ? "Dump mode" : "✅ Running — touch the screen"); return }
        retry()
    }

    private func retry() {
        DispatchQueue.main.asyncAfter(deadline: .now() + 2) { [weak self] in self?.checkPermissions() }
    }

    @objc func openHID() {
        NSWorkspace.shared.open(URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_ListenEvent")!)
    }
    @objc func openAX() {
        NSWorkspace.shared.open(URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility")!)
    }
    @objc func showLog() { NSWorkspace.shared.open(logURL) }
}

setvbuf(stdout, nil, _IOLBF, 0)
signal(SIGINT) { _ in exit(0) }
signal(SIGTERM) { _ in exit(0) }
let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.setActivationPolicy(.accessory)
app.run()
