import AppKit
import CoreGraphics
import Foundation

struct Options {
    var dump = false
    var verbose = false
    var seize = false
    var pen = true
    var gestures = true
    var longPressRightClick = false
    var invertScroll = false
    var display: String?
    var rotation: Int?
    var tapSlop = 10.0          // px a finger may wander and still count as a tap
    var dragDelay = 0.08        // s to wait for a 2nd finger before pressing the button
    var longPress = 0.6         // s
    var multiTapMax = 0.45      // s
    var pinchThreshold = 25.0   // px change in finger spread before zooming starts
    var rotateThreshold = 15.0  // degrees before rotation starts
    var swipeThreshold = 80.0   // px a 3/4-finger swipe must travel
}

/// Turns touch frames and pen samples into macOS mouse, scroll and gesture events.
///
/// Gestures (trackpad equivalents):
///   1 finger tap / double tap ... click / double click
///   1 finger drag ............... drag
///   1 finger hold ............... right click (only with --long-press-right-click)
///   2 finger tap ................ right click
///   2 finger double tap ......... smart zoom
///   2 finger drag ............... scroll
///   2 finger pinch / rotate ..... zoom / rotate (real magnify & rotate gesture events)
///   3 finger tap ................ Look Up (⌃⌘D)
///   3/4 finger swipe left/right . switch Space / full-screen app
///   3/4 finger swipe up ......... Mission Control
///   3/4 finger swipe down ....... App Exposé
///   4+ finger pinch in .......... Apps (Launchpad)
///   4+ finger spread out ........ Show Desktop
///   pen hover / tip / barrel .... move / drag with pressure / right click
final class TouchEngine {
    private enum Mode { case idle, pending, dragging, multi }
    private enum MultiKind { case undecided, scroll, transform, done }
    private typealias Pt = (id: Int, p: CGPoint)

    private let opts: Options
    private let mapper: DisplayMapper
    private let src = CGEventSource(stateID: .hidSystemState)

    private var mode = Mode.idle
    private var primaryID = -1
    private var downPoint = CGPoint.zero
    private var lastPoint = CGPoint.zero
    private var timer: DispatchWorkItem?

    // multi-finger state
    private var kind = MultiKind.undecided
    private var maxFingers = 0
    private var multiStart = 0.0
    private var fingerIDs: [Int] = []
    private var startCentroid = CGPoint.zero
    private var lastCentroid = CGPoint.zero
    private var startSpread = 0.0, lastSpread = 0.0
    private var startAngle = 0.0, lastAngle = 0.0
    private var rotating = false
    private var scrollRemainder = CGPoint.zero
    private var pendingRightClick: DispatchWorkItem?
    private var lastTwoFingerTap = 0.0
    private var lastTwoFingerTapPoint = CGPoint.zero

    private var clickCount = 0
    private var lastClickTime = 0.0
    private var lastClickPoint = CGPoint.zero

    private var penDown = false
    private var penButton = CGMouseButton.left
    private var penInRange = false
    private var penLeftAt = 0.0

    init(opts: Options, mapper: DisplayMapper) {
        self.opts = opts
        self.mapper = mapper
        src?.localEventsSuppressionInterval = 0
    }

    private var now: Double { ProcessInfo.processInfo.systemUptime }

    // MARK: - Touch

    func handleTouch(_ frame: [Contact]) {
        // Palm rejection: ignore touches while the pen is near the screen.
        if penInRange || now - penLeftAt < 0.3 { if mode != .idle { cancelTouch() }; return }

        let pts: [Pt] = frame.filter { $0.tip && $0.confident }
            .map { (id: $0.id, p: mapper.map(u: $0.u, v: $0.v)) }
            .sorted { $0.id < $1.id }

        switch mode {
        case .idle:
            if pts.count == 1 { beginPending(pts[0].id, pts[0].p) }
            else if pts.count >= 2 { beginMulti(pts) }

        case .pending:
            if pts.isEmpty { cancelTimer(); tap(at: downPoint, button: .left); mode = .idle }
            else if pts.count >= 2 { cancelTimer(); beginMulti(pts) }
            else {
                lastPoint = pts[0].p
                if dist(lastPoint, downPoint) > opts.tapSlop { cancelTimer(); beginDrag(); drag(to: lastPoint) }
            }

        case .dragging:
            if pts.isEmpty { post(.leftMouseUp, lastPoint, .left, clicks: clickCount); mode = .idle; return }
            let p = (pts.first { $0.id == primaryID } ?? pts[0]).p
            if p != lastPoint { drag(to: p) }

        case .multi:
            if pts.isEmpty { endMulti(); return }
            if pts.count > maxFingers {
                // More fingers joined: finish any 2-finger gesture and re-baseline.
                finishContinuous()
                maxFingers = pts.count
                if kind != .done { kind = .undecided }
                setBaseline(pts)
                return
            }
            if pts.count < 2 { return } // leftover finger: wait for all to lift
            if pts.map(\.id) != fingerIDs {
                setBaseline(pts, keepStart: true) // a finger swapped: avoid jumps
                return
            }
            if maxFingers == 2 { twoFinger(pts) } else { manyFinger(pts) }
        }
    }

    // MARK: 1 finger

    private func beginPending(_ id: Int, _ p: CGPoint) {
        mode = .pending
        primaryID = id
        downPoint = p; lastPoint = p
        post(.mouseMoved, p, .left)
        let delay = opts.longPressRightClick ? opts.longPress : opts.dragDelay
        let w = DispatchWorkItem { [weak self] in
            guard let self, self.mode == .pending else { return }
            if self.opts.longPressRightClick {
                self.tap(at: self.downPoint, button: .right)
                self.mode = .multi; self.kind = .done; self.maxFingers = 1 // swallow rest of touch
            } else {
                self.beginDrag()
            }
        }
        timer = w
        DispatchQueue.main.asyncAfter(deadline: .now() + delay, execute: w)
    }

    private func beginDrag() {
        mode = .dragging
        bumpClickCount(at: downPoint)
        post(.leftMouseDown, downPoint, .left, clicks: clickCount)
    }

    private func drag(to p: CGPoint) {
        lastPoint = p
        post(.leftMouseDragged, p, .left, clicks: clickCount)
    }

    // MARK: multi-finger

    private func beginMulti(_ pts: [Pt]) {
        mode = .multi
        kind = .undecided
        maxFingers = pts.count
        multiStart = now
        rotating = false
        scrollRemainder = .zero
        setBaseline(pts)
        if pts.count == 2 { post(.mouseMoved, startCentroid, .left) }
    }

    private func setBaseline(_ pts: [Pt], keepStart: Bool = false) {
        let ps = pts.map(\.p)
        let c = centroid(ps)
        fingerIDs = pts.map(\.id)
        if !keepStart {
            startCentroid = c
            startSpread = spread(ps, c)
            startAngle = angle(ps)
        } else {
            // shift start values so accumulated deltas stay continuous
            startCentroid.x += c.x - lastCentroid.x; startCentroid.y += c.y - lastCentroid.y
            startSpread += spread(ps, c) - lastSpread
            startAngle += angle(ps) - lastAngle
        }
        lastCentroid = c
        lastSpread = spread(ps, c)
        lastAngle = angle(ps)
    }

    private func twoFinger(_ pts: [Pt]) {
        let ps = pts.map(\.p)
        let c = centroid(ps), s = spread(ps, c), a = angle(ps)
        defer { lastCentroid = c; lastSpread = s; lastAngle = a }

        if kind == .undecided {
            let moved = dist(c, startCentroid)
            let pinch = abs(s - startSpread)
            let turn = abs(angleDiff(a, startAngle))
            if opts.gestures && (pinch > opts.pinchThreshold || turn > opts.rotateThreshold) {
                kind = .transform
                rotating = turn > opts.rotateThreshold
                gesture(.magnify, phase: 1, value: 0)
                if rotating { gesture(.rotate, phase: 1, value: 0) }
                return
            }
            if moved > opts.tapSlop * 1.5 {
                kind = .scroll
                scroll(dx: 0, dy: 0, phase: 1)
            }
        }
        switch kind {
        case .scroll:
            scroll(dx: c.x - lastCentroid.x, dy: c.y - lastCentroid.y, phase: 2)
        case .transform:
            if lastSpread > 1 { gesture(.magnify, phase: 2, value: s / lastSpread - 1) }
            if !rotating && abs(angleDiff(a, startAngle)) > opts.rotateThreshold {
                rotating = true
                gesture(.rotate, phase: 1, value: 0)
            }
            // Screen y points down, so a clockwise turn increases the angle; NSEvent rotation is CCW-positive.
            if rotating { gesture(.rotate, phase: 2, value: -angleDiff(a, lastAngle)) }
        default: break
        }
    }

    private func manyFinger(_ pts: [Pt]) {
        guard opts.gestures, kind == .undecided else { return }
        let ps = pts.map(\.p)
        let c = centroid(ps), s = spread(ps, c)
        lastCentroid = c; lastSpread = s
        let dx = c.x - startCentroid.x, dy = c.y - startCentroid.y
        let ds = s - startSpread

        if maxFingers >= 4 && abs(ds) > opts.swipeThreshold * 0.75 && abs(ds) > hypot(dx, dy) {
            kind = .done
            ds < 0 ? SystemActions.apps() : SystemActions.showDesktop()
            if opts.verbose { log(ds < 0 ? "gesture: pinch → Apps" : "gesture: spread → Show Desktop") }
            return
        }
        guard hypot(dx, dy) > opts.swipeThreshold else { return }
        kind = .done
        if abs(dx) > abs(dy) {
            // Like the trackpad: swiping left reveals the Space to the right.
            SystemActions.switchSpace(right: dx < 0, src: src)
            if opts.verbose { log("gesture: swipe \(dx < 0 ? "left" : "right") → Space") }
        } else if dy < 0 {
            SystemActions.missionControl()
            if opts.verbose { log("gesture: swipe up → Mission Control") }
        } else {
            SystemActions.appExpose()
            if opts.verbose { log("gesture: swipe down → App Exposé") }
        }
    }

    private func finishContinuous() {
        switch kind {
        case .scroll: scroll(dx: 0, dy: 0, phase: 4)
        case .transform:
            gesture(.magnify, phase: 4, value: 0)
            if rotating { gesture(.rotate, phase: 4, value: 0) }
        default: break
        }
    }

    private func endMulti() {
        finishContinuous()
        let quick = now - multiStart < opts.multiTapMax
        if kind == .undecided && quick {
            if maxFingers == 2 { twoFingerTap() }
            else if maxFingers == 3 && opts.gestures {
                post(.mouseMoved, startCentroid, .left)
                SystemActions.lookUp(src: src)
                if opts.verbose { log("gesture: 3-finger tap → Look Up") }
            }
        }
        mode = .idle
    }

    /// Two-finger tap = right click, unless a second two-finger tap follows (smart zoom).
    private func twoFingerTap() {
        let p = startCentroid
        if opts.gestures, now - lastTwoFingerTap < 0.35, dist(p, lastTwoFingerTapPoint) < 60 {
            pendingRightClick?.cancel(); pendingRightClick = nil
            lastTwoFingerTap = 0
            post(.mouseMoved, p, .left)
            gesture(.smartMagnify, phase: 0, value: 0)
            return
        }
        lastTwoFingerTap = now
        lastTwoFingerTapPoint = p
        guard opts.gestures else { tap(at: p, button: .right); return }
        let w = DispatchWorkItem { [weak self] in self?.tap(at: p, button: .right) }
        pendingRightClick = w
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.3, execute: w)
    }

    private func cancelTouch() {
        cancelTimer()
        if mode == .dragging { post(.leftMouseUp, lastPoint, .left, clicks: clickCount) }
        if mode == .multi { finishContinuous() }
        mode = .idle
    }

    private func cancelTimer() { timer?.cancel(); timer = nil }

    private func bumpClickCount(at p: CGPoint) {
        if now - lastClickTime < opts.multiTapMax && dist(p, lastClickPoint) < opts.tapSlop * 3 {
            clickCount = min(clickCount + 1, 3)
        } else {
            clickCount = 1
        }
        lastClickTime = now
        lastClickPoint = p
    }

    private func tap(at p: CGPoint, button: CGMouseButton) {
        if button == .left { bumpClickCount(at: p) }
        let clicks = button == .left ? clickCount : 1
        let (down, up): (CGEventType, CGEventType) = button == .left
            ? (.leftMouseDown, .leftMouseUp) : (.rightMouseDown, .rightMouseUp)
        post(down, p, button, clicks: clicks)
        post(up, p, button, clicks: clicks)
        if button == .left { lastClickTime = now }
    }

    // MARK: - Event synthesis

    /// phase: 1 began, 2 changed, 4 ended (kCGScrollPhase*)
    private func scroll(dx: Double, dy: Double, phase: Int64) {
        let sign = opts.invertScroll ? -1.0 : 1.0
        let fx = dx * sign + scrollRemainder.x, fy = dy * sign + scrollRemainder.y
        let ix = fx.rounded(.towardZero), iy = fy.rounded(.towardZero)
        scrollRemainder = CGPoint(x: fx - ix, y: fy - iy)
        guard let e = CGEvent(scrollWheelEvent2Source: src, units: .pixel, wheelCount: 2,
                              wheel1: Int32(iy), wheel2: Int32(ix), wheel3: 0) else { return }
        e.location = lastCentroid
        e.setIntegerValueField(.scrollWheelEventIsContinuous, value: 1)
        e.setIntegerValueField(.scrollWheelEventScrollPhase, value: phase)
        e.setDoubleValueField(.scrollWheelEventFixedPtDeltaAxis1, value: fy)
        e.setDoubleValueField(.scrollWheelEventFixedPtDeltaAxis2, value: fx)
        e.setDoubleValueField(.scrollWheelEventPointDeltaAxis1, value: fy)
        e.setDoubleValueField(.scrollWheelEventPointDeltaAxis2, value: fx)
        e.post(tap: .cghidEventTap)
        if opts.verbose { log("scroll dx=\(ix) dy=\(iy) phase=\(phase)") }
    }

    private enum GestureType: Int64 { case rotate = 5, magnify = 8, smartMagnify = 22 }

    /// Posts a trackpad gesture event (the same kind the built-in trackpad produces).
    /// Uses WindowServer's undocumented gesture fields; AppKit decodes them as
    /// NSEvent .magnify / .rotate / .smartMagnify.
    /// phase: IOHIDEventPhase bits — 1 began, 2 changed, 4 ended.
    private func gesture(_ type: GestureType, phase: Int64, value: Double) {
        guard let e = CGEvent(source: src) else { return }
        e.location = lastCentroid
        e.setIntegerValueField(CGEventField(rawValue: 55)!, value: 29)   // event type: gesture
        e.setIntegerValueField(CGEventField(rawValue: 110)!, value: type.rawValue)
        e.setIntegerValueField(CGEventField(rawValue: 132)!, value: phase)
        switch type {
        case .magnify: e.setDoubleValueField(CGEventField(rawValue: 113)!, value: value)
        case .rotate: e.setDoubleValueField(CGEventField(rawValue: 114)!, value: value)
        case .smartMagnify: break
        }
        e.post(tap: .cghidEventTap)
        if opts.verbose && (phase != 2) { log("gesture \(type) phase=\(phase)") }
    }

    // MARK: - Pen

    func handlePen(_ s: PenSample) {
        guard opts.pen else { return }
        let p = mapper.map(u: s.u, v: s.v)
        if s.inRange && !penInRange && mode != .idle { cancelTouch() }

        if s.tip && s.inRange {
            if !penDown {
                penDown = true
                penButton = (s.barrel || s.barrel2) ? .right : .left
                if penButton == .left { bumpClickCount(at: p) }
                postPen(penButton == .left ? .leftMouseDown : .rightMouseDown, p, s)
            } else {
                postPen(penButton == .left ? .leftMouseDragged : .rightMouseDragged, p, s)
            }
        } else {
            if penDown {
                penDown = false
                postPen(penButton == .left ? .leftMouseUp : .rightMouseUp, p, s)
            }
            if s.inRange { postPen(.mouseMoved, p, s) }
        }
        if penInRange && !s.inRange { penLeftAt = now }
        penInRange = s.inRange
    }

    private func postPen(_ type: CGEventType, _ p: CGPoint, _ s: PenSample) {
        guard let e = CGEvent(mouseEventSource: src, mouseType: type, mouseCursorPosition: p, mouseButton: penButton) else { return }
        let pressure = (type == .mouseMoved || type == .leftMouseUp || type == .rightMouseUp) ? 0 : max(s.pressure, 0.01)
        e.setIntegerValueField(.mouseEventSubtype, value: Int64(CGEventMouseSubtype.tabletPoint.rawValue))
        e.setDoubleValueField(.mouseEventPressure, value: pressure)
        e.setDoubleValueField(.tabletEventPointPressure, value: pressure)
        e.setDoubleValueField(.tabletEventTiltX, value: s.tiltX / 90)
        e.setDoubleValueField(.tabletEventTiltY, value: s.tiltY / 90)
        if penButton == .left, type != .mouseMoved {
            e.setIntegerValueField(.mouseEventClickState, value: Int64(clickCount))
        }
        e.post(tap: .cghidEventTap)
    }

    // MARK: - Helpers

    private func post(_ type: CGEventType, _ p: CGPoint, _ button: CGMouseButton, clicks: Int = 0) {
        guard let e = CGEvent(mouseEventSource: src, mouseType: type, mouseCursorPosition: p, mouseButton: button) else { return }
        if clicks > 0 { e.setIntegerValueField(.mouseEventClickState, value: Int64(clicks)) }
        e.post(tap: .cghidEventTap)
        if opts.verbose && type != .leftMouseDragged && type != .mouseMoved {
            log("\(type) at (\(Int(p.x)),\(Int(p.y))) clicks=\(clicks)")
        }
    }
}

/// System-level trackpad actions, triggered the same way the Dock / Mission Control do.
enum SystemActions {
    private static let missionControlBin = "/System/Applications/Mission Control.app/Contents/MacOS/Mission Control"

    private static func runMissionControl(_ args: [String]) {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: missionControlBin)
        p.arguments = args
        try? p.run()
    }

    static func missionControl() { runMissionControl([]) }
    static func showDesktop() { runMissionControl(["1"]) }
    static func appExpose() { runMissionControl(["2"]) }

    static func apps() {
        let fm = FileManager.default
        for path in ["/System/Applications/Apps.app", "/System/Applications/Launchpad.app"] where fm.fileExists(atPath: path) {
            NSWorkspace.shared.openApplication(at: URL(fileURLWithPath: path), configuration: .init())
            return
        }
    }

    /// ⌃← / ⌃→ — the default "Move left/right a space" shortcuts.
    static func switchSpace(right: Bool, src: CGEventSource?) {
        key(right ? 124 : 123, [.maskControl, .maskSecondaryFn], src: src)
    }

    /// ⌃⌘D — Look up word under the pointer.
    static func lookUp(src: CGEventSource?) {
        key(2, [.maskControl, .maskCommand], src: src)
    }

    private static func key(_ code: CGKeyCode, _ flags: CGEventFlags, src: CGEventSource?) {
        for down in [true, false] {
            guard let e = CGEvent(keyboardEventSource: src, virtualKey: code, keyDown: down) else { continue }
            e.flags = flags
            e.post(tap: .cghidEventTap)
        }
    }
}

private func dist(_ a: CGPoint, _ b: CGPoint) -> Double { hypot(a.x - b.x, a.y - b.y) }
private func centroid(_ ps: [CGPoint]) -> CGPoint {
    let n = Double(ps.count)
    return CGPoint(x: ps.reduce(0) { $0 + $1.x } / n, y: ps.reduce(0) { $0 + $1.y } / n)
}
/// Mean distance of the fingers from their centroid.
private func spread(_ ps: [CGPoint], _ c: CGPoint) -> Double {
    ps.reduce(0) { $0 + dist($1, c) } / Double(ps.count)
}
/// Angle (degrees) of the line through the first two fingers.
private func angle(_ ps: [CGPoint]) -> Double {
    guard ps.count >= 2 else { return 0 }
    return atan2(ps[1].y - ps[0].y, ps[1].x - ps[0].x) * 180 / .pi
}
private func angleDiff(_ a: Double, _ b: Double) -> Double {
    var d = a - b
    while d > 180 { d -= 360 }
    while d < -180 { d += 360 }
    return d
}

extension CGEventType: @retroactive CustomStringConvertible {
    public var description: String {
        switch self {
        case .leftMouseDown: "leftDown"; case .leftMouseUp: "leftUp"
        case .rightMouseDown: "rightDown"; case .rightMouseUp: "rightUp"
        case .mouseMoved: "move"; case .leftMouseDragged: "drag"
        default: "event(\(rawValue))"
        }
    }
}
