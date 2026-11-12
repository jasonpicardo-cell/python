import CoreGraphics
import Foundation

struct Options {
    var dump = false
    var verbose = false
    var seize = false
    var pen = true
    var longPressRightClick = false
    var invertScroll = false
    var display: String?
    var rotation: Int?
    var tapSlop = 10.0          // px a finger may wander and still count as a tap
    var dragDelay = 0.08        // s to wait for a 2nd finger before pressing the button
    var longPress = 0.6         // s
    var multiTapMax = 0.45      // s
}

/// Turns touch frames and pen samples into macOS mouse / scroll events.
///
/// Gestures:
///   1 finger tap ............ left click (double/triple tap = double/triple click)
///   1 finger drag ........... left drag (select, move windows, scroll bars)
///   1 finger hold ........... right click   (only with --long-press-right-click)
///   2 finger tap ............ right click
///   2 finger drag ........... scroll (vertical + horizontal)
///   pen hover / tip / barrel  move / left drag with pressure / right click
final class TouchEngine {
    private enum Mode { case idle, pending, dragging, multi }

    private let opts: Options
    private let mapper: DisplayMapper
    private let src = CGEventSource(stateID: .hidSystemState)

    private var mode = Mode.idle
    private var primaryID = -1
    private var downPoint = CGPoint.zero
    private var lastPoint = CGPoint.zero
    private var downTime = 0.0
    private var timer: DispatchWorkItem?

    private var maxFingers = 0
    private var multiStart = 0.0
    private var startCentroid = CGPoint.zero
    private var lastCentroid = CGPoint.zero
    private var scrolling = false
    private var scrollRemainder = CGPoint.zero

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

        let active = frame.filter { $0.tip && $0.confident }
        let pts = active.map { (id: $0.id, p: mapper.map(u: $0.u, v: $0.v)) }

        switch mode {
        case .idle:
            if pts.count == 1 { beginPending(pts[0].id, pts[0].p) }
            else if pts.count >= 2 { beginMulti(pts.map(\.p)) }

        case .pending:
            if pts.isEmpty { cancelTimer(); tap(at: downPoint, button: .left); mode = .idle }
            else if pts.count >= 2 { cancelTimer(); beginMulti(pts.map(\.p)) }
            else {
                lastPoint = pts[0].p
                if dist(lastPoint, downPoint) > opts.tapSlop { cancelTimer(); beginDrag(); drag(to: lastPoint) }
            }

        case .dragging:
            if pts.isEmpty { post(.leftMouseUp, lastPoint, .left, clicks: clickCount); mode = .idle; return }
            let p = (pts.first { $0.id == primaryID } ?? pts[0]).p
            if p != lastPoint { drag(to: p) }

        case .multi:
            maxFingers = max(maxFingers, pts.count)
            if pts.isEmpty {
                if scrolling { scroll(dx: 0, dy: 0, phase: 4) }
                else if now - multiStart < opts.multiTapMax && maxFingers == 2 {
                    tap(at: startCentroid, button: .right)
                }
                mode = .idle
            } else if pts.count >= 2 {
                let c = centroid(pts.map(\.p))
                if !scrolling && dist(c, startCentroid) > opts.tapSlop {
                    scrolling = true
                    scroll(dx: 0, dy: 0, phase: 1)
                }
                if scrolling { scroll(dx: c.x - lastCentroid.x, dy: c.y - lastCentroid.y, phase: 2) }
                lastCentroid = c
            }
            // One finger left over from a 2-finger gesture: ignore until all lift.
        }
    }

    private func beginPending(_ id: Int, _ p: CGPoint) {
        mode = .pending
        primaryID = id
        downPoint = p; lastPoint = p; downTime = now
        post(.mouseMoved, p, .left)
        let delay = opts.longPressRightClick ? opts.longPress : opts.dragDelay
        let w = DispatchWorkItem { [weak self] in
            guard let self, self.mode == .pending else { return }
            if self.opts.longPressRightClick {
                self.tap(at: self.downPoint, button: .right)
                self.mode = .multi            // swallow the rest of this touch
                self.maxFingers = 99; self.scrolling = false; self.multiStart = 0
                self.startCentroid = self.downPoint; self.lastCentroid = self.downPoint
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

    private func beginMulti(_ pts: [CGPoint]) {
        mode = .multi
        maxFingers = pts.count
        multiStart = now
        scrolling = false
        scrollRemainder = .zero
        startCentroid = centroid(pts)
        lastCentroid = startCentroid
        post(.mouseMoved, startCentroid, .left)
    }

    private func cancelTouch() {
        cancelTimer()
        if mode == .dragging { post(.leftMouseUp, lastPoint, .left, clicks: clickCount) }
        if mode == .multi && scrolling { scroll(dx: 0, dy: 0, phase: 4) }
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

private func dist(_ a: CGPoint, _ b: CGPoint) -> Double { hypot(a.x - b.x, a.y - b.y) }
private func centroid(_ ps: [CGPoint]) -> CGPoint {
    let n = Double(ps.count)
    return CGPoint(x: ps.reduce(0) { $0 + $1.x } / n, y: ps.reduce(0) { $0 + $1.y } / n)
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
