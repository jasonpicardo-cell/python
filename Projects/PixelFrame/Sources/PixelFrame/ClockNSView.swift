import AppKit
import QuartzCore
import SwiftUI

/// Clock drawn with Core Animation layers. The once-a-second tick only swaps one CATextLayer string,
/// so there is no SwiftUI diffing, layout or shadow re-rasterisation. Ticks stop entirely while idle/hidden.
final class ClockNSView: NSView {
    private let box = CALayer()
    private let text = CATextLayer()

    private var slot = 8
    private var fixedLine = false
    private var running = true
    private var timer: Timer?

    private var laidOutKey = ""          // config + text length the current layout was computed for

    private static let timeFormat = make("hh:mm:ss a")
    private static let dateFormat = make("dd-MMM-yyyy")
    private static let lineTimeFormat = make("hh:mm:ss")
    private static let lineDateFormat = make("dd.MMM.yy")
    private static let lineDayFormat = make("EEE")

    private static func make(_ format: String) -> DateFormatter {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.dateFormat = format
        return f
    }

    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        setup()
    }

    required init?(coder: NSCoder) {
        super.init(coder: coder)
        setup()
    }

    deinit { timer?.invalidate() }

    private func setup() {
        wantsLayer = true
        layer?.masksToBounds = false

        box.backgroundColor = NSColor.black.withAlphaComponent(0.55).cgColor
        box.shadowColor = NSColor.black.cgColor
        box.shadowOpacity = 0.8
        box.shadowRadius = 3
        box.shadowOffset = .zero
        box.actions = Self.noActions

        text.isWrapped = false
        text.truncationMode = .none
        text.actions = Self.noActions

        box.addSublayer(text)
        layer?.addSublayer(box)
    }

    private static let noActions: [String: CAAction] = [
        "contents": NSNull(), "bounds": NSNull(), "position": NSNull(), "frame": NSNull(),
        "string": NSNull(), "shadowPath": NSNull(), "cornerRadius": NSNull(), "hidden": NSNull()
    ]

    override func hitTest(_ point: NSPoint) -> NSView? { nil }

    // MARK: Configuration

    func configure(slot: Int, fixedLine: Bool, running: Bool) {
        let changed = slot != self.slot || fixedLine != self.fixedLine
        self.slot = slot
        self.fixedLine = fixedLine
        self.running = running
        if changed { render(force: true) }
        updateTimer()
    }

    override func setFrameSize(_ newSize: NSSize) {
        super.setFrameSize(newSize)
        render(force: true)
    }

    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        NotificationCenter.default.removeObserver(self, name: NSWindow.didChangeOcclusionStateNotification, object: nil)
        if let w = window {
            NotificationCenter.default.addObserver(self, selector: #selector(visibilityChanged),
                                                   name: NSWindow.didChangeOcclusionStateNotification, object: w)
        }
        viewDidChangeBackingProperties()
        render(force: true)
        updateTimer()
    }

    override func viewDidChangeBackingProperties() {
        super.viewDidChangeBackingProperties()
        text.contentsScale = window?.backingScaleFactor ?? 2
        box.contentsScale = text.contentsScale
    }

    @objc private func visibilityChanged() { updateTimer() }

    // MARK: Ticking

    private var shouldTick: Bool {
        guard running, let w = window else { return false }
        return w.occlusionState.contains(.visible)
    }

    private func updateTimer() {
        guard shouldTick else {
            timer?.invalidate()
            timer = nil
            return
        }
        guard timer == nil else { return }
        render(force: false)                                   // show the right time immediately on resume
        let next = Date(timeIntervalSinceReferenceDate: floor(Date.timeIntervalSinceReferenceDate) + 1)
        let t = Timer(fire: next, interval: 1, repeats: true) { [weak self] _ in self?.render(force: false) }
        t.tolerance = 0.05
        RunLoop.main.add(t, forMode: .common)
        timer = t
    }

    // MARK: Drawing

    private func attributedText(unit: CGFloat, timeSize: CGFloat, now: Date) -> NSAttributedString {
        let white = NSColor.white
        let para = NSMutableParagraphStyle()
        let h = slot % 3
        para.alignment = h == 0 ? .left : (h == 1 ? .center : .right)
        para.lineSpacing = timeSize * 0.08

        func font(_ size: CGFloat, _ weight: NSFont.Weight) -> NSFont {
            .monospacedSystemFont(ofSize: size, weight: weight)
        }
        let out = NSMutableAttributedString()
        if fixedLine {
            out.append(NSAttributedString(string: Self.lineTimeFormat.string(from: now), attributes: [
                .font: font(timeSize * 0.6, .bold), .foregroundColor: white,
                .kern: timeSize * 0.2, .paragraphStyle: para]))
            let date = Self.lineDateFormat.string(from: now) + " " + Self.lineDayFormat.string(from: now)
            out.append(NSAttributedString(string: date.uppercased(), attributes: [
                .font: font(timeSize * 0.3, .semibold), .foregroundColor: NSColor.gray, .paragraphStyle: para]))
        } else {
            out.append(NSAttributedString(string: Self.timeFormat.string(from: now) + "\n", attributes: [
                .font: font(timeSize, .bold), .foregroundColor: white, .paragraphStyle: para]))
            out.append(NSAttributedString(string: Self.dateFormat.string(from: now).uppercased(), attributes: [
                .font: font(timeSize * 0.5, .semibold), .foregroundColor: white, .paragraphStyle: para]))
        }
        return out
    }

    private func render(force: Bool) {
        let b = bounds
        guard b.width > 0, b.height > 0 else { return }
        let unit = min(b.width, b.height)
        let timeSize = max(16, unit * 0.075)
        let str = attributedText(unit: unit, timeSize: timeSize, now: Date())

        CATransaction.begin()
        CATransaction.setDisableActions(true)
        text.string = str

        // Monospaced text keeps the same width every second, so layout only reruns on config/size/length changes.
        let key = "\(slot)|\(fixedLine)|\(Int(b.width))x\(Int(b.height))|\(str.length)"
        if force || key != laidOutKey {
            laidOutKey = key
            layoutBox(str, unit: unit, timeSize: timeSize, in: b)
        }
        CATransaction.commit()
    }

    private func layoutBox(_ str: NSAttributedString, unit: CGFloat, timeSize: CGFloat, in b: CGRect) {
        let textSize = str.boundingRect(with: CGSize(width: 10_000, height: 10_000),
                                        options: [.usesLineFragmentOrigin]).size
        let tw = ceil(textSize.width), th = ceil(textSize.height)
        let padX = timeSize * 0.4, padY = timeSize * 0.25
        let size = CGSize(width: tw + padX * 2, height: th + padY * 2)
        let margin = unit * 0.04

        let h = slot % 3, v = slot / 3                          // v: 0 top, 1 centre, 2 bottom
        let x: CGFloat = h == 0 ? margin : (h == 1 ? (b.width - size.width) / 2 : b.width - size.width - margin)
        let y: CGFloat = v == 0 ? b.height - size.height - margin : (v == 1 ? (b.height - size.height) / 2 : margin)

        box.frame = CGRect(x: x.rounded(), y: y.rounded(), width: size.width, height: size.height)
        let radius = timeSize * 0.2
        box.cornerRadius = radius
        box.shadowPath = CGPath(roundedRect: box.bounds, cornerWidth: radius, cornerHeight: radius, transform: nil)
        text.frame = CGRect(x: padX, y: padY, width: tw, height: th)
    }
}

struct ClockDisplay: NSViewRepresentable {
    let slot: Int
    let fixedLine: Bool
    let running: Bool

    func makeNSView(context: Context) -> ClockNSView { ClockNSView(frame: .zero) }

    func updateNSView(_ nsView: ClockNSView, context: Context) {
        nsView.configure(slot: slot, fixedLine: fixedLine, running: running)
    }
}
