import AppKit
import QuartzCore
import SwiftUI

/// Layer-backed view. It only touches the GPU when the image changes (no display link, no redraw loop),
/// so an idle slideshow costs ~0% CPU/GPU between transitions.
final class PixelNSView: NSView {
    private let imageLayer = CALayer()
    private let gridLayer = CAShapeLayer()

    private var gridW = 0
    private var gridH = 0
    private var gap: CGFloat = 0.12
    private var integerScale = true
    private var lastImage: CGImage?
    private var lastItemID = -1

    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        setup()
    }

    required init?(coder: NSCoder) {
        super.init(coder: coder)
        setup()
    }

    private func setup() {
        wantsLayer = true
        layer?.backgroundColor = NSColor.black.cgColor

        imageLayer.magnificationFilter = .nearest   // hard pixel edges, no blur
        imageLayer.minificationFilter = .nearest
        imageLayer.contentsGravity = .resize
        imageLayer.isOpaque = true
        imageLayer.actions = [
            "contents": NSNull(), "bounds": NSNull(), "position": NSNull(), "frame": NSNull()
        ]

        gridLayer.fillColor = nil
        gridLayer.strokeColor = NSColor.black.cgColor
        gridLayer.masksToBounds = true
        gridLayer.actions = [
            "path": NSNull(), "lineWidth": NSNull(), "bounds": NSNull(),
            "position": NSNull(), "hidden": NSNull(), "frame": NSNull()
        ]

        layer?.addSublayer(imageLayer)
        layer?.addSublayer(gridLayer)
    }

    override func setFrameSize(_ newSize: NSSize) {
        super.setFrameSize(newSize)
        updateGeometry()
    }

    func update(image: CGImage?, itemID: Int, gap: Double, integerScale: Bool, fade: Bool) {
        var relayout = false
        if CGFloat(gap) != self.gap { self.gap = CGFloat(gap); relayout = true }
        if integerScale != self.integerScale { self.integerScale = integerScale; relayout = true }
        if let img = image, img.width != gridW || img.height != gridH {
            gridW = img.width
            gridH = img.height
            relayout = true
        }
        if relayout { updateGeometry() }

        if image !== lastImage {
            lastImage = image
            // Cross-fade only between items; animation frames of the same item swap instantly.
            let newItem = itemID != lastItemID
            lastItemID = itemID
            if fade, newItem, imageLayer.contents != nil, image != nil {
                let t = CATransition()
                t.type = .fade
                t.duration = 0.45
                t.timingFunction = CAMediaTimingFunction(name: .easeInEaseOut)
                imageLayer.add(t, forKey: "pixelFade")
            }
            imageLayer.contents = image
        }
    }

    private func updateGeometry() {
        let b = bounds
        guard b.width > 0, b.height > 0, gridW > 0, gridH > 0 else { return }

        let raw = min(b.width / CGFloat(gridW), b.height / CGFloat(gridH))
        let scale: CGFloat = (integerScale && raw >= 1) ? floor(raw) : raw
        let w = CGFloat(gridW) * scale
        let h = CGFloat(gridH) * scale
        let origin = CGPoint(x: ((b.width - w) / 2).rounded(), y: ((b.height - h) / 2).rounded())

        CATransaction.begin()
        CATransaction.setDisableActions(true)
        imageLayer.frame = CGRect(origin: origin, size: CGSize(width: w, height: h))
        gridLayer.frame = imageLayer.frame
        rebuildGrid(scale: scale, width: w, height: h)
        CATransaction.commit()
    }

    /// LED-panel look: thin black lines between pixels (static vector path, rebuilt only on resize/setting change).
    private func rebuildGrid(scale: CGFloat, width: CGFloat, height: CGFloat) {
        let lineWidth = scale * gap
        guard gap > 0.001, scale >= 3, lineWidth >= 0.5 else {
            gridLayer.isHidden = true
            gridLayer.path = nil
            return
        }
        let path = CGMutablePath()
        for i in 0...gridW {
            let x = CGFloat(i) * scale
            path.move(to: CGPoint(x: x, y: 0))
            path.addLine(to: CGPoint(x: x, y: height))
        }
        for j in 0...gridH {
            let y = CGFloat(j) * scale
            path.move(to: CGPoint(x: 0, y: y))
            path.addLine(to: CGPoint(x: width, y: y))
        }
        gridLayer.path = path
        gridLayer.lineWidth = lineWidth
        gridLayer.isHidden = false
    }
}

struct PixelDisplay: NSViewRepresentable {
    let image: CGImage?
    let itemID: Int
    let gap: Double
    let integerScale: Bool
    let fade: Bool

    func makeNSView(context: Context) -> PixelNSView {
        PixelNSView(frame: .zero)
    }

    func updateNSView(_ nsView: PixelNSView, context: Context) {
        nsView.update(image: image, itemID: itemID, gap: gap, integerScale: integerScale, fade: fade)
    }
}
