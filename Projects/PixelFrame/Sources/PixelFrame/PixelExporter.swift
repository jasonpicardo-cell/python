import AppKit
import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

/// Bakes the "LED panel" look into full-screen PNGs that macOS can use as a screen saver folder.
/// PNG is lossless, so the hard block edges and black grid lines never get JPEG/HEIC smearing.
enum PixelExporter {
    /// Renders one pixelated frame at `size` screen pixels, mirroring PixelNSView's layout.
    static func compose(frame: CGImage, size: (width: Int, height: Int), gap: Double, integerScale: Bool) -> CGImage? {
        let sw = CGFloat(size.width)
        let sh = CGFloat(size.height)
        let raw = min(sw / CGFloat(frame.width), sh / CGFloat(frame.height))
        let scale: CGFloat = (integerScale && raw >= 1) ? floor(raw) : raw
        let w = CGFloat(frame.width) * scale
        let h = CGFloat(frame.height) * scale
        let origin = CGPoint(x: ((sw - w) / 2).rounded(), y: ((sh - h) / 2).rounded())

        let space = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(
            data: nil, width: size.width, height: size.height,
            bitsPerComponent: 8, bytesPerRow: 0, space: space,
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        ) else { return nil }

        ctx.setFillColor(red: 0, green: 0, blue: 0, alpha: 1)
        ctx.fill(CGRect(x: 0, y: 0, width: sw, height: sh))
        ctx.interpolationQuality = .none
        ctx.draw(frame, in: CGRect(origin: origin, size: CGSize(width: w, height: h)))

        let lineWidth = (scale * CGFloat(gap)).rounded()
        if gap > 0.001, scale >= 3, lineWidth >= 1 {
            ctx.setFillColor(red: 0, green: 0, blue: 0, alpha: 1)
            for i in 0...frame.width {
                let x = origin.x + CGFloat(i) * scale - lineWidth / 2
                ctx.fill(CGRect(x: x.rounded(), y: origin.y, width: lineWidth, height: h))
            }
            for j in 0...frame.height {
                let y = origin.y + CGFloat(j) * scale - lineWidth / 2
                ctx.fill(CGRect(x: origin.x, y: y.rounded(), width: w, height: lineWidth))
            }
        }
        return ctx.makeImage()
    }

    static func pngData(_ image: CGImage) -> Data? {
        let data = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(data, UTType.png.identifier as CFString, 1, nil) else { return nil }
        CGImageDestinationAddImage(dest, image, nil)
        return CGImageDestinationFinalize(dest) ? data as Data : nil
    }

    static func writePNG(_ image: CGImage, to url: URL) -> Bool {
        guard let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else {
            return false
        }
        CGImageDestinationAddImage(dest, image, nil)
        return CGImageDestinationFinalize(dest)
    }
}

/// A display the user can export for (native pixel size, not points).
struct ExportTarget: Identifiable, Hashable {
    let id: String
    let name: String
    let width: Int
    let height: Int
    let scale: Double

    var title: String { "\(name)  ·  \(width) × \(height)" }

    static func available() -> [ExportTarget] {
        NSScreen.screens.enumerated().map { i, s in
            let scale = s.backingScaleFactor
            let w = Int((s.frame.width * scale).rounded())
            let h = Int((s.frame.height * scale).rounded())
            return ExportTarget(id: "\(s.localizedName)-\(w)x\(h)", name: s.localizedName.isEmpty ? "Display \(i + 1)" : s.localizedName, width: w, height: h, scale: Double(scale))
        }
    }
}
