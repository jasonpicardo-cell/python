import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

/// One displayable item: a still image (1 frame) or an animation (GIF / APNG).
struct FrameSet: @unchecked Sendable {
    let images: [CGImage]
    let delays: [Double]          // seconds per frame, same count as images

    var first: CGImage { images[0] }
    var isAnimated: Bool { images.count > 1 }
    var totalDuration: Double { delays.reduce(0, +) }
}

/// Decodes any ImageIO-supported file straight down to a tiny W x H bitmap.
/// ImageIO's thumbnail path decodes at reduced size, so a 40 MP photo never sits fully in RAM.
enum PixelRenderer {
    static let maxAnimationFrames = 300

    /// First frame only (used by the exporter).
    static func render(url: URL, config: RenderConfig) -> CGImage? {
        renderFrames(url: url, config: config, animated: false)?.first
    }

    static func renderFrames(url: URL, config: RenderConfig, animated: Bool = true) -> FrameSet? {
        autoreleasepool { () -> FrameSet? in
            let srcOpts: [CFString: Any] = [kCGImageSourceShouldCache: false]
            guard let src = CGImageSourceCreateWithURL(url as CFURL, srcOpts as CFDictionary) else { return nil }
            let count = CGImageSourceGetCount(src)
            guard count > 0 else { return nil }

            let props = CGImageSourceCopyPropertiesAtIndex(src, 0, nil) as? [CFString: Any]
            var pw = (props?[kCGImagePropertyPixelWidth] as? NSNumber)?.intValue ?? 0
            var ph = (props?[kCGImagePropertyPixelHeight] as? NSNumber)?.intValue ?? 0
            let orientation = (props?[kCGImagePropertyOrientation] as? NSNumber)?.intValue ?? 1
            if orientation >= 5 && orientation <= 8 { swap(&pw, &ph) }
            guard pw > 0, ph > 0 else { return nil }

            let geo = Geometry(srcW: CGFloat(pw), srcH: CGFloat(ph), config: config)
            let frameCount = animated ? min(count, maxAnimationFrames) : 1

            var images: [CGImage] = []
            var delays: [Double] = []
            for i in 0..<frameCount {
                guard let img = renderFrame(src: src, index: i, geo: geo, config: config) else {
                    if i == 0 { return nil }
                    break
                }
                images.append(img)
                delays.append(frameDelay(src, index: i))
            }
            return images.isEmpty ? nil : FrameSet(images: images, delays: delays)
        }
    }

    /// Where the source lands inside the grid.
    private struct Geometry {
        let gridW: CGFloat, gridH: CGFloat
        let srcW: CGFloat, srcH: CGFloat
        let drawW: CGFloat, drawH: CGFloat
        let scale: CGFloat

        init(srcW: CGFloat, srcH: CGFloat, config: RenderConfig) {
            gridW = CGFloat(config.width)
            gridH = CGFloat(config.height)
            self.srcW = srcW
            self.srcH = srcH
            let sx = gridW / srcW
            let sy = gridH / srcH
            switch config.fit {
            case .fit:
                scale = min(sx, sy)
                drawW = min(gridW, max(1, (srcW * scale).rounded()))
                drawH = min(gridH, max(1, (srcH * scale).rounded()))
            case .fill:
                scale = max(sx, sy)
                drawW = max(gridW, (srcW * scale).rounded(.up))
                drawH = max(gridH, (srcH * scale).rounded(.up))
            case .stretch:
                scale = max(sx, sy)
                drawW = gridW
                drawH = gridH
            }
        }
    }

    private static func renderFrame(src: CGImageSource, index: Int, geo: Geometry, config: RenderConfig) -> CGImage? {
        // Decode at ~2x the final size (never upscale the source), then let CG downsample.
        let thumbScale = min(1.0, geo.scale * 2.0)
        let maxPixel = max(2, Int((max(geo.srcW, geo.srcH) * thumbScale).rounded(.up)))
        let thumbOpts: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceShouldCacheImmediately: true,
            kCGImageSourceThumbnailMaxPixelSize: maxPixel
        ]
        guard let thumb = CGImageSourceCreateThumbnailAtIndex(src, index, thumbOpts as CFDictionary) else {
            return nil
        }

        let space = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(
            data: nil,
            width: config.width,
            height: config.height,
            bitsPerComponent: 8,
            bytesPerRow: 0,
            space: space,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return nil }

        ctx.setFillColor(red: 0, green: 0, blue: 0, alpha: 1)
        ctx.fill(CGRect(x: 0, y: 0, width: geo.gridW, height: geo.gridH))
        // Small pixel-art sources stay crisp; photos get smooth downsampling.
        ctx.interpolationQuality = geo.drawW > CGFloat(thumb.width) ? .none : .high
        let rect = CGRect(
            x: ((geo.gridW - geo.drawW) / 2).rounded(.down),
            y: ((geo.gridH - geo.drawH) / 2).rounded(.down),
            width: geo.drawW,
            height: geo.drawH
        )
        ctx.draw(thumb, in: rect)
        return ctx.makeImage()
    }

    static func frameDelay(_ src: CGImageSource, index: Int) -> Double {
        guard let props = CGImageSourceCopyPropertiesAtIndex(src, index, nil) as? [CFString: Any] else { return 0.1 }
        let gif = props[kCGImagePropertyGIFDictionary] as? [CFString: Any]
        let png = props[kCGImagePropertyPNGDictionary] as? [CFString: Any]
        let raw = (gif?[kCGImagePropertyGIFUnclampedDelayTime] as? Double)
            ?? (gif?[kCGImagePropertyGIFDelayTime] as? Double)
            ?? (png?[kCGImagePropertyAPNGUnclampedDelayTime] as? Double)
            ?? (png?[kCGImagePropertyAPNGDelayTime] as? Double)
            ?? 0.1
        return raw < 0.02 ? 0.1 : raw          // browsers treat tiny delays as 100 ms
    }
}

/// Finds every image (any type ImageIO / UTType calls "public.image") in files and folders.
enum ImageScanner {
    static func scan(paths: [String]) -> [URL] {
        var result: [URL] = []
        var seen = Set<String>()
        let fm = FileManager.default

        for path in paths {
            var isDir: ObjCBool = false
            guard fm.fileExists(atPath: path, isDirectory: &isDir) else { continue }
            let url = URL(fileURLWithPath: path)

            if isDir.boolValue {
                guard let en = fm.enumerator(
                    at: url,
                    includingPropertiesForKeys: [.contentTypeKey, .isRegularFileKey],
                    options: [.skipsHiddenFiles, .skipsPackageDescendants]
                ) else { continue }
                for case let file as URL in en {
                    if isImage(file), seen.insert(file.path).inserted { result.append(file) }
                }
            } else if isImage(url), seen.insert(url.path).inserted {
                result.append(url)
            }
        }

        result.sort { $0.path.localizedStandardCompare($1.path) == .orderedAscending }
        return result
    }

    static func isImage(_ url: URL) -> Bool {
        guard let values = try? url.resourceValues(forKeys: [.contentTypeKey, .isRegularFileKey]) else {
            return false
        }
        if values.isRegularFile == false { return false }
        guard let type = values.contentType else { return false }
        return type.conforms(to: .image)
    }
}

/// RAM-resident LRU store of already-pixelated items (tens of KB each).
/// Capacity is counted in frames, so an animation costs one slot per frame.
final class FrameCache {
    private var store: [URL: (set: FrameSet, stamp: UInt64)] = [:]
    private var clock: UInt64 = 0
    private(set) var cost = 0
    var maxCount: Int = 1000 {
        didSet { evict() }
    }

    var count: Int { store.count }
    var isFull: Bool { cost >= maxCount }

    func get(_ url: URL) -> FrameSet? {
        guard var entry = store[url] else { return nil }
        clock += 1
        entry.stamp = clock
        store[url] = entry
        return entry.set
    }

    func contains(_ url: URL) -> Bool { store[url] != nil }

    func insert(_ set: FrameSet, for url: URL) {
        if let old = store[url] { cost -= old.set.images.count }
        clock += 1
        store[url] = (set, clock)
        cost += set.images.count
        evict(protecting: url)
    }

    func removeAll(keeping url: URL? = nil) {
        if let url, let keep = store[url] {
            store = [url: keep]
            cost = keep.set.images.count
        } else {
            store.removeAll()
            cost = 0
        }
    }

    /// Drops least-recently-used entries until under budget (always keeps `protecting`).
    private func evict(protecting url: URL? = nil) {
        while cost > maxCount, store.count > 1 {
            guard let victim = store.filter({ $0.key != url }).min(by: { $0.value.stamp < $1.value.stamp })?.key
            else { break }
            cost -= store[victim]?.set.images.count ?? 0
            store[victim] = nil
        }
    }
}
