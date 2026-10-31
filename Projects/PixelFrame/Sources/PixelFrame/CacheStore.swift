import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

/// Describes the cache folder on disk. The slideshow and the wallpaper read only from here, never from the originals.
struct CacheManifest: Codable {
    struct Item: Codable {
        let name: String            // file stem shared by the grid frame and every wallpaper variant
        let source: String          // original file name, for display only
        var path: String? = nil     // original location; used only to preview setting changes
    }

    var signature: String           // settings the cache was built with; a mismatch means "recreate"
    var created: Date
    var gridWidth: Int
    var gridHeight: Int
    var gap: Double
    var integerScale: Bool
    var items: [Item]
    var diskBytes: Int64
}

/// Layout:  Cache/manifest.json · Cache/grid/<name>.png (tiny pixel frame, APNG if animated).
/// Slideshow and wallpaper both draw these frames from RAM; the LED gap and scaling are applied at draw time.
enum CacheStore {
    static let root: URL = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        .appendingPathComponent("PixelFrame", isDirectory: true)
    static let directory = root.appendingPathComponent("Cache", isDirectory: true)
    private static let building = root.appendingPathComponent("Cache.building", isDirectory: true)
    private static let legacyWallpapers = root.appendingPathComponent("Wallpapers", isDirectory: true)

    static func gridURL(name: String) -> URL {
        directory.appendingPathComponent("grid/\(name).png")
    }

    // MARK: Read

    static func loadManifest() -> CacheManifest? {
        guard let data = try? Data(contentsOf: directory.appendingPathComponent("manifest.json")),
              let m = try? JSONDecoder().decode(CacheManifest.self, from: data),
              !m.items.isEmpty else { return nil }
        return m
    }

    /// Decodes a cached grid file fully into memory (tiny bitmaps), one CGImage per animation frame.
    static func loadFrames(_ url: URL) -> FrameSet? {
        autoreleasepool { () -> FrameSet? in
            guard let src = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
            let count = min(CGImageSourceGetCount(src), PixelRenderer.maxAnimationFrames)
            guard count > 0 else { return nil }
            let opts: [CFString: Any] = [kCGImageSourceShouldCacheImmediately: true]
            var images: [CGImage] = []
            var delays: [Double] = []
            for i in 0..<count {
                guard let img = CGImageSourceCreateImageAtIndex(src, i, opts as CFDictionary) else { break }
                images.append(img)
                delays.append(PixelRenderer.frameDelay(src, index: i))
            }
            return images.isEmpty ? nil : FrameSet(images: images, delays: delays)
        }
    }

    // MARK: Write

    static func writeGrid(_ set: FrameSet, to url: URL) -> Bool {
        guard let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString,
                                                         set.images.count, nil) else { return false }
        if set.isAnimated {
            let loop = [kCGImagePropertyPNGDictionary: [kCGImagePropertyAPNGLoopCount: 0]]
            CGImageDestinationSetProperties(dest, loop as CFDictionary)
        }
        for (i, image) in set.images.enumerated() {
            let props: CFDictionary? = set.isAnimated
                ? [kCGImagePropertyPNGDictionary: [kCGImagePropertyAPNGDelayTime: set.delays[i]]] as CFDictionary
                : nil
            CGImageDestinationAddImage(dest, image, props)
        }
        return CGImageDestinationFinalize(dest)
    }

    /// Builds a brand-new cache in a scratch folder, then replaces the old one. Blocking: call from a background task.
    /// Returns nil when cancelled or nothing could be rendered (the previous cache is left untouched).
    static func build(urls: [URL], grid: RenderConfig, gap: Double, integerScale: Bool,
                      signature: String, progress: @Sendable (Double) -> Void) -> CacheManifest? {
        let fm = FileManager.default
        try? fm.removeItem(at: building)
        do {
            try fm.createDirectory(at: building.appendingPathComponent("grid"), withIntermediateDirectories: true)
        } catch { return nil }

        let digits = max(4, String(urls.count).count)
        var items: [CacheManifest.Item] = []
        for (i, url) in urls.enumerated() {
            if Task.isCancelled { try? fm.removeItem(at: building); return nil }
            let name = String(format: "%0\(digits)d", items.count + 1)
            let ok: Bool = autoreleasepool {
                guard let set = PixelRenderer.renderFrames(url: url, config: grid),
                      writeGrid(set, to: building.appendingPathComponent("grid/\(name).png")) else { return false }
                return true
            }
            if ok {
                items.append(.init(name: name, source: url.lastPathComponent, path: url.path))
            } else {                                                  // unreadable file: drop partial output, reuse the number
                try? fm.removeItem(at: building.appendingPathComponent("grid/\(name).png"))
            }
            progress(Double(i + 1) / Double(urls.count))
        }
        guard !items.isEmpty else { try? fm.removeItem(at: building); return nil }

        let manifest = CacheManifest(
            signature: signature, created: Date(),
            gridWidth: grid.width, gridHeight: grid.height, gap: gap, integerScale: integerScale,
            items: items, diskBytes: directorySize(building))
        guard let data = try? JSONEncoder().encode(manifest),
              (try? data.write(to: building.appendingPathComponent("manifest.json"))) != nil else {
            try? fm.removeItem(at: building)
            return nil
        }

        // Swap atomically: the old cache is deleted only once the new one is in place, so a failure keeps the old one.
        do {
            if fm.fileExists(atPath: directory.path) {
                try fm.replaceItemAt(directory, withItemAt: building)
            } else {
                try fm.moveItem(at: building, to: directory)
            }
        } catch {
            try? fm.removeItem(at: building)
            return nil
        }
        try? fm.removeItem(at: legacyWallpapers)                       // pre-cache versions wrote wallpapers here
        return manifest
    }

    static func delete() {
        try? FileManager.default.removeItem(at: directory)
        try? FileManager.default.removeItem(at: legacyWallpapers)
    }

    private static func directorySize(_ url: URL) -> Int64 {
        guard let en = FileManager.default.enumerator(at: url, includingPropertiesForKeys: [.fileSizeKey]) else { return 0 }
        var total: Int64 = 0
        for case let file as URL in en {
            total += Int64((try? file.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)
        }
        return total
    }
}
