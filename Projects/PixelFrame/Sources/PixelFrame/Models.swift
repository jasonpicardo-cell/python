import CoreGraphics
import Foundation

/// How a source image is mapped onto the pixel grid.
enum FitMode: String, CaseIterable, Identifiable {
    case fit, fill, stretch

    var id: String { rawValue }

    var title: String {
        switch self {
        case .fit: return "Fit (keep ratio, black bars)"
        case .fill: return "Fill (crop to grid ratio)"
        case .stretch: return "Stretch"
        }
    }
}

/// A low-resolution "LED panel" size.
struct PixelGrid: Identifiable, Hashable {
    let width: Int
    let height: Int
    let note: String

    var id: String { "\(width)x\(height)" }
    var title: String { "\(width) × \(height)  ·  \(note)" }

    // 8:5 (16:10) grids match the MacBook Air panels (13.6" 2560x1664, 15.3" 2880x1864),
    // so blocks stay square, scale by whole points and fill ~95-98% of the screen.
    static let widescreen: [PixelGrid] = [
        PixelGrid(width: 48, height: 30, note: "chunky blocks"),
        PixelGrid(width: 64, height: 40, note: "retro"),
        PixelGrid(width: 96, height: 60, note: "balanced (recommended)"),
        PixelGrid(width: 128, height: 80, note: "detailed (recommended)"),
        PixelGrid(width: 160, height: 100, note: "fine"),
        PixelGrid(width: 240, height: 150, note: "very fine")
    ]

    // Divoom-style square panels (centred, black side bars on a widescreen display).
    static let square: [PixelGrid] = [
        PixelGrid(width: 32, height: 32, note: "mini"),
        PixelGrid(width: 64, height: 64, note: "Divoom 64"),
        PixelGrid(width: 128, height: 128, note: "Divoom max")
    ]

    static let all: [PixelGrid] = widescreen + square
    static let recommended: PixelGrid = widescreen[2]

    /// "Auto" follows the screen: square blocks of `cell` points that tile the display for any size or aspect ratio.
    static let autoID = "auto"

    static func auto(width: CGFloat, height: CGFloat, cell: Double) -> PixelGrid {
        let c = max(2, CGFloat(cell))
        let w = min(1024, max(16, Int((width / c).rounded(.down))))
        let h = min(1024, max(16, Int((height / c).rounded(.down))))
        return PixelGrid(width: w, height: h, note: "auto")
    }

    static func find(_ id: String) -> PixelGrid {
        all.first { $0.id == id } ?? recommended
    }
}

/// Everything the renderer needs to turn a file into a pixel frame.
struct RenderConfig: Equatable {
    let width: Int
    let height: Int
    let fit: FitMode
}
