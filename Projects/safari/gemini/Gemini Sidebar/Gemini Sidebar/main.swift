import AppKit

// No storyboard: the app is a menu-bar utility that hosts the Gemini sidebar panel.
MainActor.assumeIsolated {
    let appDelegate = AppDelegate()
    NSApplication.shared.delegate = appDelegate
    NSApplication.shared.run() // never returns, so appDelegate stays alive
}
