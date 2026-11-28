import Cocoa
import SafariServices

/// First-run window explaining how to enable the Safari extension.
final class OnboardingWindowController: NSWindowController {
    private let status = NSTextField(labelWithString: "")
    private let onShowSidebar: () -> Void

    init(onShowSidebar: @escaping () -> Void) {
        self.onShowSidebar = onShowSidebar
        let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 480, height: 400),
                              styleMask: [.titled, .closable], backing: .buffered, defer: false)
        window.title = "Gemini Sidebar"
        window.isReleasedWhenClosed = false
        super.init(window: window)
        build()
    }

    required init?(coder: NSCoder) { fatalError() }

    func refresh() {
        SFSafariExtensionManager.getStateOfSafariExtension(withIdentifier: extensionBundleIdentifier) { state, error in
            let text: String
            let color: NSColor
            if let state, state.isEnabled {
                (text, color) = ("✓ The Safari extension is enabled.", .systemGreen)
            } else if state != nil {
                (text, color) = ("The Safari extension is turned off.", .systemOrange)
            } else {
                (text, color) = ("Safari can't see the extension yet — see step 1.", .systemOrange)
            }
            Task { @MainActor in
                self.status.stringValue = text
                self.status.textColor = color
            }
        }
    }

    private func build() {
        let icon = NSImageView(image: NSApp.applicationIconImage)
        icon.widthAnchor.constraint(equalToConstant: 72).isActive = true
        icon.heightAnchor.constraint(equalToConstant: 72).isActive = true

        let title = NSTextField(labelWithString: "Gemini Sidebar for Safari")
        title.font = .systemFont(ofSize: 20, weight: .semibold)
        status.font = .systemFont(ofSize: 13, weight: .medium)

        let steps = NSTextField(wrappingLabelWithString: """
            1. In Safari, choose Safari ▸ Settings ▸ Extensions and turn on “Gemini Sidebar”. \
            For a locally built copy, first enable Settings ▸ Advanced ▸ “Show features for web developers”, \
            then Develop ▸ “Allow Unsigned Extensions”.

            2. Click the ✦ toolbar button (or press ⌃⌥G anywhere) to open Gemini beside the page, \
            and sign in to Google once.

            3. Right-click a page or selected text for “Summarize Page with Gemini”, “Explain … with Gemini” \
            and more. Keyboard: ⌃⇧S summarizes, ⌃⇧A asks about the page.
            """)
        steps.font = .systemFont(ofSize: 13)
        steps.preferredMaxLayoutWidth = 420

        let openSettings = NSButton(title: "Open Safari Extension Settings", target: self, action: #selector(openSafariSettings))
        openSettings.bezelStyle = .push
        openSettings.keyEquivalent = "\r"
        let showSidebar = NSButton(title: "Show Sidebar", target: self, action: #selector(showSidebarClicked))
        showSidebar.bezelStyle = .push
        let buttons = NSStackView(views: [showSidebar, openSettings])

        let stack = NSStackView(views: [icon, title, status, steps, buttons])
        stack.orientation = .vertical
        stack.alignment = .centerX
        stack.spacing = 12
        stack.setCustomSpacing(20, after: steps)
        stack.edgeInsets = NSEdgeInsets(top: 24, left: 30, bottom: 24, right: 30)
        window?.contentView = stack
    }

    @objc private func openSafariSettings() {
        SFSafariApplication.showPreferencesForExtension(withIdentifier: extensionBundleIdentifier) { error in
            if let error { NSLog("%@", "Gemini Sidebar: could not open Safari settings: \(error)") }
        }
    }

    @objc private func showSidebarClicked() {
        close()
        onShowSidebar()
    }
}
