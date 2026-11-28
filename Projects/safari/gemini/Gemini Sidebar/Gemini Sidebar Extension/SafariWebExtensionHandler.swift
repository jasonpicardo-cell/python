import AppKit
import SafariServices
import os.log

/// Receives `browser.runtime.sendNativeMessage` calls from background.js and forwards them
/// to the container app (which hosts the sidebar) through its `geminisidebar://` URL scheme.
class SafariWebExtensionHandler: NSObject, NSExtensionRequestHandling {

    func beginRequest(with context: NSExtensionContext) {
        let request = context.inputItems.first as? NSExtensionItem
        let message = request?.userInfo?[SFExtensionMessageKey] as? [String: Any]

        guard let action = message?["action"] as? String,
              ["toggle", "show", "hide", "new", "ask"].contains(action) else {
            return reply(context, ["ok": false, "error": "unknown action"])
        }

        var components = URLComponents()
        components.scheme = "geminisidebar"
        components.host = action
        var query: [URLQueryItem] = []
        if let prompt = message?["prompt"] as? String { query.append(URLQueryItem(name: "prompt", value: prompt)) }
        if let send = message?["send"] as? Bool { query.append(URLQueryItem(name: "send", value: send ? "1" : "0")) }
        components.queryItems = query.isEmpty ? nil : query

        guard let url = components.url else { return reply(context, ["ok": false, "error": "bad url"]) }

        let configuration = NSWorkspace.OpenConfiguration()
        configuration.activates = true
        NSWorkspace.shared.open(url, configuration: configuration) { _, error in
            if let error {
                os_log(.error, "Gemini Sidebar: failed to open app: %{public}@", error.localizedDescription)
                self.reply(context, ["ok": false, "error": error.localizedDescription])
            } else {
                self.reply(context, ["ok": true])
            }
        }
    }

    private func reply(_ context: NSExtensionContext, _ payload: [String: Any]) {
        let response = NSExtensionItem()
        response.userInfo = [SFExtensionMessageKey: payload]
        context.completeRequest(returningItems: [response], completionHandler: nil)
    }
}
