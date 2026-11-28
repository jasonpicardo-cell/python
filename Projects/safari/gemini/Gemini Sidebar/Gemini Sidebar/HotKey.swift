import Carbon.HIToolbox

/// System-wide keyboard shortcut via Carbon's RegisterEventHotKey (no Accessibility permission needed).
final class HotKey {
    private static var handlers: [UInt32: () -> Void] = [:]
    private static var nextID: UInt32 = 1
    private static var handlerInstalled = false
    private var ref: EventHotKeyRef?

    init(keyCode: UInt32, modifiers: UInt32, handler: @escaping () -> Void) {
        Self.installHandlerIfNeeded()
        let id = Self.nextID
        Self.nextID += 1
        Self.handlers[id] = handler
        let hotKeyID = EventHotKeyID(signature: OSType(0x474D_4E49) /* 'GMNI' */, id: id)
        let status = RegisterEventHotKey(keyCode, modifiers, hotKeyID, GetApplicationEventTarget(), 0, &ref)
        if status != noErr { NSLog("Gemini Sidebar: could not register hot key (\(status))") }
    }

    deinit {
        if let ref { UnregisterEventHotKey(ref) }
    }

    private static func installHandlerIfNeeded() {
        guard !handlerInstalled else { return }
        handlerInstalled = true
        var spec = EventTypeSpec(eventClass: OSType(kEventClassKeyboard), eventKind: UInt32(kEventHotKeyPressed))
        InstallEventHandler(GetApplicationEventTarget(), { _, event, _ in
            var hotKeyID = EventHotKeyID()
            GetEventParameter(event, EventParamName(kEventParamDirectObject), EventParamType(typeEventHotKeyID),
                              nil, MemoryLayout<EventHotKeyID>.size, nil, &hotKeyID)
            let id = hotKeyID.id
            // Carbon delivers hot key events on the main thread.
            MainActor.assumeIsolated { HotKey.handlers[id]?() }
            return noErr
        }, 1, &spec, nil, nil)
    }
}
