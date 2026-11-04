import CoreAudio
import Foundation

/// Lists audio output devices and switches the system default output.
enum AudioOutput {
    struct Device { let id: AudioDeviceID; let name: String; let isDefault: Bool }

    private static func address(_ selector: AudioObjectPropertySelector,
                                _ scope: AudioObjectPropertyScope = kAudioObjectPropertyScopeGlobal) -> AudioObjectPropertyAddress {
        AudioObjectPropertyAddress(mSelector: selector, mScope: scope, mElement: kAudioObjectPropertyElementMain)
    }

    static func defaultDeviceID() -> AudioDeviceID {
        var id = AudioDeviceID(0)
        var size = UInt32(MemoryLayout<AudioDeviceID>.size)
        var a = address(kAudioHardwarePropertyDefaultOutputDevice)
        AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &a, 0, nil, &size, &id)
        return id
    }

    static func outputDevices() -> [Device] {
        var a = address(kAudioHardwarePropertyDevices)
        var size: UInt32 = 0
        guard AudioObjectGetPropertyDataSize(AudioObjectID(kAudioObjectSystemObject), &a, 0, nil, &size) == noErr else { return [] }
        var ids = [AudioDeviceID](repeating: 0, count: Int(size) / MemoryLayout<AudioDeviceID>.size)
        guard AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &a, 0, nil, &size, &ids) == noErr else { return [] }
        let current = defaultDeviceID()
        return ids.compactMap { id in
            guard hasOutput(id), let name = name(id) else { return nil }
            return Device(id: id, name: name, isDefault: id == current)
        }
    }

    private static func hasOutput(_ id: AudioDeviceID) -> Bool {
        var a = address(kAudioDevicePropertyStreams, kAudioObjectPropertyScopeOutput)
        var size: UInt32 = 0
        return AudioObjectGetPropertyDataSize(id, &a, 0, nil, &size) == noErr && size > 0
    }

    private static func name(_ id: AudioDeviceID) -> String? {
        var a = address(kAudioObjectPropertyName)
        var cf: Unmanaged<CFString>?
        var size = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
        guard AudioObjectGetPropertyData(id, &a, 0, nil, &size, &cf) == noErr, let s = cf else { return nil }
        return s.takeRetainedValue() as String
    }

    /// Routes system sound (and the default output) to this device.
    @discardableResult
    static func setDefault(_ id: AudioDeviceID) -> Bool {
        var dev = id
        let size = UInt32(MemoryLayout<AudioDeviceID>.size)
        var ok = true
        for sel in [kAudioHardwarePropertyDefaultOutputDevice, kAudioHardwarePropertyDefaultSystemOutputDevice] {
            var a = address(sel)
            ok = AudioObjectSetPropertyData(AudioObjectID(kAudioObjectSystemObject), &a, 0, nil, size, &dev) == noErr && ok
        }
        return ok
    }
}
