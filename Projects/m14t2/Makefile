APP      := build/M14tTouch.app
BIN      := $(APP)/Contents/MacOS/m14t-touch
SOURCES  := $(wildcard Sources/*.swift)
LABEL    := com.local.m14t-touch
AGENT    := $(HOME)/Library/LaunchAgents/$(LABEL).plist
INSTALL  := $(HOME)/Applications/M14tTouch.app
# Stable local identity keeps macOS permissions across rebuilds; falls back to ad-hoc.
SIGN_ID  := $(shell security find-identity -p codesigning 2>/dev/null | grep -q "M14tTouch Local Signing" && echo "M14tTouch Local Signing" || echo -)

.PHONY: all run dump install uninstall logs clean

all: $(BIN)

$(BIN): $(SOURCES) Resources/Info.plist
	mkdir -p $(APP)/Contents/MacOS
	cp Resources/Info.plist $(APP)/Contents/Info.plist
	swiftc -O -swift-version 5 -target arm64-apple-macos13 $(SOURCES) -o $(BIN)
	codesign --force --sign "$(SIGN_ID)" --identifier $(LABEL) $(APP)

run: $(BIN)
	$(BIN) --verbose

dump: $(BIN)
	$(BIN) --dump

install: $(BIN)
	mkdir -p $(HOME)/Applications $(HOME)/Library/LaunchAgents $(HOME)/Library/Logs
	launchctl bootout gui/$$(id -u)/$(LABEL) 2>/dev/null || true
	rm -rf $(INSTALL) && cp -R $(APP) $(INSTALL)
	sed "s|__BIN__|$(INSTALL)/Contents/MacOS/m14t-touch|; s|__LOG__|$(HOME)/Library/Logs/m14t-touch.log|" \
		Resources/launchagent.plist > $(AGENT)
	launchctl bootstrap gui/$$(id -u) $(AGENT)
	@echo "Installed. Grant M14tTouch both Input Monitoring and Accessibility (menu-bar icon → Open … Settings)."

uninstall:
	launchctl bootout gui/$$(id -u)/$(LABEL) 2>/dev/null || true
	rm -f $(AGENT)
	rm -rf $(INSTALL)

logs:
	tail -f $(HOME)/Library/Logs/m14t-touch.log

clean:
	rm -rf build
