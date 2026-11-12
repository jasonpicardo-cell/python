#!/bin/bash
# Creates a self-signed code-signing certificate "M14tTouch Local Signing" in your login
# keychain. Signing with a stable identity lets macOS keep the Input Monitoring and
# Accessibility permissions across rebuilds (ad-hoc signatures change on every build).
# Remove later with:  security delete-identity -c "M14tTouch Local Signing"
set -euo pipefail
NAME="M14tTouch Local Signing"
if security find-identity -p codesigning | grep -q "$NAME"; then echo "Already exists."; exit 0; fi
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
cat > "$TMP/cs.cnf" <<EOF
[req]
distinguished_name=dn
x509_extensions=ext
prompt=no
[dn]
CN=$NAME
[ext]
basicConstraints=critical,CA:false
keyUsage=critical,digitalSignature
extendedKeyUsage=critical,codeSigning
EOF
openssl req -x509 -newkey rsa:2048 -nodes -keyout "$TMP/key.pem" -out "$TMP/cert.pem" \
    -days 3650 -config "$TMP/cs.cnf" 2>/dev/null
openssl pkcs12 -export -inkey "$TMP/key.pem" -in "$TMP/cert.pem" -out "$TMP/id.p12" \
    -passout pass:tmp -name "$NAME" 2>/dev/null
security import "$TMP/id.p12" -k ~/Library/Keychains/login.keychain-db -P tmp -T /usr/bin/codesign
security find-identity -p codesigning | grep "$NAME"
echo "Done. Now run: make clean install"
