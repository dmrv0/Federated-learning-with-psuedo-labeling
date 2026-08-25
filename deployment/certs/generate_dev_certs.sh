#!/usr/bin/env bash
# Self-signed CA + SuperLink leaf certificate for the TLS dev profile.
# ponytail: openssl self-signed pair, not a real PKI -- fine for a local/dev deployment smoke
# test; a production deployment needs certs from a real CA instead of this script.
set -euo pipefail

OUT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DAYS=825
SAN="${SUPERLINK_SAN:-DNS:localhost,IP:127.0.0.1}"

cd "$OUT_DIR"

openssl genrsa -out ca.key 4096
openssl req -x509 -new -nodes -key ca.key -sha256 -days "$DAYS" \
  -subj "/CN=ssfl-dev-ca" -out ca.crt

openssl genrsa -out server.key 4096
openssl req -new -key server.key -subj "/CN=ssfl-superlink" -out server.csr
openssl x509 -req -in server.csr -CA ca.crt -CAkey ca.key -CAcreateserial \
  -days "$DAYS" -sha256 -extfile <(printf "subjectAltName=%s" "$SAN") -out server.pem

rm -f server.csr ca.srl
chmod 600 ca.key server.key
echo "wrote ca.crt (root CA), server.pem + server.key (SuperLink TLS cert) to $OUT_DIR"
