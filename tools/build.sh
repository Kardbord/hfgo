#!/usr/bin/env bash

# NOTE: Configuration options may differ between development and CI.
# This is a dev utility script, and should not be run in CI.

set -e

# Check that all prerequisite tools are installed before doing any work.
check_dependencies() {
  local missing=0
  for tool in "$@"; do
    if ! command -v "${tool}" &>/dev/null; then
      echo "ERROR: prerequisite not found on PATH: ${tool}" >&2
      missing=1
    fi
  done
  return "${missing}"
}

# gotestsum is optional; the script falls back to plain go test when absent.
check_dependencies go gofmt actionlint golangci-lint govulncheck

pushd "$(dirname "${BASH_SOURCE[0]}")/.." >/dev/null

export GOCACHE=/tmp/hfgo-cache

TEST_CMD="go test"
if command -v gotestsum &>/dev/null; then
  TEST_CMD="gotestsum --format-icons hivis --format pkgname-and-test-fails --"
fi

RUN_INTEGRATION_TESTS=
RUN_FUZZ_TESTS=
while getopts "if" opt; do
  case "${opt}" in
    i) RUN_INTEGRATION_TESTS="true" ;;
    f) RUN_FUZZ_TESTS="true" ;;
    *)
      echo "Usage: ${0} [-i] [-f]" >&2
      exit 2
      ;;
  esac
done

echo "Formatting code..."
gofmt -s -w .
echo

echo "Tidying module files..."
go mod tidy
echo

echo "Vetting..."
go vet ./...
echo

echo "Linting..."
actionlint
golangci-lint config verify
golangci-lint run --fix --disable godox ./...
echo

echo "Checking for vulnerabilities..."
govulncheck ./...
echo

echo "Building..."
go build ./...
echo

echo "Running unit tests..."
${TEST_CMD} -shuffle on ./...
echo

echo "Checking for race conditions..."
${TEST_CMD} -shuffle on -race ./...
echo

if [[ "${RUN_FUZZ_TESTS}" = "true" ]]; then
  echo "Running fuzz tests..."
  find . -name '*_fuzz_test.go' -exec dirname {} \; | sort -u | while read -r dir; do
    pkg=$(go list -f '{{.ImportPath}}' "$dir" 2>/dev/null)
    [ -z "$pkg" ] && continue
    go test -list '^Fuzz' "$pkg" 2>/dev/null | grep '^Fuzz' | while read -r target; do
      echo "Fuzzing ${target} in ${pkg}..."
      ${TEST_CMD} -run=^$ -fuzz="^${target}$" -fuzztime=5s -timeout=30s -count=1 "$pkg"
    done
  done
  echo
fi

if [[ "${RUN_INTEGRATION_TESTS}" = "true" ]]; then
  echo "Running integration tests..."
  ${TEST_CMD} -shuffle on -tags integration ./...
  echo
fi

echo "Reporting code coverage..."
${TEST_CMD} -shuffle on -cover ./...
echo

popd >/dev/null
