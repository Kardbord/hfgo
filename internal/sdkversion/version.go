package sdkversion

import "fmt"

// Version is the current version of the hfgo SDK.
// This follows semantic versioning (semver.org).
const Version = "4.0.0-rc2"

// UserAgent returns the User-Agent string used for HTTP requests.
// The format is: hfgo/<version> (Go).
func UserAgent() string {
	return fmt.Sprintf("hfgo/%s (Go)", Version)
}
