// Package dto contains Data Transfer Objects for hfgo.
// They live here instead of the root module so the public providers package
// (same module) can reference DTO types without creating an import cycle.
// The root module re-exports these types by alias so downstream users keep
// importing from the root package.
package dto
