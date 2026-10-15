package hfopts_test

import (
	"context"
	"fmt"

	"github.com/Kardbord/hfgo/v4/hfopts"
)

// ExampleWithToken demonstrates setting an API token on options.
func ExampleWithToken() {
	opts := hfopts.NewOptions().With(hfopts.WithToken("my-api-token"))
	fmt.Println(opts.Token)
	// Output: my-api-token
}

// ExampleWithModel demonstrates setting a model on options.
func ExampleWithModel() {
	opts := hfopts.NewOptions().With(hfopts.WithModel("mistral-7b"))
	fmt.Println(opts.Model)
	// Output: mistral-7b
}

// ExampleWithContext demonstrates attaching a context to options.
func ExampleWithContext() {
	type ctxKey string
	ctx := context.WithValue(context.Background(), ctxKey("key"), "value")

	opts := hfopts.NewOptions().With(hfopts.WithContext(ctx))
	fmt.Println(opts.Context().Value(ctxKey("key")))
	// Output: value
}
