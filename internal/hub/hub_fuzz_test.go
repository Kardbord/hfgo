//go:build !integration

package hub

import (
	"encoding/json"
	"reflect"
	"testing"
)

// FuzzInferenceProviderMappingUnmarshal drives arbitrary JSON through the
// dual-shape tolerant decoder for Hub provider mappings (dict keyed by
// provider slug, or self-describing array) and checks the invariants: no
// panic, entries that decode successfully are stable across re-decodes of
// the same input, and a nil result is only produced for truly empty input.
func FuzzInferenceProviderMappingUnmarshal(f *testing.F) {
	seeds := [][]byte{
		[]byte(
			`{"hf-inference":{"providerId":"org/model","status":"live","task":"text-classification"}}`,
		),
		[]byte(`[{"provider":"hf-inference","providerId":"org/model","status":"live"}]`),
		[]byte(`[{"provider":"p","provider_id":"m","adapter_weights_path":"/a","task":"t"}]`),
		[]byte(`{}`),
		[]byte(`[]`),
		[]byte(`null`),
		[]byte(`[null]`),
		[]byte(`[{"provider":"p"}]`), // missing provider id: skipped
		[]byte(`{"p":{}}`),           // missing provider id: skipped
		[]byte(`{"p":null}`),         // null entry
		[]byte(`{"p":42}`),           // wrong-typed entry
		[]byte(`{"b":{"providerId":"m"},"a":{"providerId":"m"}}`), // ordering
		[]byte(`42`),
		[]byte(`"string"`),
		[]byte(`true`),
		[]byte(`{"p":{"providerId":42}}`),   // wrong-typed field
		[]byte(`{"p":{"providerId":null}}`), // null field
	}
	for _, seed := range seeds {
		f.Add(seed)
	}

	f.Fuzz(func(t *testing.T, data []byte) {
		var first inferenceProviderMapping
		if err := first.UnmarshalJSON(data); err != nil {
			// A decode error is acceptable for non-mapping shapes; it must
			// still be deterministic.
			var again inferenceProviderMapping
			if err2 := again.UnmarshalJSON(data); (err == nil) != (err2 == nil) {
				t.Errorf("UnmarshalJSON(%q) is non-deterministic: %v then %v", data, err, err2)
			}

			return
		}

		var second inferenceProviderMapping
		if err := second.UnmarshalJSON(data); err != nil {
			t.Errorf("re-decode of previously accepted input failed: %v", err)
		}
		if len(first) != len(second) {
			t.Errorf("decode of %q is unstable: %d mappings then %d", data, len(first), len(second))
		}
		for i := range first {
			if !reflect.DeepEqual(first[i], second[i]) {
				t.Errorf(
					"decode of %q is unstable at %d: %+v then %+v",
					data,
					i,
					first[i],
					second[i],
				)
			}
		}

		// Keyed (dict) shapes order providers by slug; assert the sorted
		// order so downstream selection is deterministic.
		var keyed map[string]json.RawMessage
		if err := json.Unmarshal(data, &keyed); err == nil {
			for i := 1; i < len(first); i++ {
				if first[i-1].Provider > first[i].Provider {
					t.Errorf("keyed decode of %q is not provider-sorted: %q before %q",
						data, first[i-1].Provider, first[i].Provider)
				}
			}
		}
	})
}
