package hftypes

// clonePtr returns a copy of the pointee of ptr, or nil if ptr is nil.
func clonePtr[T any](ptr *T) *T {
	if ptr == nil {
		return nil
	}
	value := *ptr

	return &value
}

// cloneStructPtr returns a deep copy of the pointed-to struct, or nil if src is nil,
// using clone to produce the copy.
func cloneStructPtr[T any](src *T, clone func(*T) T) *T {
	if src == nil {
		return nil
	}
	dst := clone(src)

	return &dst
}

// cloneSlice returns a deep copy of src, or nil if src is nil, using clone on each element.
func cloneSlice[T any](src []T, clone func(*T) T) []T {
	if src == nil {
		return nil
	}
	dst := make([]T, len(src))
	for idx := range src {
		dst[idx] = clone(&src[idx])
	}

	return dst
}
