# Bruno API Collection

This project includes a [Bruno](https://www.usebruno.com/) collection used for
manually exercising the Hugging Face Inference API endpoints that the SDK
targets. It lives in [`tools/bruno/hfgo/`](../tools/bruno/hfgo/).

The collection is primarily useful to people developing this SDK: it lets you
send the same kinds of requests the SDK makes without writing Go code.

See the collection's own [README](../tools/bruno/hfgo/README.md)
for usage details.

## Cost notice

These requests hit the live Hugging Face Inference API and may incur charges on
your account, just like the integration tests.
