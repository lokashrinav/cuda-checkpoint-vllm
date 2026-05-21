.PHONY: gvisor-gpu-ckpt clean

gvisor-gpu-ckpt:
	CGO_ENABLED=1 go build -o bin/gvisor-gpu-ckpt ./cmd/gvisor-gpu-ckpt

clean:
	rm -rf bin/
