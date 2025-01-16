.PHONY: test-cuda test-metal test-cpu clippy-cuda clippy-metal clippy-cpu

test-metal:
	cargo test -r --manifest-path ug-core/Cargo.toml
	cargo test -r --manifest-path ug-metal/Cargo.toml

test-cuda:
	cargo test -r --manifest-path ug-core/Cargo.toml
	cargo test -r --manifest-path ug-cuda/Cargo.toml

test-cpu:
	cargo test -r --manifest-path ug-core/Cargo.toml

clippy-cpu:
	cargo clippy --tests --examples --manifest-path ug-core/Cargo.toml
	cargo clippy --tests --examples --manifest-path ug-llama/Cargo.toml

clippy-cuda:
	cargo clippy --tests --examples --manifest-path ug-core/Cargo.toml
	cargo clippy --tests --examples --manifest-path ug-cuda/Cargo.toml
	cargo clippy --tests --examples --manifest-path ug-llama/Cargo.toml --features cuda

clippy-metal:
	cargo clippy --tests --examples --manifest-path ug-core/Cargo.toml
	cargo clippy --tests --examples --manifest-path ug-metal/Cargo.toml
	cargo clippy --tests --examples --manifest-path ug-llama/Cargo.toml --features metal
