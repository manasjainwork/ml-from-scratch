# ML from scratch

Machine learning pieces written from scratch with NumPy, with tests and plain-English comments. This is my public learning record for the [Vaani](https://github.com/manasjainwork/vaani) project.

Each piece is checked against known values or a trusted library before I move on.

| Piece | State |
|---|---|
| softmax, log-softmax, cross-entropy | Done |
| backpropagation (micrograd style) | Next |
| attention and KV-cache | Planned |
| CTC loss and beam search (for speech) | Planned |
| k-means, bandits, conformal prediction | Planned |

## Run the tests

```bash
uv sync
uv run pytest -q
```

## License

MIT. See [LICENSE](LICENSE).
