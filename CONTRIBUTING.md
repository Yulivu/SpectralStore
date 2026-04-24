# Contributing

Thanks for your interest in SpectralStore.

The project is early-stage research software, so useful contributions include:

- synthetic data generators,
- compression algorithms,
- query-engine implementations,
- reproducible experiment configs,
- baseline wrappers,
- documentation and paper notes.

## Development

```bash
python -m pip install -e ".[dev]"
pytest
ruff check .
```

Please keep changes focused and include tests for behavior that affects public APIs.
