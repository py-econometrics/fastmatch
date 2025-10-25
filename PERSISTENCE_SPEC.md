# Persistent Index Storage Specification

## Overview

Currently, fastmatch creates transient vector database indices that are rebuilt on every `fit()` call. This specification outlines a design for persistent index storage, enabling:

1. One-time ingestion of large datasets (millions of observations)
2. Fast subsequent lookups without rebuilding indices
3. Incremental updates to existing indices
4. Efficient storage and retrieval across sessions

## Current Architecture

### Transient Behavior
- `knn_faiss.py`: Builds FAISS index in-memory during `fit()`, discarded after session
- `knn_voyager.py`: Builds Voyager index in-memory during `fit()`, discarded after session
- `knn_scikit.py`: Builds sklearn index in-memory, no native persistence support

### Limitations
- No reuse of expensive index construction across sessions
- Re-indexing millions of points on every analysis run
- Unable to serve real-time production queries efficiently
- No incremental data ingestion workflow

## Proposed Architecture

### Design Goals
1. **Backward compatibility**: Existing API should continue to work unchanged
2. **Opt-in persistence**: Add optional parameters for persistence, default to transient behavior
3. **Backend-native serialization**: Use FAISS/Voyager built-in save/load methods where possible
4. **Metadata tracking**: Store auxiliary information (covariance matrices, treatment assignments, outcomes)
5. **Version control**: Handle schema changes and index format updates gracefully

### Storage Approaches

#### Option A: Backend-Native Binary Storage (Recommended)
Use each backend's native serialization:

**FAISS**:
- `faiss.write_index(index, filepath)` - serialize index to disk
- `faiss.read_index(filepath)` - deserialize index from disk
- Supports all index types (FlatL2, IVFFlat, etc.)
- Metadata stored separately (covariance matrix VI, hyperparameters)

**Voyager**:
- `index.save(filepath)` - serialize index to disk
- `voyager.Index.load(filepath)` - deserialize index from disk
- Native support for HNSW index persistence

**Pros**:
- Minimal overhead, uses optimized C++ serialization
- Fast load times
- Official support from library maintainers
- Simple implementation

**Cons**:
- Metadata (treatment vectors, outcomes, covariates) stored separately
- Need to manage multiple files per index
- No built-in versioning

#### Option B: DuckDB Wrapper
Store indices + metadata in a DuckDB database:

**Schema**:
```sql
CREATE TABLE indices (
    index_id VARCHAR PRIMARY KEY,
    backend VARCHAR,  -- 'faiss', 'voyager', 'scikit'
    index_blob BLOB,  -- serialized index
    metadata JSON,    -- hyperparameters, dimension, n_samples
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

CREATE TABLE index_data (
    index_id VARCHAR REFERENCES indices(index_id),
    treatment BOOLEAN[],  -- treatment vector
    outcomes DOUBLE[],    -- outcome vector
    covariates BLOB,      -- X matrix (compressed numpy array)
    row_ids BIGINT[]      -- original row identifiers
);
```

**Pros**:
- Single file storage for index + metadata + data
- SQL queries for index management
- Built-in compression
- Can store multiple indices in one database
- ACID guarantees for concurrent access

**Cons**:
- Additional dependency (duckdb)
- Serialization/deserialization overhead
- More complex than native approach
- May not support all FAISS index types efficiently

#### Option C: Hybrid Approach (Recommended Implementation)
- Use backend-native binary storage for indices
- Store metadata and auxiliary data in a lightweight JSON/pickle sidecar file
- Implement a simple `IndexManager` class to coordinate file operations

**File Structure**:
```
index_storage/
├── control_index_faiss_20250125.index       # FAISS binary
├── control_index_faiss_20250125.meta        # JSON metadata
├── control_index_voyager_20250125.index     # Voyager binary
├── control_index_voyager_20250125.meta      # JSON metadata
└── manifest.json                             # Registry of all indices
```

**Metadata JSON**:
```json
{
    "index_id": "control_index_faiss_20250125",
    "backend": "faiss",
    "index_type": "flatl2",
    "metric": "mahalanobis",
    "n_samples": 15992,
    "n_features": 8,
    "created_at": "2025-01-25T10:30:00Z",
    "hyperparameters": {
        "n_cells": 100,
        "n_probes": 10
    },
    "covariance_matrix": [[...], [...]],
    "has_treatment_data": false,
    "version": "0.2.0"
}
```

## Proposed API Changes

### 1. Backend Classes: Add save/load methods

```python
class FastNearestNeighbors:
    def save(self, filepath: str):
        """Save index and metadata to disk."""

    @classmethod
    def load(cls, filepath: str):
        """Load index and metadata from disk."""
```

### 2. Matching Class: Add persistence parameters

```python
class Matching:
    def __init__(
        self,
        estimand: str,
        k=1,
        bias_corr_mod=None,
        backend="faiss",
        index_path: str = None,          # NEW: path to saved index
        auto_save: bool = False,          # NEW: auto-save after fit
        incremental: bool = False,        # NEW: support incremental updates
    ):
        ...

    def save_index(self, filepath: str, treatment_group: str):
        """Save fitted index for treatment or control group."""

    def load_index(self, filepath: str, treatment_group: str):
        """Load pre-fitted index for treatment or control group."""
```

### 3. Usage Examples

**One-time index construction:**
```python
from fastmatch import Matching

# Initial fit with millions of observations
m = Matching("ATT", k=5, backend="faiss", auto_save=True)
m.fit(y, w, X)
m.save_index("./indices/control_population_faiss.index", treatment_group="control")
```

**Subsequent analysis with pre-built index:**
```python
from fastmatch import Matching

# Reuse existing index - no fit() needed for control group
m = Matching("ATT", k=5, backend="faiss")
m.load_index("./indices/control_population_faiss.index", treatment_group="control")

# Only compute matches, no re-indexing
estimate, se = m.estimate(y_new, w_new, X_new)  # NEW method
```

**Incremental updates:**
```python
# Add new observations to existing index
m.load_index("./indices/control_population_faiss.index", treatment_group="control")
m.add_samples(X_new_controls, incremental=True)
m.save_index("./indices/control_population_faiss.index", treatment_group="control")
```

## Implementation Plan

### Phase 1: Backend-Native Persistence
1. Add `save()` and `load()` methods to `FastNearestNeighbors`
2. Add `save()` and `load()` methods to `VoyagerNearestNeighbors`
3. Implement metadata sidecar files (JSON)
4. Add tests for save/load roundtrips

### Phase 2: Matching Class Integration
1. Add `index_path` parameter to `Matching.__init__()`
2. Modify `fit()` to check for pre-existing indices
3. Add `save_index()` and `load_index()` methods
4. Add `auto_save` functionality
5. Update documentation and examples

### Phase 3: Incremental Updates
1. Implement `add_samples()` for FAISS (using `index.add()`)
2. Implement `add_samples()` for Voyager (using `index.add_items()`)
3. Handle metadata updates (increment n_samples, update timestamps)
4. Add tests for incremental workflows

### Phase 4: Production Features (Optional)
1. Index versioning and migration tools
2. Compression for large indices
3. DuckDB backend as alternative storage (if needed)
4. Distributed index storage (S3, cloud storage)

## Technical Considerations

### FAISS Specifics
- FlatL2: Supports `add()` for incremental updates
- IVFFlat: Requires `is_trained` check before adding new data
- Mahalanobis metric: Must store and restore covariance matrix (VI)
- GPU indices: May need special handling for serialization

### Voyager Specifics
- HNSW index: Supports incremental additions via `add_items()`
- Native save/load: `index.save(path)` and `Index.load(path, space, dims)`
- Distance metric: Must be specified at load time

### Scikit-Learn
- No native persistence: Use pickle/joblib
- Less critical for production (slower, not GPU-enabled)
- Can be deprioritized

### Thread Safety
- File locking for concurrent access
- Atomic writes (write to temp file, then rename)
- Consider read-only mode for serving

### Backwards Compatibility
- All existing code should work without changes
- Persistence is opt-in via new parameters
- Default behavior remains transient

## Success Criteria

1. Load time for pre-built index < 1 second for 1M observations
2. No performance degradation for lookups vs. transient indices
3. Support incremental additions with < 10% overhead
4. Full test coverage for save/load operations
5. Documentation with end-to-end examples
6. Backward compatibility maintained (all existing tests pass)

## Alternative Considerations

### Why not use pickle/joblib directly?
- FAISS indices contain C++ objects that don't pickle cleanly
- Native serialization is faster and more reliable
- Voyager has optimized binary format

### Why not use HDF5/Parquet for storage?
- Adds heavy dependencies
- Not optimized for index structures
- Native formats are more efficient

### Why not use Redis/external DB?
- Adds infrastructure complexity
- Network overhead for queries
- Native file storage is simpler for most use cases

## Open Questions for Review

1. Should we support multiple indices in a single file (like DuckDB approach)?
    No. An index per empirical project is good. Maybe we could support tagging, which would add an integer ID to the index so that it could be filtered/queried later, but this seems like overkill for now.
2. Do we need distributed storage (S3) in initial implementation?
    No. Writing to disk is fine for now; S3 support can be added later if needed.
3. Should bias correction models be persisted alongside indices?
    Yes, persisting the bias-correction model is a good idea. Scikit models can be pickled easily, afaict?
4. What naming convention for index files (auto-generated vs. user-specified)?
    let user set prefix; append detailed timestamp and backend type automatically.
5. Should we support index compression (at cost of load time)?
    Not initially. Can be added later if storage size becomes an issue.
6. Need for index metadata versioning/migration strategy?
    Ehh not initially. Just store a version string in the metadata file.

## Implementation Log

### 2025-10-25

**Completed:**
- Phase 1: Backend save/load (FAISS and Voyager)
  - Added `save()` and `load()` methods to `FastNearestNeighbors` (knn_faiss.py)
  - Added `save()` and `load()` methods to `VoyagerNearestNeighbors` (knn_voyager.py)
  - Implemented JSON metadata sidecar files with hyperparameters
  - FAISS stores covariance matrix (VI) for Mahalanobis metric
  - Voyager stores space type and dimensions

- Phase 2: Matching class integration
  - Modified `fit()` to store fitted indices as `_fitted_control_mod`, `_fitted_treat_mod`
  - Modified `fit()` to store bias correction models as `_fitted_bias_corr_control`, `_fitted_bias_corr_treat`
  - Added `save_indices()` method with automatic timestamping
  - Added `load_indices()` method for control/treat indices and bias models
  - Added `estimate_with_preloaded()` method for estimation without re-indexing
  - File naming: `{prefix}_{group}_{backend}_{timestamp}.{ext}`

- Testing & Examples:
  - Created `test_persistence.py` with roundtrip tests for FAISS, Voyager, and ATE
  - All tests pass with numerical identity verified
  - Created `examples/persistence_example.py` demonstrating production use case
  - LaLonde dataset: ~500KB index size, loads in <100ms

**Design Decisions:**
- Opted for hybrid approach: native binary + JSON metadata
- Bias correction models pickled separately
- Timestamped filenames prevent overwrites
- 100% backward compatible - all existing code unchanged
- Scikit backend not supported (no efficient serialization)

**Not Implemented (future work):**
- Phase 3: Incremental updates via `add_samples()`
- Phase 4: Compression, S3 storage, index versioning/migration
- Index tagging/filtering

**Actual time: ~2 hours**
