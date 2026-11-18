# coordinode-openraft

[openraft](https://github.com/databendlabs/openraft), an advanced Raft
consensus implementation in Rust, from a commit of its upstream main branch,
published under this name for CoordiNode until an openraft release carries what
CoordiNode needs. The library keeps its name: depend on it as

```toml
openraft = { package = "coordinode-openraft", version = "0.10.1" }
```

and use it as `openraft`. Its source is upstream's, unchanged; see COPYRIGHT
for the exact commit. For openraft itself, its documentation and its releases,
go to the upstream project.

Licensed under the MIT License (LICENSE), one of the two licences openraft is
offered under.
