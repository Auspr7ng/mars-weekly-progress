# Included third-party source snapshots

The launch bootstrap prioritizes this directory so replay is independent of a machine-specific editable installation.

| Component | Snapshot | License |
| --- | --- | --- |
| mjlab | Tested 1.4.0 source copied from the Week4 frozen framework; original project metadata in `mjlab_pyproject_toml` | [Apache-2.0](mjlab_LICENSE) |
| Viser | Tested 1.1.1 Python source, frontend source and prebuilt client, including scene drag lifecycle support | [MIT](viser_LICENSE) |

The package versions alone do not define the viewer contract. Its actual drag start/update/end API and prebuilt client are included and checked. The mjlab source snapshot is preserved; the release does not upgrade the existing simulation environment. Source-file hashes are recorded in the publication manifest. Third-party notices in the source files remain intact.

Original projects: [mjlab](https://github.com/mujocolab/mjlab), [Viser](https://github.com/nerfstudio-project/viser). Install the matching runtime dependencies from the tested environment specification; these source snapshots do not include Python, CUDA, or wheel binaries.
