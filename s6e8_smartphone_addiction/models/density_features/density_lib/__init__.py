"""Shared code of the density-feature pipeline: column lists and loaders (base), the static block and nested target
encodings (features), the fold-wise feature assembler (assemble), smooth response fields and row-embedding PCs (extras),
the masked lattice language model (mlm) and the fold-wise exact-token response tables (completion).

Other folders can import it after putting the competition folder and this folder on sys.path:
    sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / 'models' / 'density_features'))
    from density_lib.assemble import assemble
"""
