# pycamcalib
Camera calibration UI

* Goal of this project: Replace our MATLAB calibration toolbox.
* Desired functionality:
  * Simple UI to load images & visualize calibration results
  * Option to skip images with large reconstruction error
  * Image enhancement
    * Apply to all images
    * Apply on per-image basis ("live fine-tuning")
* Feasibility:
  * Bouguet's toolbox is fully integrated in OpenCV, so we "only" need the marker detection and correspondence search.
  * Result plots (e.g. marker/camera poses) can be done with matplotlib 3d plots
  * Interactive 3d plots can be shown via Qt + matplotlib (create a custom widget)
* Caveats:
  * Correspondence search is obviously tricky
  * For sub-pixel accuracy, we need inverse compositional image alignment

## Status
* [ ] Use configurable template
  * [x] Render to SVG
  * [x] Render to PNG
  * [ ] Refactor eddie (pattern_specs is quite cluttered & uses ambiguous names - e.g. "grid")
* [x] Detect eddie center marker
  * [x] Contour-based detection, estimate initial homography
  * [x] Support clipped center markers (near image borders)
  * [ ] Refactoring
* [ ] Correspondence search for eddie
  * [ ] Initial point correspondences
  * [ ] Refine matches via FCA/ICA
  * [ ] Refactoring
* [ ] Extensions
  * [x] Integrate standard checkerboard & clipped/shifted checkerboard
  * [ ] Preprocessing submodule (histeq, binarization fixed/adaptive, normalization, cutoff bi-ended-slider, ...)
  * [ ] Each pattern should provide config widget (import/export config JSON or TOML, export SVG/PNG/PDF)
  * [ ] Each pattern should provide preconfigured calibration boards
  * [ ] Rethink extensions (each pattern submodule could provide a "Specification" and "Detector") - `pcc.patterns` could iterate submodules and create a table of known patterns
  * [ ] `pcc.patterns` should provide mechanism to register default patterns (e.g. A4 checkerboard, A0 something)
  * [ ] Each detector should be able to `visualize(image, img_pts)`, compute and visualize coverage (convex hull over list of img_pts)
* [ ] UI
  * [ ] Image loading
  * [ ] Visualize marker & point matches 
  * [ ] Visualize calibration results
  * [ ] De-select "bad" images
  * [ ] Visualize camera/marker poses (3d plot)
  * [ ] Export calibration
  * [ ] Parallelization
* UI elements
  * [x] Folder selection
  * [ ] Board selection (pre-configured vs custom)
  * [ ] Board config widget checkerboard std/shifted
  * [ ] Board config widget eddie
  * [ ] Preproc UI (list view)
  * [ ] image board/gallery (show original, preprocessed, detected, coverage)
  * [ ] matplotlib 3d plot
  * [ ] Export/save widget
* TODO Documentation
  * Conventions (`NxM` means number of squares, NOT internal corners)

