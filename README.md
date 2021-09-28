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
  * [ ] Refactoring (pattern_specs is quite cluttered & uses ambiguous names - e.g. "grid")
* [x] Detect center marker
  * [x] Contour-based detection, estimate initial homography
  * [x] Support clipped center markers (near image borders)
  * [ ] Refactoring
* [ ] Correspondence search
  * [ ] Initial point correspondences
  * [ ] Refine matches via ICA
  * [ ] Refactoring
* [ ] UI
  * [ ] Image loading
  * [ ] Visualize marker & point matches 
  * [ ] Visualize calibration results
  * [ ] De-select "bad" images
  * [ ] Visualize camera/marker poses (3d plot)
  * [ ] Export calibration
  * [ ] Parallelization
* TODO Documentation
  * Conventions (`NxM` means number of squares, NOT internal corners)
