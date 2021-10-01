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
  * [x] Integrate standard checkerboard & shifted checkerboard
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
* Change opencv-python install
  * Issue: https://stackoverflow.com/questions/52337870/python-opencv-error-current-thread-is-not-the-objects-thread
  * Potential solution (setup.py): https://stackoverflow.com/questions/68436434/how-to-instruct-pip-to-install-a-requirement-with-no-binary-flag
  * Don't want to fix opencv version yet (maybe after deployment, to ensure major version is 2 or 3), thus for now: 
    1) set up venv
    2) pip install -r requirements.txt
    3) grab a coffee, go for a walk...
    4) pip install -e .
    
    opencv-python --no-binary opencv-python # FIXME this only works for GUI (CLI then no longer works...)
    use system-wide opencv (with official pybindings instead, see setup scripts)
  
