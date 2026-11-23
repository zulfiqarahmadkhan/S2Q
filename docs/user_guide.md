# S2Q user guide

S2Q finds cells in microscopy images, measures them, lets you correct the result by hand, trains your own models, and plots the measurements. The sidebar follows that order:

**Home → Segmentation → Correction → Training → Viewer → Plotting → Tools**

At the bottom of the sidebar:

- **Status:** whether a GPU, Cellpose and StarDist were found.
- **GPU | CPU:** GPU is the default. Choose CPU when the GPU is busy with another program. It applies from the next run on and is remembered.
- **Day / Night mode:** switches the colour theme.

---

## Home

Shortcuts to each task, a system status card, and your recent folders.

## Segmentation

1. **Input**
   - **Folder:** every image in a folder.
   - **Spreadsheet (xlsx):** an experiment sheet. Press **Parse**, then tick the wells, stages, channels and types you want.
2. **Output folder:** where masks (`<image>_cp_masks.tif`), Excel sheets and overlay images are written. The input folder is never changed.
3. **Pipeline**
   - **Run segmentation:** find the cells.
   - **Run label → ROI analysis:** measure every cell. This writes one Excel sheet per image and `SummarySheet/Summary.xlsx`.
   - **Segment on / Measure intensity in:** e.g. segment on the phase image (Ph) and measure fluorescence in the red image (R). The matching image is found by well, stage and type in the file name. The hint below shows how many pairs were found.
   - **Skip images that already have a mask:** resume an interrupted run.
   - **Remove cells smaller than … / touching the border:** optional clean-up before measuring.
4. **Model:** a dropdown of the pretrained models, then **Custom (my trained model)**. When Custom is chosen, a second row appears with your trained models (and **Browse** for one trained elsewhere).
   - **Cellpose:** `cpsam` (Cellpose-SAM). Cellpose 4 has only this one pretrained model; it replaces the older cyto / nuclei models. Leave the diameter at "Auto" to detect it automatically.
   - **StarDist:** `2D_versatile_fluo` (fluorescent nuclei), `2D_versatile_he` (H&E, colour images), `2D_paper_dsb2018`, `2D_demo`. StarDist finds nuclei (round objects); for whole cells use Cellpose.
   - **Advanced settings:** thresholds. **Reset to defaults** restores them.
5. **👁 Preview one image:** try the settings on one image and see the cells and the time per image.
6. **Start processing.** **Stop** finishes the current image first.

Each run also saves `run_settings_<date>.json` in the output folder. It records the model, thresholds, channels and whether the GPU or CPU was used.

When the run is done, **🖊 Correct masks** opens the correction page, and **📂 Open output folder** shows the results.

## Correction

Pick an image (or **Choose image…**) and press **Open editor**. The mask is found in the output folder automatically, or you can choose the mask yourself.
- **Images without masks:** a folder of images that have no masks yet also works. Its images are listed as **"(no mask yet)"**, and opening one starts an empty mask to draw on. That mask is saved in the mask folder.
- **Open a specific pair — or a whole folder:** open any image (**File…**); leave the mask empty to start a new one. Or press **Folder…** to pick a folder of images: the editor opens with all of them, and ◀ Previous / Next ▶ walks through them. The mask field can be a mask file or a **folder of masks**, matched to the images by file name; images without a mask start with an empty one, saved in that mask folder (or next to the images when none is given).

**Microscopy files.** Besides TIFF, PNG and JPG, the editor opens:
- ND2, LIF, CZI and LSM files;
- OME-TIFF;
- OME-Zarr: use **Zarr folder…**, because these images are folders.

For files with several series, time points, channels or Z-planes, a **Plane** panel appears:
- **Choose the plane:** series, time point, Z-plane or **Max projection**, and one channel or all channels in colour.
- **Masks per plane:** each series / time point / Z-plane has its own mask file, e.g. `movie_t3_cp_masks.tif`. Changing the time point saves the current mask and opens that time point's mask. **[** and **]** step through time.
- **Channels share a mask:** changing the channel keeps the same mask.

Press **F1** in the editor for every shortcut. The main tools:

| Key | Tool | What it does |
|---|---|---|
| P | Pencil | Paint a new cell, or add to the cell you start in. Hold **Ctrl** to paint over neighbours. |
| E | Eraser | Erase part of a cell |
| L | Outline | Draw around a cell; it is filled when you let go |
| W | Magic wand | Click a cell to fill it by brightness |
| A | AI add cell | Draw a box around (or click on) a missed cell, and Cellpose-SAM outlines it |
| D | Delete | Click a cell, or drag a rectangle to delete several |
| F | Fill holes | Click a cell (or use *Fill all holes*) |
| R | Area | Drag a rectangle (or tick *Freehand* and draw around a region): only the cells in that area are shown |
| I | Line profile | Drag a line: a separate window plots the intensity of every channel along it (see below) |

**Line profile window**

Draw a line with the Line profile tool and a separate window opens, with:
- **The plot:** one curve per channel. 1, 2 or 3 channels, all channels of a microscopy file, or Red / Green / Blue for colour images.
- **Line width:** how many pixels across the line are included. The plot shows their mean, and the band is drawn on the image.
- **Plot width / height:** the size of the plot on screen and when saved.
- **Options:**
  - **Normalise each channel (0–1):** compare channels of very different brightness.
  - **Shade cells:** grey bands where the line crosses cells of the mask.
  - **Legend:** inside the plot, outside on the right, below the plot (never covers the curves), or off.
  - **Channels:** untick a channel to hide its curve when curves overlap. The table and the CSV still contain all channels.
- **Table:** mean, min, max and **integrated intensity** for each channel. The integrated intensity is the sum of all pixel values inside the line; with a wider line it includes all pixels of the band.
- **Distance:** in µm when the file records the pixel size, otherwise in pixels.
- **Export:**
  - **Save plot…** saves a PNG, SVG, PDF or JPG on a white background, at the chosen size and 300 dpi.
  - **Export CSV…** saves every point of every channel, the cell label under each point, and the summary table.

The profile follows your edits and changes of time point. With the Line profile tool:
- **Move the line:** drag it by its middle. The cursor shows a move arrow over the line.
- **Adjust an end:** drag one of its two end points.
- **Delete it:** click on the line, or press **Clear line**.

The graph updates while you move the line. Closing the graph window keeps the line on the image, and moving or adjusting the line shows the graph again.

Other controls:

- **Undo / redo:** Ctrl+Z and Ctrl+Y.
- **Zoom and pan:** + and − to zoom, Space + drag to pan.
- **Cell information:** hover a cell to see its number, area and intensity. **Show cell numbers** labels all cells.
- **Colour:** RGB, R, G or B, with intensity, brightness and contrast sliders.
- **View (in the Mask panel, or V):**
  - **Image + mask**;
  - **Cells only:** the segmented area, with the background removed;
  - **Background only:** the cells removed;
  - **Image only**;
  - **Mask only**.
- **🗗 Compare:** opens a second window with *Image + mask*, *Image only*, *Cells only* and *Background only* side by side. It shows the same part of the image as the editor and follows zoom, scrolling and edits.
- **Area of interest:**
  - **Select:** after drawing an area with the Area tool, only the cells in it are shown (cells touching the edge count), and the rest of the image is dimmed.
  - **Buttons:** **Zoom to area**, **Clear**, and **Save area cells…** (saves those cells as a new mask).
  - **🤖 AI: add all missed cells in the area:** Cellpose‑SAM segments the area and adds every cell that is not in the mask yet. Dragging a box with the AI tool (A) does the same for the box; a click adds one cell. One Ctrl+Z undoes them all.
- **Layers**, like layers in Photoshop:
  - **Fixed layers:** layer 1 is the opened image and layer 2 the mask you edit, both shown by default. Untick them to hide them; with every layer unticked the screen is empty.
  - **Add:**
    - **+ Image…:** another channel or image.
    - **+ Mask…:** another mask, drawn as outlines.
    - **+ Time points:** the masks of the other time points of the same position, plus their images (tick to show).
  - **Arrange:** tick a layer to show it, and set its colour and strength. **▲ / ▼** change the drawing order.
  - **Edit this mask:** makes the selected mask layer the one you edit; the current mask becomes a layer.
  - **Move layer ← ↑ ↓ →:** slides the selected layer by the step (1 px by default; hold a button to keep moving) to line it up with the image, e.g. a drifted time point or channel. **Reset** puts it back; the line under the buttons shows how far it moved. Layer 1 (the image) is the fixed reference. Selecting layer 2 moves the mask you edit (an edit: Ctrl+Z undoes it, and it is saved with the mask). Measurements (📊, and Save when that layer is the one shown) use the layer where it is shown.
  - **📊 Measure mask on images…:** lays one mask (the edited mask or a mask layer) on layer 1 and the image layers you tick, and gives each cell's area, centre, mean and integrated intensity (sum of the original pixel values) in every image and channel. With **Subtract background** on, the median outside all cells is also removed (integrated − area × background, i.e. CTCF). The **Summary** tab totals each image / channel. **Save Excel…** writes the sheets *Per cell*, *Summary* and *Info* (full file names and settings); **Save CSV** writes the per-cell table. Typical use: segment the cells in one channel, then add the other channels or time points as layers and measure the same cells there.
- **Grow / shrink cells** (Tools group): set the pixels, then **⊕ Grow** or **⊖ Shrink**. Growing only uses background pixels, so neighbouring cells never merge (they meet half-way); shrinking removes pixels from each cell's edge. With an area selected only the cells in the area change. Ctrl+Z undoes it.
- **Clean-up:** remove small cells or cells touching the border.
- **Window and side panel:** the window has normal minimise / maximise buttons. **⛶ Full screen** (F11) fills the whole screen; F11 or Esc goes back. **□ Maximise** maximises or restores the window. **Side panel size S / M / L** sets the width of the right-hand panel; you can also drag the divider between the image and the panel. **⇥** (or Ctrl+B) hides the panel and Ctrl+B brings it back. The width is remembered.
- **Pixel size (pixel → µm):** tick **Show sizes in micrometres (µm)** and set **1 pixel = … µm**. When the image file records its pixel size (ImageJ-calibrated or OME-TIFF, microscopy files), it is filled in for you; **From file** restores that value. **From line…** calibrates from a line drawn with the Line tool over a scale bar or anything of known length: type its length in µm. With the box ticked, the cell info, the area of interest and the line profile show µm. The Excel sheet gets extra columns (Area µm², Perimeter, Feret's diameter, major and minor axis, centroid X/Y in µm, and the pixel size), and **Measure mask on images** adds Area (µm²) and the centre in µm. The pixel columns stay as they are. The value is remembered for next time.
- **Saving:** **Ctrl+S** saves in place. **◀ Previous / Next** moves through the folder and saves automatically. After saving, the image's Excel sheet is measured again. **Save as…** writes the mask, its Excel sheet and its ROI picture in the folder you choose. The Excel sheet measures the image you are showing: layer 1 when it is ticked, otherwise the top-most ticked image layer (so untick layer 1 and tick another image to measure the same cells there).

## Training

1. Choose **Cellpose** (uses the GPU) or **StarDist** (CPU on Windows).
2. **Data folder:** images with their masks (`<image>_cp_masks.tif`). Masks corrected in the editor work well as training data.
3. **Training % / Testing %** (80 / 20 by default; they always add up to 100). The testing images are picked at random, never used for training, and saved with the model, so testing later uses exactly the same images. The hint shows how many images go to each.
4. **Start from:**
   - **Cellpose:** `cpsam`, or **Custom (my trained model)** to keep improving one of yours.
   - **StarDist:** **From scratch** (a new network), one of the four pretrained models (fine-tuning), or **Custom (my trained model)**. When fine-tuning, the rays, grid and channels come from the starting model.
5. **Stop early when the model stops improving** (on by default, patience 20 epochs): if the loss has not improved for that many epochs, training stops and the best model so far is kept.
   - **Cellpose:** uses the training loss.
   - **StarDist:** uses the loss on its validation images, which come from the training part.
   - The test images are never used for this.
6. **Start training.** The progress bar moves after every training step. The loss is shown after every epoch and should go down, and the status line estimates the time left.
7. Testing:
   When training is done, **🧪 Test model** appears. It runs the model on the testing images and saves the results in a new folder: next to the model, or in **Test results folder** if you set one. Tick **Also test the starting model** to get a before / after comparison on the same images. The results include:
   - **`summary_metrics.csv`:** every metric with its 95 % confidence interval:
     - AP, precision, recall and F1 at IoU 0.5 and 0.75;
     - mAP 0.5–0.95;
     - panoptic quality;
     - pixel Dice, Jaccard, sensitivity and specificity;
     - pixel **AUROC** and AUPRC;
     - cell-count error and correlation.
   - **`per_image_metrics.csv`, `metrics_vs_iou_threshold.csv`, `roc_curve.csv`, `pr_curve.csv`, `matched_objects.csv`.**
   - **`figures/`:** PNG (300 dpi) and SVG figures:
     - ROC curve and precision-recall curve;
     - metrics vs IoU threshold;
     - count agreement and IoU histogram;
     - F1 per image;
     - model comparison;
     - training loss.
   - **`examples/`:** each test image with true outlines (green), predicted outlines (magenta) and missed cells (red).
   - **`README.txt`:** what each file means, plus a ready-to-adapt methods paragraph for a paper.
7. **Your trained models** lists everything you trained, with the train / test split and the test AP. **Use in Segmentation** selects a model there; **🧪 Test selected model** tests any listed model again.

## Viewer

Browse a folder of images and masks as thumbnails (tick **Include subfolders** to include them). Click a thumbnail to preview it, scroll to zoom and drag to pan. Use **← Previous / Next →** or the arrow keys to move between images.

## Plotting

- **Distributions (one sheet):** histograms and scatter plots of one Excel sheet. A summary sheet gives one bar per image.
- **Compare groups:** load a results folder, then choose a measurement and group by well, stage, channel, type, image or **condition**. Name conditions like `A6,A7=WT; A8-A10=mutant`.
  - **Plots:** box or violin.
  - **Tests:** Mann-Whitney, Welch t-test or Kruskal-Wallis, with corrected pairwise tests.
  - **Statistical unit:**
    - *Cells* treats every cell as a sample;
    - *Image means* uses one value per image, which is more honest when there are several images per group.
  - **Export table (CSV).**
- **Explore cells:** any measurement against any other, coloured by group, with optional log scales. Hover a point to see the cell; click it to open that cell in the correction editor.

**Save plot…** always saves on a white background (PNG, SVG or PDF), whatever the app theme.

### SRPlots (126 plot types)

The fourth tab draws any of 126 chart types from **any table** — a CSV / TSV / TXT file, an Excel sheet (pick the sheet), or S2Q's own measurement sheets. The types follow the list of the bioinformatics.com.cn plotting tools, grouped as: bar charts · line & area · pie, donut & proportions · distributions (box, violin, raincloud…) · scatter & correlation · heatmaps, clusters & PCA · clinical & statistics (KM survival, ROC, forest…) · sets & flows (Venn, UpSet, Sankey, chord) · transcriptome & enrichment (volcano, enrichment bubbles…) · genome & sequence (Manhattan, SNP density, motif logo, oncoplot…).

1. **Choose a plot.** The tab opens on picture cards of all plot types (18 per page, in the order of the original site; **Previous / 1 … 7 / Next** below the cards). Narrow the cards with the menu on the left (*ALL modules*, *Basic plot → pie / bar / line / scatter plot*, *Genome*, *Transcriptome*, *Epigenome*, *Clinical plot*, *Miscellaneous*) or type a keyword or number and press **Search**. Click a card to open that plot; **← All plots** goes back. The plot is drawn at once with its **example data**, which also shows the table layout it expects; **Save example as template…** saves that table to fill in with your own values.
   The plot (with size, **Draw** and **Save figure…**) is at the top; **Data**, **Columns** and **Options** are underneath. The preview has the true proportions of the saved figure.
2. **Load your table** with **Browse**. For each input of the plot (e.g. *Groups*, *Values*) pick a column under **Columns**; S2Q guesses them from left to right. Inputs marked *(optional)* can stay *(none)*; inputs with a list take several columns (tick them).
3. **Options:** the plot's own settings (e.g. SD / SEM, test, thresholds) and the common ones — title, axis labels, colours (Nature, Science, Lancet, colour-blind safe…), colour map, font size, legend position (or off) and grid lines. With **Redraw automatically** on, the preview follows every change.
4. **Save figure…** writes PNG, TIFF, SVG, PDF or JPG on white, at the **Save DPI** (300 by default) and the chosen size in inches.

**Text adapts to the size.** With **Fit text automatically** on (default), every plot adjusts its text to the chosen size, font size and data: crowded axis labels are turned (35° … 90°), long ones wrapped onto two lines and, only if they still do not fit, thinned to every 2nd / 3rd label; crowded y labels get a slightly smaller font; gene / point labels with pointer lines are pushed apart (also off the legend); labels inside bars shrink to the bar (or are hidden on very thin bars); value labels above bars get room by extending the axis; axis titles, panel titles and legends that are too long wrap, shrink or use fewer columns; text never leaves the figure. Switch it off to see the plot exactly as drawn.

Plots that stand for an analysis on the website (GO / pathway enrichment, miRanda target prediction, pathway map colouring) draw **your results table**; S2Q does not run those analyses or download databases.

## Tools

- **Image Enhancer:** background subtraction, denoising, contrast (CLAHE), sharpening and 2× upsampling, for one image or a folder.
- **Channel check:** a table of which channel images exist or are missing for every well and stage.
- **Pretrained models:** every Cellpose and StarDist pretrained model, what it is for, and whether it is downloaded. **Download missing models** fetches the rest. The installer already downloads them all, and a missing model is also downloaded the first time it is used.
- **Cellpose GUI:** opens the Cellpose program.
- **Verify installation:** lists packages and checks the GPU.

---

## Installing and updating

Nothing has to be installed first. If there is no conda, the installer installs Miniforge for you. The first run downloads several GB and takes 10–30 minutes; later starts take seconds.

- **Windows:** double-click `S2Q_Setup.exe`. It installs S2Q to `%USERPROFILE%\S2Q`, adds an **S2Q** desktop shortcut, sets everything up and starts the app. Use the shortcut afterwards. If Windows says "Windows protected your PC", click **More info → Run anyway** (the installer is not code-signed).
- **macOS:** unzip `S2Q_mac.zip`, drag **S2Q.app** to Applications and open it. The first time, macOS blocks it because it is not signed by Apple:
  - **macOS 15 and newer:** go to System Settings → Privacy & Security → **Open Anyway**.
  - **Older macOS:** right-click the app → **Open**.

  A Terminal window then does the setup; keep it open while S2Q runs.
- **Developers:** after changing the code, run `python build_installer.py` on Windows. It builds both `S2Q_Setup.exe` and `S2Q_mac.zip`.
- **GitHub:** `install_and_run.bat` / `install_and_run.sh` install from GitHub (needs Git).
- **PyTorch:** `install_torch.py` picks the build for your NVIDIA driver; see the README.

## Troubleshooting

- **The setup stops with "Required packages could not be installed":** usually a broken download. Run the installer (or the desktop shortcut) again; it continues where it stopped.

- **"Processing failed" with a strange pandas or Excel error after installing something:** restart S2Q. Packages changed while it was running.
- **Out of GPU memory:** close other GPU programs, switch to **CPU** in the sidebar, or process fewer or smaller images.
- **Everything is slow:** check the sidebar says "GPU: on". If it says "CPU only", run `python install_torch.py --check`.
- **A crash:** details are written to `s2q_crash.log` in the S2Q folder.
