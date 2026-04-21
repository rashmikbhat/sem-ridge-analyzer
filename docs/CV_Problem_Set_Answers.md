### COMPUTER VISION PROBLEM SET - SEM IMAGE ANALYSIS

#### QUESTION 1: GAP (TRENCH) CALCULATION METHOD

**Implementation Strategy:**

I used a **ridge boundary-based projection approach**:
1. Compute a 1D vertical projection (summing pixel intensities column-wise) to collapse the 2D texture.
2. Isolate ridge regions using a relative peak threshold (25% of the maximum projection intensity).
3. Calculate the physical gap dynamically: `gap = next_ridge_start - current_ridge_end`.

I considered other approaches like using Canny edge detection to find ridge boundaries directly, or measuring peak-to-peak distances in the projection. However, edge detection struggled with the noisy ridge textures, and peak-to-peak only gives the pitch (ridge + gap), not the actual trench width.

**Why this method works well for SEM:**
- **Noise Tolerance:** Vertical projection naturally smooths over the internal granular texture (secondary electron noise) on the ridge surfaces, making it more robust compared to pure 2D Canny/Sobel edge detection.
- **Accurate Trench Measurement:** Unlike a peak-to-peak distance calculation (which only yields the pitch/period), this boundary method actually isolates the empty trench space.

---

#### QUESTION 2: ALGORITHM ASSUMPTIONS & ROBUSTNESS

**Current Pipeline Constraints & Assumptions:**

During robustness testing, I found two main assumptions in this traditional CV approach:

1. **Fixed Search Windows (Scale Dependency)**
   - *Assumption:* The vertical projection relies on a fixed ~150px search range, assuming relatively constant magnification/scale.
   - *Limitation:* The algorithm fails to segment properly if the image scale is doubled (2x zoom). I tried converting this to a dynamic percentage (e.g., `image_height * 0.30`) but that caused a cascade effect, requiring me to retune downstream thresholds.

2. **Uniform Background Illumination**
   - *Assumption:* The dark substrate background is relatively uniform.
   - *Limitation:* Deep black padding or severe electron beam charging (bright spots) messes up the global Histogram Equalization.
   - *Possible improvement:* Using masked equalization or localized CLAHE instead of global histogram equalization might help, but I didn't test this.

**Handling of Local Variations:**

Traditional CV pipelines require hardcoded tolerances. For example, I had to use a 15% Center of Mass (COM) offset to capture asymmetric ridge peaks. Also, localized illumination variations (like the darker second ridge) required manual threshold tuning (intensity 127 vs. 122). This is a known limitation of global thresholding - production systems would probably use adaptive thresholding (CLAHE).

**Robustness Testing (Augmentation Suite):**

To simulate real-world variations (stage tilt, defocus, electron beam noise), I tested the pipeline against 7 augmented edge cases:

*   **PASS (6/7):** Gaussian Blur (defocus), Brightness shift (-40%), Gaussian noise (static), Rotation (8° stage tilt), Horizontal flip, Position shift.
    *   *Note: The substrate baseline detection using RANSAC + gradient detection was pretty resilient to the 8° rotation.*
*   **FAIL (1/7):** 2x Magnification Scale (only detected 4/5 ridges due to the fixed pixel search window).

**Architectural Trade-offs: Traditional CV vs. Deep Learning**

This prototype works on 6 out of 7 test variations without needing any training data. However, the pipeline has a lot of interdependent hardcoded parameters (like the 25% projection threshold, 0.7 merge ratio, 4% gap tolerance). If I change one parameter, it breaks things downstream and I have to retune everything.

While this rule-based approach is great when you don't have labeled data, a production system would probably use Deep Learning instead (like U-Net). A CNN could learn the patterns from 50-100 annotated SEM images and wouldn't need all these manual thresholds. It would also handle different magnifications better since it learns the features instead of relying on fixed pixel ranges.

---

**Summary:**

The algorithm successfully detects all 5 ridges on the original image with accurate measurements. Robustness testing shows the pipeline generalizes to 6 out of 7 edge-case augmentations (detection only - I didn't validate measurement accuracy on transformed images).
