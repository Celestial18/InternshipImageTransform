# Internship Image Transformation
This project takes input images and generates three retina-inspired outputs:

1. **Foveated image (remap + Gaussian pyramid)**  
   - center stays sharp (fovea)
   - periphery becomes progressively blurrier

2. **Sparse ganglion warp**
   - samples the input according to a ganglion-cell density model

3. **Sparse cone warp**
   - samples the input according to a cone density model

Outputs are saved to three separate folders.

---

## Requirements

Install dependencies:
```bash
pip install numpy opencv-python-headless scipy
