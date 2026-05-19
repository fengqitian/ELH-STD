# ELH-STD

MATLAB implementation of **ELH-STD: Explicit Local Hypergraph Prior-Guided Spatiotemporal Tensor Decomposition for Infrared Small Target Detection**.

This repository provides the core ELH-STD solver and a simple test script for generating binary infrared small target detection results.

## Repository Structure

```text
ELH-STD/
├── 1.zip                 # Example image sequence, optional
├── ELH_STD_Solver.m      # Core ELH-STD algorithm
├── test.m                # Test script
├── README.md             # Project description
└── LICENSE               # Open-source license
```

After extracting `1.zip`, the data folder should be organized as:

```text
ELH-STD/
├── 1/
│   └── images/
│       ├── 000001.png
│       ├── 000002.png
│       ├── 000003.png
│       └── ...
├── ELH_STD_Solver.m
└── test.m
```

## Requirements

- MATLAB R2020a or later is recommended
- Image Processing Toolbox

The code uses several MATLAB image-processing functions, including:

```matlab
imtophat
strel
imfilter
fspecial
padarray
mat2gray
bwareaopen
```

## Usage

1. Download or clone this repository.

2. Extract `1.zip`, or put your own infrared image sequence into:

```text
1/images/
```

3. Run the test script in MATLAB:

```matlab
test
```

4. The binary detection results will be saved in:

```text
1/binary_out/
```

Each output image corresponds to one input frame.

## Input Format

The input sequence should be stored as individual image files in:

```text
1/images/
```

Supported image formats include:

```text
.bmp, .png, .jpg, .jpeg, .tif, .tiff
```

The test script reads all images and stacks them into a 3-D tensor:

```matlab
Img_Seq ∈ R^(H × W × F)
```

where:

- `H` is the image height
- `W` is the image width
- `F` is the number of frames

## Output

The output binary detection maps are saved in:

```text
1/binary_out/
```

For example:

```text
000001_ELH_STD_binary.png
000002_ELH_STD_binary.png
000003_ELH_STD_binary.png
...
```

The output file name is generated from the corresponding input image name.

## Running Mode

The current `test.m` script uses the full infrared sequence as input:

```matlab
[Target_Tensor, Background_Tensor, out] = ELH_STD_Solver(Img_Seq, opts);
```

Then it generates binary detection maps for all frames:

```matlab
for k = 1:F
    SaliencyMap = Target_Tensor(:,:,k);
    ...
    imwrite(BinaryMap, outPath);
end
```

This means the solver is executed once for the whole sequence, and all frame-level binary maps are saved afterwards.

## Parameter Settings

The main ELH-STD parameters used in `test.m` are:

```matlab
opts.alphaTop4 = 0.5;
opts.seRadius = 9;
opts.d = 3;
opts.cellRadius = 1;

opts.alphaSigma = 0.35;
opts.alphaTemp  = 0.60;

opts.maxIter = 30;
opts.kappa = 0.05;
opts.useMotionPosterior = false;
opts.globalPruneRatio = 0.025;

opts.hgTauScale    = 1.10;
opts.hgBeta        = 10.0;
opts.hgProxyMix    = 0.10;
opts.hgSmoothSigma = 0.6;
opts.hgEps         = 1e-6;
```

The binary post-processing parameters are:

```matlab
k_thresh = 3.0;
border   = 10;
min_area = 2;
```

where:

- `k_thresh` controls the adaptive threshold level
- `border` removes boundary artifacts
- `min_area` removes very small isolated connected components

## Algorithm Overview

ELH-STD consists of two main stages.

### Stage 1: Explicit Local Hypergraph-Guided Prior Construction

The algorithm first constructs a target-sensitive spatiotemporal prior using:

- White top-hat transformation
- Top-4 directional local contrast response
- Explicit local hypergraph spatial response
- Short-term temporal max-min response

These components are fused to generate a prior-guided weight tensor.

### Stage 2: Prior-Guided Spatiotemporal Tensor Decomposition

The infrared sequence is modeled as:

```matlab
D = B + T
```

where:

- `D` is the observed infrared sequence tensor
- `B` is the low-rank background tensor
- `T` is the sparse target tensor

The model is optimized using ADMM. The background tensor is updated by a t-SVD-based tensor nuclear norm proximal operator, and the target tensor is updated by a dynamic reweighted sparse constraint.

## Main Function

```matlab
[Target_Tensor, Background_Tensor, out] = ELH_STD_Solver(Img_Seq, opts);
```

### Inputs

- `Img_Seq`: input infrared image sequence, with size `H × W × F`
- `opts`: parameter structure

### Outputs

- `Target_Tensor`: detected target tensor
- `Background_Tensor`: estimated background tensor
- `out`: intermediate results and diagnostic information

The `out` structure may include:

```matlab
out.I_top_seq
out.W_prior
out.S_top4_seq
out.S_hg_seq
out.T_map_seq
out.Joint_seq
out.err_hist
out.lambda
```

## Notes

- The current test script does not require ground-truth masks.
- The output is a binary detection map for each input frame.
- If the input sequence is very long or has high resolution, the runtime and memory usage may increase.
- For very large sequences, users may split the sequence into shorter clips and run the script separately.

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{zhu2026elhstd,
  title   = {Explicit local hypergraph prior-guided spatiotemporal tensor decomposition for infrared small target detection},
  author  = {Zhu, Zhuo and Jia, Minmin and Wu, Chengmao},
  journal = {Infrared Physics \& Technology},
  year    = {2026}
}
```

The BibTeX information will be updated after publication.

## License

This project is released under the MIT License.
