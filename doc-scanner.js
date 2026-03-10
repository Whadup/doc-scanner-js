/**
 * DocScanner — Browser-based document dewarping using ONNX Runtime Web + OpenCV.js
 */

const INPUT_WIDTH = 488;
const INPUT_HEIGHT = 712;

export class DocScanner {
    /**
     * @param {string} modelPath — URL or path to the ONNX model file
     * @param {object} [options]
     * @param {number} [options.inset=0] — Fraction of the image to inset from edges (e.g. 0.02 for 2%)
     * @param {boolean} [options.enhance=false] — Apply professional scan enhancement
     */
    constructor(modelPath, options = {}) {
        this.modelPath = modelPath;
        this.inputWidth = options.inputWidth || INPUT_WIDTH;
        this.inputHeight = options.inputHeight || INPUT_HEIGHT;
        this.inset = options.inset || 0;
        this.session = null;
        this._cvReady = false;
    }

    /**
     * Initialize the ONNX session and wait for OpenCV.js to be ready.
     */
    async init() {
        if (typeof cv === 'undefined') {
            throw new Error("OpenCV.js not found. Ensure it is loaded before calling init().");
        }

        if (!cv.Mat) {
            await new Promise((resolve) => {
                if (cv.onRuntimeInitialized) {
                    const orig = cv.onRuntimeInitialized;
                    cv.onRuntimeInitialized = () => {
                        orig();
                        resolve();
                    };
                } else {
                    cv["onRuntimeInitialized"] = resolve;
                }
            });
        }
        this._cvReady = true;

        this.session = await ort.InferenceSession.create(this.modelPath, {
            executionProviders: ["wasm"],
            graphOptimizationLevel: "all",
        });

        return this;
    }

    /**
     * Run the dewarping pipeline.
     * @param {HTMLImageElement|HTMLCanvasElement|HTMLVideoElement} imageSource
     * @param {object} [options] — Override instance options
     * @param {number} [options.inset]
     * @param {boolean} [options.enhance]
     * @returns {Promise<{ canvas: HTMLCanvasElement, debugCanvas: HTMLCanvasElement, blob: Blob, dataUrl: string }>}
     */
    async scan(imageSource, options = {}) {
        if (!this.session) throw new Error("Call init() first");

        const inset = options.hasOwnProperty('inset') ? options.inset : this.inset;
        const enhance = options.enhance || false; // apply scan enhancement pipeline

        const src = cv.imread(imageSource);
        const origH = src.rows;
        const origW = src.cols;

        // 1. Preprocess
        const rgbSrc = new cv.Mat();
        cv.cvtColor(src, rgbSrc, cv.COLOR_RGBA2RGB);

        const floatSrc = new cv.Mat();
        rgbSrc.convertTo(floatSrc, cv.CV_32FC3, 1.0 / 255.0);
        rgbSrc.delete();

        const inputMat = new cv.Mat();
        cv.resize(floatSrc, inputMat, new cv.Size(this.inputWidth, this.inputHeight));

        const inputData = new Float32Array(3 * this.inputWidth * this.inputHeight);
        const data = inputMat.data32F;
        for (let c = 0; c < 3; c++) {
            for (let y = 0; y < this.inputHeight; y++) {
                for (let x = 0; x < this.inputWidth; x++) {
                    inputData[c * this.inputWidth * this.inputHeight + y * this.inputWidth + x] =
                        data[(y * this.inputWidth + x) * 3 + c];
                }
            }
        }
        inputMat.delete();

        // 2. Inference
        const feeds = { input: new ort.Tensor("float32", inputData, [1, 3, this.inputHeight, this.inputWidth]) };
        const results = await this.session.run(feeds);
        const bmData = Object.values(results)[0].data;

        // 3. Postprocess Mapping
        const gridH = 45;
        const gridW = 31;
        const gridSize = gridH * gridW;

        const debugCanvas = this._generateDebugCanvas(bmData, gridW, gridH);

        const bm0Mat = cv.matFromArray(gridH, gridW, cv.CV_32F, bmData.slice(0, gridSize));
        const bm1Mat = cv.matFromArray(gridH, gridW, cv.CV_32F, bmData.slice(gridSize, 2 * gridSize));

        const bm0Resized = new cv.Mat();
        const bm1Resized = new cv.Mat();
        cv.resize(bm0Mat, bm0Resized, new cv.Size(origW, origH), 0, 0, cv.INTER_LINEAR);
        cv.resize(bm1Mat, bm1Resized, new cv.Size(origW, origH), 0, 0, cv.INTER_LINEAR);
        bm0Mat.delete();
        bm1Mat.delete();

        const mapX = new cv.Mat(origH, origW, cv.CV_32F);
        const mapY = new cv.Mat(origH, origW, cv.CV_32F);
        const b0D = bm0Resized.data32F;
        const b1D = bm1Resized.data32F;
        const mXD = mapX.data32F;
        const mYD = mapY.data32F;

        for (let i = 0; i < origH * origW; i++) {
            mXD[i] = ((b0D[i] + 1) / 2) * (origW - 1);
            mYD[i] = ((b1D[i] + 1) / 2) * (origH - 1);
        }
        bm0Resized.delete();
        bm1Resized.delete();

        // 4. Remap
        const dst = new cv.Mat();
        cv.remap(floatSrc, dst, mapX, mapY, cv.INTER_LINEAR, cv.BORDER_CONSTANT, new cv.Scalar(0, 0, 0, 0));
        mapX.delete();
        mapY.delete();
        floatSrc.delete();

        // 5. Output
        const result8u = new cv.Mat();
        dst.convertTo(result8u, cv.CV_8UC3, 255.0);
        dst.delete();

        // 6. Enhancement
        if (enhance) {
            this._enhanceScan(result8u);
        }

        const resultRGBA = new cv.Mat();
        cv.cvtColor(result8u, resultRGBA, cv.COLOR_RGB2RGBA);
        result8u.delete();

        // Optional Inset Crop
        let finalMat = resultRGBA;
        if (inset > 0) {
            const marginX = Math.floor(origW * inset);
            const marginY = Math.floor(origH * inset);
            const rect = new cv.Rect(marginX, marginY, origW - 2 * marginX, origH - 2 * marginY);
            finalMat = resultRGBA.roi(rect);
        }

        const canvas = document.createElement("canvas");
        canvas.width = finalMat.cols;
        canvas.height = finalMat.rows;
        cv.imshow(canvas, finalMat);

        const dataUrl = canvas.toDataURL("image/png");
        const blob = await new Promise(resolve => canvas.toBlob(resolve, "image/png"));

        // Cleanup
        src.delete();
        resultRGBA.delete();
        if (finalMat !== resultRGBA) finalMat.delete();

        return { canvas, debugCanvas, blob, dataUrl };
    }

    /**
     * Professional Scan Enhancement:
     * Combines morphological background flattening and CLAHE.
     * @private
     */
    _enhanceScan(mat) {
        // 1. Local Background Flattening (Multi-scale illumination correction)
        this._flattenBackground(mat);

        // 2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
        const lab = new cv.Mat();
        cv.cvtColor(mat, lab, cv.COLOR_RGB2Lab);

        const channels = new cv.MatVector();
        cv.split(lab, channels);
        const lChannel = channels.get(0);

        // Lower clipLimit (1.5) to avoid harsh artifacts and noise amplification
        const clahe = new cv.CLAHE(1.5, new cv.Size(8, 8));
        clahe.apply(lChannel, lChannel);
        clahe.delete();

        // Optional: Slight gamma compression to clean up paper noise
        // Map 245-255 to 255
        const lut = new cv.Mat(1, 256, cv.CV_8U);
        const lutData = lut.data;
        for (let i = 0; i < 256; i++) {
            if (i > 240) lutData[i] = 255;
            else lutData[i] = i;
        }
        cv.LUT(lChannel, lut, lChannel);
        lut.delete();

        channels.set(0, lChannel);
        cv.merge(channels, lab);
        cv.cvtColor(lab, mat, cv.COLOR_Lab2RGB);

        lab.delete();
        channels.delete();
        lChannel.delete();
    }

    /**
     * Flattens uneven lighting by normalizing against a smooth illumination map.
     * This avoids halos around text by using a large-scale estimation.
     * @private
     */
    _flattenBackground(mat) {
        const gray = new cv.Mat();
        cv.cvtColor(mat, gray, cv.COLOR_RGB2GRAY);

        // 1. Create a downsampled illumination map to ignore local details (text)
        const scale = 0.25;
        const small = new cv.Mat();
        cv.resize(gray, small, new cv.Size(0, 0), scale, scale, cv.INTER_AREA);

        // 2. Strong smoothing on the thumbnail
        const blurSize = Math.max(21, Math.floor(small.cols / 10)) | 1; // ~10% of width
        cv.medianBlur(small, small, blurSize % 2 === 0 ? blurSize + 1 : blurSize);
        cv.GaussianBlur(small, small, new cv.Size(blurSize, blurSize), 0);

        // 3. Upscale the smooth map back to original size
        const illumination = new cv.Mat();
        cv.resize(small, illumination, gray.size(), 0, 0, cv.INTER_LINEAR);

        // 4. Normalize the original image using division
        // result = original / illumination * 255
        const lab = new cv.Mat();
        cv.cvtColor(mat, lab, cv.COLOR_RGB2Lab);
        const channels = new cv.MatVector();
        cv.split(lab, channels);
        const lChannel = channels.get(0);

        const normalizedL = new cv.Mat();
        cv.divide(lChannel, illumination, normalizedL, 255);

        channels.set(0, normalizedL);
        cv.merge(channels, lab);
        cv.cvtColor(lab, mat, cv.COLOR_Lab2RGB);

        // Cleanup
        gray.delete();
        small.delete();
        illumination.delete();
        lab.delete();
        channels.delete();
        lChannel.delete();
        normalizedL.delete();
    }

    _generateDebugCanvas(bmData, width, height) {
        const canvas = document.createElement("canvas");
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext("2d");
        const imgData = ctx.createImageData(width, height);
        const gridSize = width * height;
        for (let i = 0; i < gridSize; i++) {
            imgData.data[i * 4 + 0] = Math.max(0, Math.min(255, ((bmData[i] + 1) / 2) * 255));
            imgData.data[i * 4 + 1] = Math.max(0, Math.min(255, ((bmData[gridSize + i] + 1) / 2) * 255));
            imgData.data[i * 4 + 2] = 128;
            imgData.data[i * 4 + 3] = 255;
        }
        ctx.putImageData(imgData, 0, 0);
        return canvas;
    }

    dispose() {
        if (this.session) {
            this.session.release();
            this.session = null;
        }
    }
}
