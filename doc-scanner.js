/**
 * DocScanner — Browser-based document dewarping using ONNX Runtime Web + OpenCV.js
 */

const INPUT_WIDTH = 488;
const INPUT_HEIGHT = 712;

export class DocScanner {
    /**
     * @param {string} modelPath — URL or path to the ONNX model file
     * @param {object} [options]
     */
    constructor(modelPath, options = {}) {
        this.modelPath = modelPath;
        this.inputWidth = options.inputWidth || INPUT_WIDTH;
        this.inputHeight = options.inputHeight || INPUT_HEIGHT;
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
     * @returns {Promise<{ canvas: HTMLCanvasElement, debugCanvas: HTMLCanvasElement, blob: Blob, dataUrl: string }>}
     */
    async scan(imageSource) {
        if (!this.session) throw new Error("Call init() first");

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

        const resultRGBA = new cv.Mat();
        cv.cvtColor(result8u, resultRGBA, cv.COLOR_RGB2RGBA);
        result8u.delete();

        const canvas = document.createElement("canvas");
        canvas.width = origW;
        canvas.height = origH;
        cv.imshow(canvas, resultRGBA);

        const dataUrl = canvas.toDataURL("image/png");
        const blob = await new Promise(resolve => canvas.toBlob(resolve, "image/png"));

        // Cleanup
        src.delete();
        resultRGBA.delete();

        return { canvas, debugCanvas, blob, dataUrl };
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
