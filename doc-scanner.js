export class DocScanner {
    /**
     * @param {string|object} modelConfig — URL/path to model or config object { main: path, seg: path, geo: path }
     * @param {object} [options]
     * @param {number} [options.inset=0] — Fraction of the image to inset from edges (e.g. 0.02 for 2%)
     * @param {boolean} [options.enhance=false] — Apply scan enhancement (background flattening + CLAHE)
     * @param {string} [options.type='best'] — 'fast' (UVDoc) or 'best' (DocTr)
     */
    constructor(modelConfig, options = {}) {
        this.type = options.type || "best";
        this.modelConfig = modelConfig;
        this.inset = options.inset || 0;
        this.sessions = {};
        this._cvReady = false;

        if (this.type === "fast") {
            this.inputWidth = options.inputWidth || 488;
            this.inputHeight = options.inputHeight || 712;
        } else if (this.type === "best") {
            this.inputWidth = options.inputWidth || 288;
            this.inputHeight = options.inputHeight || 288;
        }
    }

    /**
     * Initialize the ONNX session(s) and wait for OpenCV.js to be ready.
     */
    async init() {
        if (typeof cv === 'undefined' || !cv.Mat) {
            await this._waitForOpenCV();
        }
        this._cvReady = true;

        if (typeof this.modelConfig === "string") {
            // Single model path (usually UVDoc)
            this.sessions.main = await this._createSession(this.modelConfig);
        } else {
            // Multiple models (useful for DocTr geo + ill)
            for (const [name, path] of Object.entries(this.modelConfig)) {
                this.sessions[name] = await this._createSession(path);
            }
        }

        return this;
    }

    async _waitForOpenCV() {
        return new Promise((resolve) => {
            if (typeof cv !== 'undefined' && cv.onRuntimeInitialized) {
                const orig = cv.onRuntimeInitialized;
                cv.onRuntimeInitialized = () => {
                    orig();
                    resolve();
                };
            } else if (typeof cv !== 'undefined') {
                cv["onRuntimeInitialized"] = resolve;
            } else {
                // Poll if script not loaded yet
                const check = setInterval(() => {
                    if (typeof cv !== 'undefined' && cv.Mat) {
                        clearInterval(check);
                        resolve();
                    }
                }, 50);
            }
        });
    }

    async _createSession(path) {
        return await ort.InferenceSession.create(path, {
            executionProviders: ["wasm"],
            graphOptimizationLevel: "all",
        });
    }

    /**
     * Run the dewarping pipeline.
     * @param {HTMLImageElement|HTMLCanvasElement|HTMLVideoElement} imageSource
     * @param {object} [options] — Override instance options
     * @param {boolean} [options.useMask=false] — Explicitly use segmentation mask (for UVDoc)
     * @returns {Promise<{ canvas: HTMLCanvasElement, debugCanvas: HTMLCanvasElement, blob: Blob, dataUrl: string }>}
     */
    async scan(imageSource, options = {}) {
        if (Object.keys(this.sessions).length === 0) throw new Error("Call init() first");

        const inset = options.hasOwnProperty('inset') ? options.inset : this.inset;
        const enhance = options.enhance || false;
        const useMask = options.useMask || false;

        console.time("ScanTotal");
        console.log(`[DocScanner] Starting scan (Arch: ${this.type}, Enhance: ${enhance}, Inset: ${inset}, Mask: ${useMask})`);

        const src = cv.imread(imageSource);
        const origH = src.rows;
        const origW = src.cols;

        let result8u;
        let debugCanvas;

        if (this.type === "best") {
            const res = await this._scanBest(src, enhance);
            result8u = res.result8u;
            debugCanvas = res.debugCanvas;
        } else {
            const res = await this._scanFast(src, enhance, useMask);
            result8u = res.result8u;
            debugCanvas = res.debugCanvas;
        }

        const resultRGBA = new cv.Mat();
        cv.cvtColor(result8u, resultRGBA, cv.COLOR_RGB2RGBA);
        result8u.delete();

        // Optional Inset Crop
        let finalMat = resultRGBA;
        if (inset > 0) {
            console.log("[DocScanner] Applying final inset crop...");
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

        console.timeEnd("ScanTotal");
        console.log("[DocScanner] Scan complete.");

        return { canvas, debugCanvas, blob, dataUrl };
    }

    /**
     * Generates a segmentation mask for the image.
     * @param {HTMLImageElement|HTMLCanvasElement|HTMLVideoElement} imageSource 
     * @returns {Promise<{ canvas: HTMLCanvasElement, blob: Blob }>}
     */
    async getMask(imageSource) {
        if (!this.sessions.seg) throw new Error("Segmentation model (seg) not loaded in config");
        const src = cv.imread(imageSource);
        const { maskMat } = await this._getSegmentationMask(src);
        
        const canvas = document.createElement("canvas");
        cv.imshow(canvas, maskMat);
        const blob = await new Promise(r => canvas.toBlob(r, 'image/png'));
        
        src.delete(); maskMat.delete();
        return { canvas, blob };
    }

    async _scanFast(src, enhance, useMask) {
        const origH = src.rows;
        const origW = src.cols;

        const rgbSrc = new cv.Mat();
        cv.cvtColor(src, rgbSrc, cv.COLOR_RGBA2RGB);

        let activeSrc = rgbSrc;
        // Optional masking for UVDoc
        if (useMask && this.sessions.seg) {
            console.log("[DocScanner] Applying mask to UVDoc input...");
            const { maskMat } = await this._getSegmentationMask(src);
            const maskResized = new cv.Mat();
            cv.resize(maskMat, maskResized, rgbSrc.size());
            const masked = new cv.Mat();
            rgbSrc.copyTo(masked, maskResized);
            
            maskMat.delete();
            maskResized.delete();
            activeSrc = masked;
        }

        const floatSrc = new cv.Mat();
        activeSrc.convertTo(floatSrc, cv.CV_32FC3, 1.0 / 255.0);

        // Cleanup: only delete activeSrc if it's different from rgbSrc
        if (activeSrc !== rgbSrc) activeSrc.delete();
        rgbSrc.delete();

        const inputMat = new cv.Mat();
        cv.resize(floatSrc, inputMat, new cv.Size(this.inputWidth, this.inputHeight));
        const inputData = this._matToTensor(inputMat);
        inputMat.delete();

        const feeds = { input: new ort.Tensor("float32", inputData, [1, 3, this.inputHeight, this.inputWidth]) };
        const results = await this.sessions.main.run(feeds);
        const bmData = Object.values(results)[0].data;

        const gridH = 45;
        const gridW = 31;
        const debugCanvas = this._generateDebugCanvas(bmData, gridW, gridH);

        const mapX = new cv.Mat(origH, origW, cv.CV_32F);
        const mapY = new cv.Mat(origH, origW, cv.CV_32F);
        this._computeUvdocMaps(bmData, gridW, gridH, origW, origH, mapX, mapY);

        const dst = new cv.Mat();
        cv.remap(floatSrc, dst, mapX, mapY, cv.INTER_LINEAR, cv.BORDER_CONSTANT, new cv.Scalar(0, 0, 0, 0));
        mapX.delete(); mapY.delete(); floatSrc.delete();

        const result8u = new cv.Mat();
        dst.convertTo(result8u, cv.CV_8UC3, 255.0);
        dst.delete();

        if (enhance) this._enhanceScan(result8u);

        return { result8u, debugCanvas };
    }

    async _getSegmentationMask(src) {
        const rgb = new cv.Mat();
        cv.cvtColor(src, rgb, cv.COLOR_RGBA2RGB);
        const input288 = new cv.Mat();
        cv.resize(rgb, input288, new cv.Size(288, 288));
        const float288 = new cv.Mat();
        input288.convertTo(float288, cv.CV_32FC3, 1.0 / 255.0);
        
        const inputData = this._matToTensor(float288);
        const feedsSeg = { input: new ort.Tensor("float32", inputData, [1, 3, 288, 288]) };
        const resultsSeg = await this.sessions.seg.run(feedsSeg);
        const mskData = Object.values(resultsSeg)[0].data;

        // Convert to 8-bit mask
        const maskMat = new cv.Mat(288, 288, cv.CV_8UC1);
        for (let i = 0; i < 288 * 288; i++) {
            maskMat.data[i] = mskData[i] > 0.5 ? 255 : 0;
        }

        rgb.delete(); input288.delete(); float288.delete();
        return { maskMat, mskData, inputData };
    }

    async _scanBest(src, enhance) {
        const origH = src.rows;
        const origW = src.cols;

        const rgbSrc = new cv.Mat();
        cv.cvtColor(src, rgbSrc, cv.COLOR_RGBA2RGB);

        // 1. Segmentation
        console.log("[DocScanner] Running Best Segmentation...");
        const { maskMat, mskData, inputData } = await this._getSegmentationMask(src);

        // Apply Hard Mask
        const maskedData = new Float32Array(inputData.length);
        const spatialSize = 288 * 288;
        for (let c = 0; c < 3; c++) {
            for (let i = 0; i < spatialSize; i++) {
                const maskVal = mskData[i] > 0.5 ? 1.0 : 0.0;
                maskedData[c * spatialSize + i] = inputData[c * spatialSize + i] * maskVal;
            }
        }
        maskMat.delete();

        // 2. Geometric Transform
        console.log("[DocScanner] Running Best Geometric Dewarping...");
        const feedsGeo = { input: new ort.Tensor("float32", maskedData, [1, 3, 288, 288]) };
        if (!resultsGeo || !Object.values(resultsGeo)[0]) throw new Error("Geometric model failed to produce output");
        const bmData = Object.values(resultsGeo)[0].data;

        const gridSize = 288 * 288;
        console.log(`[DocScanner] bmData size: ${bmData.length}, target gridSize: ${2 * gridSize}`);
        
        // Normalize coordinates from [0, 287] pixels to normalized [-1, 1] range
        const bmDataNormalized = new Float32Array(bmData.length);
        for (let i = 0; i < bmData.length; i++) {
            bmDataNormalized[i] = ((bmData[i] / 287.0) * 2.0 - 1.0) * 0.99;
        }

        const bm0Mat = cv.matFromArray(288, 288, cv.CV_32F, bmDataNormalized.slice(0, gridSize));
        const bm1Mat = cv.matFromArray(288, 288, cv.CV_32F, bmDataNormalized.slice(gridSize, 2 * gridSize));

        // Smooth the mapping
        cv.blur(bm0Mat, bm0Mat, new cv.Size(3, 3));
        cv.blur(bm1Mat, bm1Mat, new cv.Size(3, 3));

        const debugCanvas = this._generateDebugCanvas(bmDataNormalized, 288, 288);

        const bm0Resized = new cv.Mat();
        const bm1Resized = new cv.Mat();
        cv.resize(bm0Mat, bm0Resized, new cv.Size(origW, origH), 0, 0, cv.INTER_LINEAR);
        cv.resize(bm1Mat, bm1Resized, new cv.Size(origW, origH), 0, 0, cv.INTER_LINEAR);
        bm0Mat.delete(); bm1Mat.delete();

        console.log(`[DocScanner] Generating coordinate maps (${origW}x${origH})...`);
        const mapX = new cv.Mat(origH, origW, cv.CV_32F);
        const mapY = new cv.Mat(origH, origW, cv.CV_32F);
        const alphaX = (origW - 1) / 2;
        const alphaY = (origH - 1) / 2;
        
        bm0Resized.convertTo(mapX, cv.CV_32F, alphaX, alphaX);
        bm1Resized.convertTo(mapY, cv.CV_32F, alphaY, alphaY);
        bm0Resized.delete(); bm1Resized.delete();

        console.log("[DocScanner] Executing remap...");
        const result8u = new cv.Mat();
        cv.remap(rgbSrc, result8u, mapX, mapY, cv.INTER_LINEAR, cv.BORDER_CONSTANT, new cv.Scalar(255, 255, 255, 255));
        mapX.delete(); mapY.delete(); rgbSrc.delete();

        // 3. Enhancement
        if (enhance) {
            console.log("[DocScanner] Running Enhancement...");
            this._enhanceScan(result8u);
        }

        return { result8u, debugCanvas };
    }



    _matToTensor(mat) {
        const inputData = new Float32Array(3 * mat.cols * mat.rows);
        const data = mat.data32F;
        for (let c = 0; c < 3; c++) {
            for (let y = 0; y < mat.rows; y++) {
                for (let x = 0; x < mat.cols; x++) {
                    inputData[c * mat.cols * mat.rows + y * mat.cols + x] = data[(y * mat.cols + x) * 3 + c];
                }
            }
        }
        return inputData;
    }



    _computeUvdocMaps(bmData, gridW, gridH, origW, origH, mapX, mapY) {
        const gridSize = gridH * gridW;
        const bm0Mat = cv.matFromArray(gridH, gridW, cv.CV_32F, bmData.slice(0, gridSize));
        const bm1Mat = cv.matFromArray(gridH, gridW, cv.CV_32F, bmData.slice(gridSize, 2 * gridSize));
        const bm0Resized = new cv.Mat(), bm1Resized = new cv.Mat();
        cv.resize(bm0Mat, bm0Resized, new cv.Size(origW, origH), 0, 0, cv.INTER_LINEAR);
        cv.resize(bm1Mat, bm1Resized, new cv.Size(origW, origH), 0, 0, cv.INTER_LINEAR);
        bm0Mat.delete(); bm1Mat.delete();
        const b0D = bm0Resized.data32F, b1D = bm1Resized.data32F, mXD = mapX.data32F, mYD = mapY.data32F;
        for (let i = 0; i < origH * origW; i++) {
            mXD[i] = ((b0D[i] + 1) / 2) * (origW - 1);
            mYD[i] = ((b1D[i] + 1) / 2) * (origH - 1);
        }
        bm0Resized.delete(); bm1Resized.delete();
    }

    _enhanceScan(mat) {
        this._flattenBackground(mat);
        const lab = new cv.Mat();
        cv.cvtColor(mat, lab, cv.COLOR_RGB2Lab);
        const channels = new cv.MatVector();
        cv.split(lab, channels);
        const lChannel = channels.get(0);
        const clahe = new cv.CLAHE(1.5, new cv.Size(8, 8));
        clahe.apply(lChannel, lChannel);
        clahe.delete();
        const lut = new cv.Mat(1, 256, cv.CV_8U);
        for (let i = 0; i < 256; i++) { lut.data[i] = i > 240 ? 255 : i; }
        cv.LUT(lChannel, lut, lChannel);
        lut.delete();
        channels.set(0, lChannel);
        cv.merge(channels, lab);
        cv.cvtColor(lab, mat, cv.COLOR_Lab2RGB);
        lab.delete(); channels.delete(); lChannel.delete();
    }

    _flattenBackground(mat) {
        const gray = new cv.Mat();
        cv.cvtColor(mat, gray, cv.COLOR_RGB2GRAY);
        const small = new cv.Mat();
        cv.resize(gray, small, new cv.Size(0, 0), 0.25, 0.25, cv.INTER_AREA);
        const blurSize = Math.max(21, Math.floor(small.cols / 10)) | 1;
        cv.medianBlur(small, small, blurSize);
        cv.GaussianBlur(small, small, new cv.Size(blurSize, blurSize), 0);
        const illumination = new cv.Mat();
        cv.resize(small, illumination, gray.size(), 0, 0, cv.INTER_LINEAR);
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
        gray.delete(); small.delete(); illumination.delete(); lab.delete(); channels.delete(); lChannel.delete(); normalizedL.delete();
    }

    _generateDebugCanvas(bmData, width, height) {
        const canvas = document.createElement("canvas");
        canvas.width = width; canvas.height = height;
        const ctx = canvas.getContext("2d");
        const imgData = ctx.createImageData(width, height);
        const gridSize = width * height;
        for (let i = 0; i < gridSize; i++) {
            imgData.data[i * 4 + 0] = Math.max(0, Math.min(255, ((bmData[i] + 1) / 2) * 255));
            imgData.data[i * 4 + 1] = Math.max(0, Math.min(255, ((bmData[gridSize + i] + 1) / 2) * 255));
            imgData.data[i * 4 + 2] = 128; imgData.data[i * 4 + 3] = 255;
        }
        ctx.putImageData(imgData, 0, 0);
        return canvas;
    }

    dispose() {
        for (const session of Object.values(this.sessions)) { session.release(); }
        this.sessions = {};
    }
}
