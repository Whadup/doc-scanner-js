# ONNX Document Scanner (UVDoc)

A high-performance, client-side document dewarping library for modern web applications. Powered by **UVDoc** (State-of-the-Art) running natively in the browser via **ONNX Runtime Web** and **OpenCV.js**.

## Features
- **Client-side Processing**: No server-side processing or data transmission.
- **SOTA Fidelity**: Uses UVDoc for superior geometric rectification.
- **Optimized**: 7.6MB quantized INT8 model for fast web delivery.
- **Modern API**: ESM-first with a dedicated React Hook.

## Installation

### 1. Dependencies
Add the following scripts to your `index.html` (required for WASM acceleration):
```html
<!-- peer dependencies -->
<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js"></script>
<script src="https://docs.opencv.org/4.9.0/opencv.js"></script>
```

### 2. Copy the library
Copy `doc-scanner.js`, `react-hook.js`, and `doc-scanner-js.onnx` into your project.

---

## Usage in React

The easiest way to use the library is via the `useDocScanner` hook.

```jsx
import { useDocScanner } from './hooks/useDocScanner';

function App() {
  const { scan, isReady, isLoading, error } = useDocScanner('/models/doc-scanner-js.onnx');

  const handleFile = async (e) => {
    const file = e.target.files[0];
    const img = await loadImage(file); // helper to create HTMLImageElement
    
    // Scan returns a high-level result object
    const { canvas, blob, dataUrl } = await scan(img);
    
    console.log("Processed image:", dataUrl);
  };

  if (isLoading) return <div>Loading AI Model...</div>;
  if (error) return <div>Error: {error.message}</div>;

  return (
    <div>
      <input type="file" onChange={handleFile} disabled={!isReady} />
    </div>
  );
}
```

## Vanilla JavaScript (ESM)

```javascript
import { DocScanner } from './doc-scanner.js';

const scanner = new DocScanner("doc-scanner-js.onnx");
await scanner.init();
```
An interactive drop-zone demo is structurally provided in [`index.html`](index.html).

Start a local server to view the demonstration:
```bash
python3 -m http.server 8081
```
