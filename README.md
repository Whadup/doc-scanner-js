# doc-scanner-js

A high-performance, client-side document dewarping library for modern web applications. Powered by a small machine learning model running natively in the browser via **ONNX Runtime Web** and **OpenCV.js**.

## Features
- **Client-side Processing**: No server-side processing or data transmission.
- **High Fidelity**: Specialized geometric rectification model.
- **Optimized**: Choice of a lightweight 8.0MB model for speed or a 43.3MB model for maximum structural accuracy.
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
Copy `doc-scanner.js`, `react-hook.js`, and the `models/` folder into your project.

---

## Usage

The core `DocScanner` class provides the main functionality.

```javascript
import { DocScanner } from './doc-scanner.js';

const config = {
  seg: 'models/seg.onnx',
  geo: 'models/best.onnx'
};

const scanner = new DocScanner(config, { type: 'best' }); 
await scanner.init();

// Scan returns a high-level result object
const { canvas, blob } = await scanner.scan(imgElement, { 
  enhance: true, // apply scan enhancement
  useMask: true  // explicitly use segmentation mask
});
```

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `type` | `string` | `'best'` | `'fast'` (8.0MB model) or `'best'` (43.3MB model). |
| `inset` | `number` | `0` | Fraction of the image to inset from edges (e.g. `0.02` for 2%). |
| `enhance` | `boolean` | `false` | Apply classical scan enhancement (whitens background, improves contrast). |
| `useMask` | `boolean` | `false` | When using the `'fast'` engine, applies a segmentation mask before dewarping. |

### Diagnostic API

```javascript
// Get the 8-bit segmentation mask directly
const { canvas, blob } = await scanner.getMask(imgElement);
```

---

## React Usage

The easiest way to use the library is via the `useDocScanner` hook.

```jsx
import { useDocScanner } from './react-hook.js';

const config = {
  seg: 'models/seg.onnx',
  geo: 'models/best.onnx'
};

function App() {
  const { scan, isReady, isLoading, error } = useDocScanner(config, {
    type: 'best',
    enhance: true
  });

  const handleFile = async (e) => {
    const file = e.target.files[0];
    const img = await loadImage(file); // helper to create HTMLImageElement
    
    // Scan returns a high-level result object
    const { canvas, blob, dataUrl } = await scan(img, { enhance: true }); // or override per-scan
    
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

---

## CDN Usage (No Install)

You can use the library directly in the browser via **unpkg** or **jsDelivr**. Since the `.onnx` model is included in the package, you can load it directly from the CDN as well.

### 1. Simple Script Tag (ESM)
```html
<script type="module">
  // Load the library directly from jsDelivr (GitHub Proxy)
  import { DocScanner } from 'https://cdn.jsdelivr.net/gh/Whadup/doc-scanner-js@main/doc-scanner.js';

  const config = {
    seg: 'https://cdn.jsdelivr.net/gh/Whadup/doc-scanner-js@main/models/seg.onnx',
    geo: 'https://cdn.jsdelivr.net/gh/Whadup/doc-scanner-js@main/models/best.onnx'
  };

  const scanner = new DocScanner(config, { type: 'best' });
  await scanner.init();
  
  // ... scan logic ...
</script>
```

### 2. Using the React Hook via CDN
```javascript
import { useDocScanner } from 'https://cdn.jsdelivr.net/gh/Whadup/doc-scanner-js@main/react-hook.js';
```

---

An interactive drop-zone demo is structurally provided in [`index.html`](index.html).

Start a local server to view the demonstration:
```bash
python3 -m http.server 8081
```
