import { useState, useEffect, useCallback, useRef } from 'react';
import { DocScanner } from './doc-scanner.js';

/**
 * useDocScanner Hook
 * 
 * @param {string} modelPath - Path to the ONNX model
 * @param {object} options - Scanner options
 * @returns {object} { scan, isReady, isLoading, error }
 */
export function useDocScanner(modelPath, options = {}) {
    const [isReady, setIsReady] = useState(false);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState(null);
    const scannerRef = useRef(null);

    useEffect(() => {
        let isMounted = true;
        const scanner = new DocScanner(modelPath, options);
        scannerRef.current = scanner;

        const init = async () => {
            try {
                setIsLoading(true);
                await scanner.init();
                if (isMounted) {
                    setIsReady(true);
                    setIsLoading(false);
                }
            } catch (err) {
                if (isMounted) {
                    setError(err);
                    setIsLoading(false);
                }
            }
        };

        init();

        return () => {
            isMounted = false;
            scanner.dispose();
            scannerRef.current = null;
        };
    }, [modelPath, JSON.stringify(options)]);

    const scan = useCallback(async (source, scanOptions = {}) => {
        if (!scannerRef.current || !isReady) {
            throw new Error("Scanner not ready");
        }
        return await scannerRef.current.scan(source, scanOptions);
    }, [isReady]);

    return { scan, isReady, isLoading, error };
}
