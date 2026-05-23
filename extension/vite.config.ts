import { defineConfig } from 'vite';
import path from 'path';
import { copyFileSync, mkdirSync, readFileSync, writeFileSync } from 'fs';

export default defineConfig({
  build: {
    outDir: 'dist',
    rollupOptions: {
      input: {
        popup: path.resolve(__dirname, 'src/popup/index.html'),
        content: path.resolve(__dirname, 'src/content/index.ts'),
        background: path.resolve(__dirname, 'src/background/index.ts'),
      },
      output: {
        entryFileNames: '[name].js',
      },
    },
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  plugins: [
    {
      name: 'copy-manifest-and-assets',
      writeBundle() {
        mkdirSync('dist', { recursive: true });
        copyFileSync('manifest.json', 'dist/manifest.json');
        
        // Copy popup.html and update script reference
        let htmlContent = readFileSync('src/popup/index.html', 'utf-8');
        htmlContent = htmlContent.replace('./index.ts', './popup.js');
        writeFileSync('dist/popup.html', htmlContent);
      },
    },
  ],
});