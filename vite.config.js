import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Add this to your vite.config.js
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:5000',
        changeOrigin: true,
        secure: false
      }
    }
  }
});