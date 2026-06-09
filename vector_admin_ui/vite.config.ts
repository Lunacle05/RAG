import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "0.0.0.0",
    port: 6008,
    // AutoDL / SeetaCloud 公网域名访问 dev server 时需要放行
    allowedHosts: [".seetacloud.com", "localhost", "127.0.0.1"],
    // 浏览器只暴露 6008 时，将 /admin 代理到本机 FastAPI（6006）
    proxy: {
      "/admin": {
        target: "http://127.0.0.1:6006",
        changeOrigin: true
      }
    }
  }
});
