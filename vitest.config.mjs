import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    environment: "jsdom",
    include: ["tests/js/**/*.test.mjs"],
    globals: false,
    reporters: "default",
    coverage: {
      provider: "v8",
      reporter: ["text", "html"],
      include: ["scribe/static/js/**/*.mjs"],
    },
  },
});
