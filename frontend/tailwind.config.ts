import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/app/**/*.{ts,tsx}",
    "./src/components/**/*.{ts,tsx}"
  ],
  theme: {
    extend: {
      boxShadow: {
        "plume-red": "0 0 28px rgba(248, 113, 113, 0.42)",
        "plume-green": "0 0 24px rgba(16, 185, 129, 0.38)"
      }
    }
  },
  plugins: []
};

export default config;
